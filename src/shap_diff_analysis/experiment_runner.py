"""End-to-end execution and durable outputs for paper experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, cast

import pandas as pd

from .baselines import ALL_METHODS, RANKING_METHODS, rank_baseline_features
from .experiment_types import DEFAULT_EQP_MODE, EQP_MODES, DatasetBundle, EqpMode
from .gads import DEFAULT_GADS_BINS, DEFAULT_GADS_EPSILON, GADS_DISTANCES, rank_gads_features
from .generate_data import N_DEVICES, generate_dataset_bundle
from .metrics import add_result_context, evaluate_rankings, evaluate_rankings_per_equipment
from .modeling import ModelType, fit_cross_fitted_model
from .protocol_filters import PoolKpiFilterSpec, apply_pool_kpi_filter_to_bundle, pool_shap_std_floor
from .scenario import normalize_scenario
from .secom import (
    DEFAULT_RAW_DIR,
    HARMLESS_SHIFT_MAGNITUDE,
    MECHANISM_LOCAL_WEIGHT,
    PHYSICAL_SHIFT_MAGNITUDE,
    SecoMDataError,
    generate_dataset_b,
    verify_secom_files,
)

DEFAULT_MATRIX_SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4-main", "S4-null")


@dataclass(frozen=True)
class DatasetGenerationParams:
    """Generation parameters threaded into on-the-fly dataset bundle creation.

    Defaults reproduce the historical protocol exactly; they are recorded in
    every run manifest so each bundle can be regenerated bit-for-bit.
    """

    dataset_a_n_samples: int = 1_000
    dataset_a_noise_level: float = 0.2
    dataset_a_n_devices: int = N_DEVICES
    dataset_a_distribution_shift: float = 3.0
    dataset_a_mechanism_scale: float = 1.0
    dataset_a_harmless_drift: float = 3.5
    dataset_b_raw_dir: Path | None = None
    dataset_b_group_seed: int = 2024
    dataset_b_physical_shift: float = PHYSICAL_SHIFT_MAGNITUDE
    dataset_b_mechanism_weight: float = MECHANISM_LOCAL_WEIGHT
    dataset_b_harmless_shift: float = HARMLESS_SHIFT_MAGNITUDE
    dataset_b_include_missing_indicators: bool = True


def _generation_params_record(params: DatasetGenerationParams) -> dict[str, Any]:
    """Return a JSON-serialisable record of every generation parameter."""
    record = asdict(params)
    record["dataset_b_raw_dir"] = str(params.dataset_b_raw_dir) if params.dataset_b_raw_dir is not None else None
    return record


@dataclass(frozen=True)
class ExperimentSettings:
    """Model and ranking controls recorded in every experiment manifest.

    ``shap_std_floor_percentile`` is a protocol-level ranking control: when
    set, process features whose normal-pool OOF SHAP standard deviation is
    below this percentile are excluded from every method's ranking (the KPI
    model itself still sees all bundle process features).

    The ``m2oe_*`` and ``xpe_*`` fields configure the modern baselines
    (``M2OE-Group``, ``XPE``) without affecting any legacy method; the
    defaults follow the M2OE upstream hyperparameters and a fixed-seed
    permutation-Shapley budget for XPE. ``xpe_max_rows`` is an optional
    ``(target, source)`` row-cap pair applied before XPE's optimal-transport
    solve: seeded, KPI-label-stratified subsampling of the device rows and
    the normal pool (``None`` disables capping entirely; every cap and the
    ``xpe_seed`` that drives the subsampling is recorded in the manifest via
    ``settings``).
    """

    methods: tuple[str, ...] = ALL_METHODS
    model_type: ModelType = "xgboost"
    model_seed: int = 42
    cv_seed: int | None = None
    cv_splits: int = 5
    model_params: dict[str, Any] | None = None
    top_ks: tuple[int, ...] = (1, 3, 5)
    auc_threshold: float | None = None
    psi_bins: int = 10
    gads_distance: str = "wasserstein"
    gads_bins: int = DEFAULT_GADS_BINS
    gads_epsilon: float = DEFAULT_GADS_EPSILON
    eqp_mode: EqpMode = DEFAULT_EQP_MODE
    shap_std_floor_percentile: float | None = None
    m2oe_seed: int = 42
    m2oe_epochs: int = 30
    m2oe_n_neighbors: int = 30
    m2oe_batch_size: int = 16
    m2oe_max_group_rows: int | None = 128
    xpe_seed: int = 42
    xpe_permutations: int = 8
    xpe_max_rows: tuple[int, int] | None = None


@dataclass(frozen=True)
class ExperimentRunResult:
    """Long-form outputs for one dataset/scenario/seed run."""

    rankings: pd.DataFrame
    metrics: pd.DataFrame
    equipment_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    manifest: dict[str, Any]


def git_commit() -> str | None:
    """Return the current git commit hash, or None outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dataset_name(bundle: DatasetBundle) -> str:
    value = bundle.metadata.get("dataset") or bundle.metadata.get("dataset_name")
    return str(value) if value is not None else "dataset_a"


def _validate_bundle(bundle: DatasetBundle) -> None:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    required = {bundle.group_column, bundle.target_column, *bundle.process_features}
    missing = required - set(bundle.data.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    if set(bundle.abnormal_devices) & set(bundle.normal_devices):
        raise ValueError("abnormal and normal device pools must be disjoint")
    groups = set(bundle.data[bundle.group_column].astype(str).unique())
    expected = set(bundle.abnormal_devices) | set(bundle.normal_devices)
    if not expected <= groups:
        raise ValueError(f"bundle device pools reference missing devices: {sorted(expected - groups)}")


def _validate_eqp_mode(eqp_mode: str) -> EqpMode:
    if eqp_mode not in EQP_MODES:
        raise ValueError(f"unsupported eqp_mode {eqp_mode!r}; expected one of: {EQP_MODES}")
    return eqp_mode


def _model_feature_columns(bundle: DatasetBundle, eqp_mode: EqpMode) -> list[str]:
    if eqp_mode == "no_eqp":
        return list(bundle.process_features)
    columns = [*bundle.process_features, bundle.group_column]
    if len(columns) != len(set(columns)):
        raise ValueError("group column must not also be listed as a process feature")
    return columns


def _metric_ranking_view(eqp_mode: EqpMode) -> str:
    return "all" if eqp_mode == "all_features" else "process"


def _gads_views(eqp_mode: EqpMode) -> tuple[str, ...]:
    if eqp_mode == "no_eqp":
        return ("process",)
    return ("process", "all", "residual")


def run_experiment_bundle(
    bundle: DatasetBundle,
    *,
    settings: ExperimentSettings | None = None,
) -> ExperimentRunResult:
    """Run the KPI model, GADS views, baselines and ranking metrics."""
    _validate_bundle(bundle)
    settings = ExperimentSettings() if settings is None else settings
    unknown_methods = set(settings.methods) - set(RANKING_METHODS)
    if unknown_methods:
        raise ValueError(f"unsupported methods: {sorted(unknown_methods)}")
    if settings.xpe_max_rows is not None and (len(settings.xpe_max_rows) != 2 or min(settings.xpe_max_rows) < 1):
        raise ValueError("xpe_max_rows must be a (target, source) pair of positive integers when provided")
    if settings.gads_distance not in GADS_DISTANCES:
        raise ValueError(f"unsupported gads_distance {settings.gads_distance!r}; expected one of: {GADS_DISTANCES}")
    eqp_mode = _validate_eqp_mode(settings.eqp_mode)

    feature_columns = _model_feature_columns(bundle, eqp_mode)
    X = bundle.data.loc[:, feature_columns]
    y = cast(pd.Series, pd.to_numeric(bundle.data[bundle.target_column], errors="raise")).astype(int)
    groups = cast(pd.Series, bundle.data[bundle.group_column]).astype(str)
    attribution = fit_cross_fitted_model(
        X,
        y,
        model_type=settings.model_type,
        model_seed=settings.model_seed,
        cv_seed=settings.cv_seed,
        cv_splits=settings.cv_splits,
        model_params=settings.model_params,
    )
    status = "valid"
    if settings.auc_threshold is not None and attribution.mean_auc < settings.auc_threshold:
        status = "invalid_auc"

    ranking_process_features = bundle.process_features
    protocol_filter_records: dict[str, Any] = dict(bundle.metadata.get("protocol_filters", {}))
    if settings.shap_std_floor_percentile is not None:
        ranking_process_features, floor_record = pool_shap_std_floor(
            attribution.shap_values,
            groups,
            bundle.normal_devices,
            bundle.process_features,
            settings.shap_std_floor_percentile,
        )
        protocol_filter_records["shap_std_floor"] = floor_record

    ranking_frames: list[pd.DataFrame] = []
    metric_view = _metric_ranking_view(eqp_mode)
    if "GADS" in settings.methods:
        ranking_frames.extend(
            rank_gads_features(
                attribution.shap_values,
                groups,
                bundle.abnormal_devices,
                bundle.normal_devices,
                process_features=ranking_process_features,
                view=view,
                distance=settings.gads_distance,
                bins=settings.gads_bins,
                epsilon=settings.gads_epsilon,
            )
            for view in _gads_views(eqp_mode)
        )
    for method in settings.methods:
        if method == "GADS":
            continue
        kwargs: dict[str, Any] = {}
        if method == "PSI-X":
            kwargs["bins"] = settings.psi_bins
        if method == "Domain-SHAP":
            kwargs.update(
                {
                    "model_type": settings.model_type,
                    "model_seed": settings.model_seed,
                    "cv_seed": settings.cv_seed,
                    "cv_splits": settings.cv_splits,
                    "model_params": settings.model_params,
                    "include_eqp_in_classifier": eqp_mode == "all_features",
                }
            )
        if method == "M2OE-Group":
            kwargs.update(
                {
                    "seed": settings.m2oe_seed,
                    "epochs": settings.m2oe_epochs,
                    "n_neighbors": settings.m2oe_n_neighbors,
                    "batch_size": settings.m2oe_batch_size,
                    "max_group_rows": settings.m2oe_max_group_rows,
                }
            )
        if method == "XPE":
            kwargs.update(
                {
                    "seed": settings.xpe_seed,
                    "permutations": settings.xpe_permutations,
                    "max_target_rows": settings.xpe_max_rows[0] if settings.xpe_max_rows is not None else None,
                    "max_source_rows": settings.xpe_max_rows[1] if settings.xpe_max_rows is not None else None,
                }
            )
        ranking_frames.append(
            rank_baseline_features(
                method,
                X=X,
                shap_values=attribution.shap_values,
                y=y,
                attribution=attribution,
                groups=groups,
                abnormal_devices=bundle.abnormal_devices,
                normal_devices=bundle.normal_devices,
                process_features=ranking_process_features,
                view=metric_view,
                **kwargs,
            )
        )
    if not ranking_frames:
        raise ValueError("settings.methods must contain at least one method")

    dataset = _dataset_name(bundle)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    rankings.insert(0, "dataset", dataset)
    rankings.insert(1, "scenario", bundle.scenario)
    rankings.insert(2, "seed", bundle.seed)
    rankings["is_root_cause"] = [
        feature in set(bundle.root_causes.get(str(device), ()))
        for device, feature in zip(rankings["equipment"], rankings["feature"], strict=True)
    ]
    rankings["is_harmless"] = rankings["feature"].isin(bundle.harmless_features)
    rankings["eqp_mode"] = eqp_mode

    metric_rankings = rankings.loc[rankings["view"] == metric_view]
    metrics = evaluate_rankings(
        metric_rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=settings.top_ks,
    )
    metrics = add_result_context(metrics, dataset=dataset, scenario=bundle.scenario, seed=bundle.seed)
    metrics["kpi_auc"] = attribution.mean_auc
    metrics["status"] = status
    metrics["eqp_mode"] = eqp_mode
    equipment_metrics = evaluate_rankings_per_equipment(
        metric_rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=settings.top_ks,
    )
    equipment_metrics = add_result_context(
        equipment_metrics,
        dataset=dataset,
        scenario=bundle.scenario,
        seed=bundle.seed,
    )
    equipment_metrics["kpi_auc"] = attribution.mean_auc
    equipment_metrics["status"] = status
    equipment_metrics["eqp_mode"] = eqp_mode
    fold_metrics = attribution.fold_metrics.copy()
    fold_metrics.insert(0, "dataset", dataset)
    fold_metrics.insert(1, "scenario", bundle.scenario)
    fold_metrics.insert(2, "seed", bundle.seed)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "dataset": dataset,
        "bundle": bundle.manifest(),
        # Both bundled generators record their full parameter set under
        # ``metadata["generation_params"]``; manually constructed bundles may
        # legitimately omit it, in which case the record stays empty.
        "generation_params": dict(bundle.metadata.get("generation_params", {})),
        "protocol_filters": protocol_filter_records,
        "settings": asdict(settings),
        "model": attribution.metadata,
        "status": status,
    }
    return ExperimentRunResult(rankings, metrics, equipment_metrics, fold_metrics, manifest)


def _gads_distance_method_label(distance: str) -> str:
    return f"GADS-{distance}"


def _tag_gads_distance_ablation_result(result: ExperimentRunResult, distance: str) -> ExperimentRunResult:
    """Relabel GADS rows for multi-distance ablation with an explicit distance column."""
    method = _gads_distance_method_label(distance)
    rankings = result.rankings.copy()
    rankings["method"] = method
    rankings["gads_distance"] = distance
    return ExperimentRunResult(
        rankings=rankings,
        metrics=result.metrics.assign(method=method, gads_distance=distance),
        equipment_metrics=result.equipment_metrics.assign(method=method, gads_distance=distance),
        fold_metrics=result.fold_metrics,
        manifest={**result.manifest, "gads_distance": distance},
    )


@dataclass(frozen=True)
class MatrixRunSelection:
    """Scenario/seed slice taken from one partial experiment-matrix directory."""

    source_dir: Path
    scenarios: tuple[str, ...]
    seeds: tuple[int, ...]


def load_experiment_run_result(run_dir: str | Path) -> ExperimentRunResult:
    """Load durable outputs for one matrix run directory."""
    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("rankings", "metrics", "equipment_metrics", "fold_metrics")
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        parquet_path = directory / f"{name}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"run output not found: {parquet_path}")
        frames[name] = pd.read_parquet(parquet_path)
    return ExperimentRunResult(
        rankings=frames["rankings"],
        metrics=frames["metrics"],
        equipment_metrics=frames["equipment_metrics"],
        fold_metrics=frames["fold_metrics"],
        manifest=manifest,
    )


def assemble_experiment_matrix(
    *,
    output_dir: str | Path,
    selections: Sequence[MatrixRunSelection],
    dataset: str = "dataset_b",
    exclude_invalid_auc: bool = True,
) -> dict[str, Any]:
    """Merge partial matrix runs into one canonical output with explicit deduplication."""
    if not selections:
        raise ValueError("at least one MatrixRunSelection is required")
    root = Path(output_dir)
    runs_dir = root / "runs"
    aggregate_dir = root / "aggregate"
    runs_dir.mkdir(parents=True, exist_ok=True)

    seen_keys: dict[tuple[str, int], Path] = {}
    all_results: list[ExperimentRunResult] = []
    run_records: list[dict[str, Any]] = []

    for selection in selections:
        source_root = Path(selection.source_dir)
        source_runs = source_root / "runs"
        if not source_runs.is_dir():
            raise FileNotFoundError(f"source runs directory not found: {source_runs}")
        for scenario in selection.scenarios:
            canonical = normalize_scenario(scenario)
            for seed in selection.seeds:
                run_id = _matrix_run_id(dataset, canonical, int(seed))
                key = (canonical, int(seed))
                source_run_dir = source_runs / run_id
                if not source_run_dir.is_dir():
                    raise FileNotFoundError(f"selected run not found: {source_run_dir}")
                if key in seen_keys:
                    raise ValueError(
                        f"duplicate (scenario, seed) selection rejected: {canonical} seed={seed}; "
                        f"already taken from {seen_keys[key]}"
                    )
                seen_keys[key] = source_run_dir
                destination_run_dir = runs_dir / run_id
                if destination_run_dir.exists():
                    shutil.rmtree(destination_run_dir)
                shutil.copytree(source_run_dir, destination_run_dir)
                result = load_experiment_run_result(destination_run_dir)
                all_results.append(result)
                run_records.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "scenario": canonical,
                        "seed": int(seed),
                        "status": result.manifest["status"],
                        "source_dir": str(source_run_dir),
                    }
                )

    aggregate_source = filter_valid_results(all_results) if exclude_invalid_auc else list(all_results)
    if not aggregate_source:
        raise ValueError("no valid runs remain after excluding status=invalid_auc")
    combined = combine_run_results(aggregate_source)
    aggregate_paths = write_experiment_outputs(combined, aggregate_dir)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "assembled_from": [
            {
                "source_dir": str(selection.source_dir),
                "scenarios": [normalize_scenario(scenario) for scenario in selection.scenarios],
                "seeds": [int(seed) for seed in selection.seeds],
            }
            for selection in selections
        ],
        "dataset": dataset,
        "exclude_invalid_auc": exclude_invalid_auc,
        "n_runs_total": len(all_results),
        "n_runs_valid": len(aggregate_source),
        "n_runs_invalid_auc": len(all_results) - len(filter_valid_results(all_results)),
        "n_unique_keys": len(seen_keys),
        "runs": run_records,
        "aggregate": {name: str(path) for name, path in aggregate_paths.items()},
    }
    summary_path = root / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def combine_run_results(results: Iterable[ExperimentRunResult]) -> ExperimentRunResult:
    """Combine independent run outputs without changing their measurements."""
    result_list = list(results)
    if not result_list:
        raise ValueError("at least one run result is required")
    return ExperimentRunResult(
        rankings=pd.concat([result.rankings for result in result_list], ignore_index=True),
        metrics=pd.concat([result.metrics for result in result_list], ignore_index=True),
        equipment_metrics=pd.concat([result.equipment_metrics for result in result_list], ignore_index=True),
        fold_metrics=pd.concat([result.fold_metrics for result in result_list], ignore_index=True),
        manifest={"runs": [result.manifest for result in result_list]},
    )


def write_experiment_outputs(result: ExperimentRunResult, output_dir: str | Path) -> dict[str, Path]:
    """Write Parquet, CSV, JSON and a table-ready LaTeX summary."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "rankings": result.rankings,
        "metrics": result.metrics,
        "equipment_metrics": result.equipment_metrics,
        "fold_metrics": result.fold_metrics,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        parquet_path = directory / f"{name}.parquet"
        csv_path = directory / f"{name}.csv"
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)
        paths[f"{name}_parquet"] = parquet_path
        paths[f"{name}_csv"] = csv_path
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path

    table_columns = [
        column
        for column in [
            "dataset",
            "scenario",
            "eqp_mode",
            "method",
            "gads_distance",
            "mrr",
            "hr_at_1",
            "hr_at_3",
            "hr_at_5",
            "far_at_1",
            "far_at_5",
        ]
        if column in result.metrics.columns
    ]
    latex_path = directory / "metrics_table.tex"
    latex_path.write_text(
        result.metrics.loc[:, table_columns].to_latex(index=False, float_format="%.4f", na_rep="--"),
        encoding="utf-8",
    )
    paths["metrics_latex"] = latex_path
    return paths


def load_dataset_bundle(data_path: Path, manifest_path: Path) -> DatasetBundle:
    """Load a generated dataset table and its JSON manifest."""
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_path.suffix.lower() == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix.lower() in {".csv", ".tsv"}:
        data = pd.read_csv(data_path, sep="\t" if data_path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError("data_path must be a .parquet, .csv, or .tsv file")
    return DatasetBundle(
        data=data,
        scenario=str(manifest["scenario"]),
        seed=int(manifest["seed"]),
        group_column=str(manifest["group_column"]),
        target_column=str(manifest["target_column"]),
        process_features=tuple(manifest["process_features"]),
        abnormal_devices=tuple(manifest["abnormal_devices"]),
        normal_devices=tuple(manifest["normal_devices"]),
        root_causes={str(k): tuple(v) for k, v in manifest.get("root_causes", {}).items()},
        harmless_features=tuple(manifest.get("harmless_features", ())),
        metadata=dict(manifest.get("metadata", {})),
    )


def write_dataset_bundle(bundle: DatasetBundle, output_dir: str | Path, *, stem: str) -> dict[str, Path]:
    """Persist a generated dataset table and evaluation manifest."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{stem}.parquet"
    manifest_path = directory / f"{stem}.manifest.json"
    bundle.data.to_parquet(data_path, index=False)
    manifest_payload = bundle.manifest()
    if bundle.metadata:
        manifest_payload["metadata"] = bundle.metadata
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"data": data_path, "manifest": manifest_path}


def filter_valid_results(results: Iterable[ExperimentRunResult]) -> list[ExperimentRunResult]:
    """Drop runs whose manifest status is ``invalid_auc``."""
    return [result for result in results if result.manifest.get("status") != "invalid_auc"]


def combine_valid_run_results(results: Iterable[ExperimentRunResult]) -> ExperimentRunResult:
    """Combine only runs that passed the AUC gate."""
    valid = filter_valid_results(results)
    if not valid:
        raise ValueError("no valid runs remain after excluding status=invalid_auc")
    return combine_run_results(valid)


def is_secom_raw_available(raw_dir: Path | None = None) -> bool:
    """Return whether official SECOM raw files are present and verified."""
    directory = raw_dir or DEFAULT_RAW_DIR
    try:
        verify_secom_files(directory)
    except SecoMDataError:
        return False
    return True


def _matrix_run_id(dataset: str, scenario: str, seed: int) -> str:
    return f"{dataset}_{scenario}_seed{seed}"


def _generate_matrix_bundle(
    *,
    dataset: str,
    scenario: str,
    seed: int,
    generation: DatasetGenerationParams,
) -> DatasetBundle:
    canonical = normalize_scenario(scenario)
    if dataset == "dataset_a":
        return generate_dataset_bundle(
            scenario=canonical,
            n_samples=generation.dataset_a_n_samples,
            seed=seed,
            noise_level=generation.dataset_a_noise_level,
            n_devices=generation.dataset_a_n_devices,
            distribution_shift=generation.dataset_a_distribution_shift,
            mechanism_scale=generation.dataset_a_mechanism_scale,
            harmless_drift=generation.dataset_a_harmless_drift,
        )
    if dataset == "dataset_b":
        bundle = generate_dataset_b(
            scenario=canonical,  # type: ignore[arg-type]
            seed=seed,
            group_seed=generation.dataset_b_group_seed,
            raw_dir=generation.dataset_b_raw_dir,
            physical_shift_magnitude=generation.dataset_b_physical_shift,
            mechanism_local_weight=generation.dataset_b_mechanism_weight,
            harmless_shift_magnitude=generation.dataset_b_harmless_shift,
            include_missing_indicators=generation.dataset_b_include_missing_indicators,
        )
        metadata = dict(bundle.metadata)
        metadata.setdefault("dataset", "dataset_b")
        return DatasetBundle(
            data=bundle.data,
            scenario=bundle.scenario,
            seed=bundle.seed,
            group_column=bundle.group_column,
            target_column=bundle.target_column,
            process_features=bundle.process_features,
            abnormal_devices=bundle.abnormal_devices,
            normal_devices=bundle.normal_devices,
            root_causes=bundle.root_causes,
            harmless_features=bundle.harmless_features,
            metadata=metadata,
        )
    raise ValueError(f"unsupported dataset {dataset!r}; expected 'dataset_a' or 'dataset_b'")


def run_experiment_matrix(
    *,
    output_dir: str | Path,
    scenarios: Sequence[str] = DEFAULT_MATRIX_SCENARIOS,
    seeds: Sequence[int] = (42,),
    settings: ExperimentSettings | None = None,
    datasets: Sequence[str] = ("dataset_a",),
    dataset_a_n_samples: int = 1_000,
    dataset_a_noise_level: float = 0.2,
    dataset_a_n_devices: int = N_DEVICES,
    dataset_a_distribution_shift: float = 3.0,
    dataset_a_mechanism_scale: float = 1.0,
    dataset_a_harmless_drift: float = 3.5,
    dataset_b_raw_dir: Path | None = None,
    dataset_b_group_seed: int = 2024,
    dataset_b_physical_shift: float = PHYSICAL_SHIFT_MAGNITUDE,
    dataset_b_mechanism_weight: float = MECHANISM_LOCAL_WEIGHT,
    dataset_b_harmless_shift: float = HARMLESS_SHIFT_MAGNITUDE,
    dataset_b_include_missing_indicators: bool = True,
    pool_feature_filter: PoolKpiFilterSpec | None = None,
    exclude_invalid_auc: bool = True,
) -> dict[str, Any]:
    """Run the paper matrix and aggregate durable outputs.

    ``pool_feature_filter`` applies the normal-pool KPI-association pre-filter
    to every generated bundle before the experiment runs; the resulting
    feature set (recorded in each run manifest under
    ``protocol_filters.pool_kpi_association``) defines the KPI-model input and
    every method's ranking surface.
    """
    settings = ExperimentSettings() if settings is None else settings
    generation = DatasetGenerationParams(
        dataset_a_n_samples=dataset_a_n_samples,
        dataset_a_noise_level=dataset_a_noise_level,
        dataset_a_n_devices=dataset_a_n_devices,
        dataset_a_distribution_shift=dataset_a_distribution_shift,
        dataset_a_mechanism_scale=dataset_a_mechanism_scale,
        dataset_a_harmless_drift=dataset_a_harmless_drift,
        dataset_b_raw_dir=dataset_b_raw_dir,
        dataset_b_group_seed=dataset_b_group_seed,
        dataset_b_physical_shift=dataset_b_physical_shift,
        dataset_b_mechanism_weight=dataset_b_mechanism_weight,
        dataset_b_harmless_shift=dataset_b_harmless_shift,
        dataset_b_include_missing_indicators=dataset_b_include_missing_indicators,
    )
    root = Path(output_dir)
    runs_dir = root / "runs"
    aggregate_dir = root / "aggregate"
    all_results: list[ExperimentRunResult] = []
    skipped: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    requested_datasets = tuple(dict.fromkeys(datasets))
    for dataset in requested_datasets:
        if dataset == "dataset_b" and not is_secom_raw_available(dataset_b_raw_dir):
            skipped.append(
                {
                    "dataset": dataset,
                    "reason": "secom_raw_unavailable",
                    "raw_dir": str(dataset_b_raw_dir or DEFAULT_RAW_DIR),
                }
            )
            continue
        for scenario in scenarios:
            canonical = normalize_scenario(scenario)
            for seed in seeds:
                run_id = _matrix_run_id(dataset, canonical, int(seed))
                run_dir = runs_dir / run_id
                bundle = _generate_matrix_bundle(
                    dataset=dataset,
                    scenario=canonical,
                    seed=int(seed),
                    generation=generation,
                )
                if pool_feature_filter is not None:
                    bundle = apply_pool_kpi_filter_to_bundle(bundle, pool_feature_filter)
                bundle_paths = write_dataset_bundle(bundle, run_dir, stem="bundle")
                result = run_experiment_bundle(bundle, settings=settings)
                output_paths = write_experiment_outputs(result, run_dir)
                all_results.append(result)
                run_records.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "scenario": canonical,
                        "seed": int(seed),
                        "status": result.manifest["status"],
                        "bundle_data": str(bundle_paths["data"]),
                        "bundle_manifest": str(bundle_paths["manifest"]),
                        "outputs": {name: str(path) for name, path in output_paths.items()},
                    }
                )

    if not all_results:
        raise ValueError("experiment matrix produced no runs")

    aggregate_source = filter_valid_results(all_results) if exclude_invalid_auc else list(all_results)
    if not aggregate_source:
        raise ValueError("no valid runs remain after excluding status=invalid_auc")
    combined = combine_run_results(aggregate_source)
    aggregate_paths = write_experiment_outputs(combined, aggregate_dir)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "settings": asdict(settings),
        "generation_params": _generation_params_record(generation),
        "pool_feature_filter": asdict(pool_feature_filter) if pool_feature_filter is not None else None,
        "datasets_requested": list(requested_datasets),
        "scenarios": [normalize_scenario(scenario) for scenario in scenarios],
        "seeds": [int(seed) for seed in seeds],
        "exclude_invalid_auc": exclude_invalid_auc,
        "n_runs_total": len(all_results),
        "n_runs_valid": len(aggregate_source),
        "n_runs_invalid_auc": len(all_results) - len(filter_valid_results(all_results)),
        "skipped_datasets": skipped,
        "runs": run_records,
        "aggregate": {name: str(path) for name, path in aggregate_paths.items()},
    }
    summary_path = root / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_gads_distance_ablation(
    bundle: DatasetBundle,
    *,
    distances: Sequence[str] = GADS_DISTANCES,
    settings: ExperimentSettings | None = None,
) -> ExperimentRunResult:
    """Run GADS-only rankings for multiple distance functions on one bundle."""
    base = ExperimentSettings() if settings is None else settings
    results: list[ExperimentRunResult] = []
    for distance in distances:
        if distance not in GADS_DISTANCES:
            raise ValueError(f"unsupported distance {distance!r}; expected one of: {GADS_DISTANCES}")
        run_settings = ExperimentSettings(
            methods=("GADS",),
            model_type=base.model_type,
            model_seed=base.model_seed,
            cv_seed=base.cv_seed,
            cv_splits=base.cv_splits,
            model_params=base.model_params,
            top_ks=base.top_ks,
            auc_threshold=base.auc_threshold,
            psi_bins=base.psi_bins,
            gads_distance=distance,
            gads_bins=base.gads_bins,
            gads_epsilon=base.gads_epsilon,
            eqp_mode=base.eqp_mode,
        )
        result = run_experiment_bundle(bundle, settings=run_settings)
        results.append(_tag_gads_distance_ablation_result(result, distance))
    return combine_run_results(results)


def run_gads_distance_ablation_matrix(
    *,
    output_dir: str | Path,
    scenarios: Sequence[str] = DEFAULT_MATRIX_SCENARIOS,
    seeds: Sequence[int] = (42,),
    settings: ExperimentSettings | None = None,
    distances: Sequence[str] = GADS_DISTANCES,
    datasets: Sequence[str] = ("dataset_a",),
    dataset_a_n_samples: int = 1_000,
    dataset_b_raw_dir: Path | None = None,
    dataset_b_group_seed: int = 2024,
    exclude_invalid_auc: bool = True,
) -> dict[str, Any]:
    """Run the GADS distance ablation across dataset/scenario/seed combinations."""
    settings = ExperimentSettings() if settings is None else settings
    validated_distances = tuple(distances)
    for distance in validated_distances:
        if distance not in GADS_DISTANCES:
            raise ValueError(f"unsupported distance {distance!r}; expected one of: {GADS_DISTANCES}")
    generation = DatasetGenerationParams(
        dataset_a_n_samples=dataset_a_n_samples,
        dataset_b_raw_dir=dataset_b_raw_dir,
        dataset_b_group_seed=dataset_b_group_seed,
    )
    root = Path(output_dir)
    runs_dir = root / "runs"
    aggregate_dir = root / "aggregate"
    all_results: list[ExperimentRunResult] = []
    skipped: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    requested_datasets = tuple(dict.fromkeys(datasets))
    for dataset in requested_datasets:
        if dataset == "dataset_b" and not is_secom_raw_available(dataset_b_raw_dir):
            skipped.append(
                {
                    "dataset": dataset,
                    "reason": "secom_raw_unavailable",
                    "raw_dir": str(dataset_b_raw_dir or DEFAULT_RAW_DIR),
                }
            )
            continue
        for scenario in scenarios:
            canonical = normalize_scenario(scenario)
            for seed in seeds:
                run_id = _matrix_run_id(dataset, canonical, int(seed))
                run_dir = runs_dir / run_id
                bundle = _generate_matrix_bundle(
                    dataset=dataset,
                    scenario=canonical,
                    seed=int(seed),
                    generation=generation,
                )
                bundle_paths = write_dataset_bundle(bundle, run_dir, stem="bundle")
                distance_results: list[ExperimentRunResult] = []
                for distance in validated_distances:
                    run_settings = ExperimentSettings(
                        methods=("GADS",),
                        model_type=settings.model_type,
                        model_seed=settings.model_seed,
                        cv_seed=settings.cv_seed,
                        cv_splits=settings.cv_splits,
                        model_params=settings.model_params,
                        top_ks=settings.top_ks,
                        auc_threshold=settings.auc_threshold,
                        psi_bins=settings.psi_bins,
                        gads_distance=distance,
                        gads_bins=settings.gads_bins,
                        gads_epsilon=settings.gads_epsilon,
                        eqp_mode=settings.eqp_mode,
                    )
                    result = run_experiment_bundle(bundle, settings=run_settings)
                    distance_results.append(_tag_gads_distance_ablation_result(result, distance))
                result = combine_run_results(distance_results)
                all_results.extend(distance_results)
                output_paths = write_experiment_outputs(result, run_dir)
                run_records.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "scenario": canonical,
                        "seed": int(seed),
                        "status_by_distance": {
                            str(distance_result.manifest["gads_distance"]): distance_result.manifest["status"]
                            for distance_result in distance_results
                        },
                        "distances": list(validated_distances),
                        "bundle_data": str(bundle_paths["data"]),
                        "bundle_manifest": str(bundle_paths["manifest"]),
                        "outputs": {name: str(path) for name, path in output_paths.items()},
                    }
                )

    if not all_results:
        raise ValueError("GADS distance ablation matrix produced no runs")

    aggregate_source = filter_valid_results(all_results) if exclude_invalid_auc else list(all_results)
    if not aggregate_source:
        raise ValueError("no valid runs remain after excluding status=invalid_auc")
    combined = combine_run_results(aggregate_source)
    aggregate_paths = write_experiment_outputs(combined, aggregate_dir)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "settings": asdict(settings),
        "distances": list(validated_distances),
        "datasets_requested": list(requested_datasets),
        "scenarios": [normalize_scenario(scenario) for scenario in scenarios],
        "seeds": [int(seed) for seed in seeds],
        "exclude_invalid_auc": exclude_invalid_auc,
        "n_runs_total": len(all_results),
        "n_runs_valid": len(aggregate_source),
        "n_runs_invalid_auc": len(all_results) - len(filter_valid_results(all_results)),
        "skipped_datasets": skipped,
        "runs": run_records,
        "aggregate": {name: str(path) for name, path in aggregate_paths.items()},
    }
    summary_path = root / "distance_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_eqp_ablation(
    bundle: DatasetBundle,
    *,
    modes: Sequence[EqpMode] = EQP_MODES,
    settings: ExperimentSettings | None = None,
) -> ExperimentRunResult:
    """Compare contextual, no-EQP and all-feature ranking modes on one bundle."""
    base = ExperimentSettings() if settings is None else settings
    results: list[ExperimentRunResult] = []
    for mode in modes:
        selected = _validate_eqp_mode(mode)
        run_settings = ExperimentSettings(
            methods=base.methods,
            model_type=base.model_type,
            model_seed=base.model_seed,
            cv_seed=base.cv_seed,
            cv_splits=base.cv_splits,
            model_params=base.model_params,
            top_ks=base.top_ks,
            auc_threshold=base.auc_threshold,
            psi_bins=base.psi_bins,
            gads_distance=base.gads_distance,
            gads_bins=base.gads_bins,
            gads_epsilon=base.gads_epsilon,
            eqp_mode=selected,
        )
        results.append(run_experiment_bundle(bundle, settings=run_settings))
    return combine_run_results(results)


def run_eqp_ablation_matrix(
    *,
    output_dir: str | Path,
    scenarios: Sequence[str] = DEFAULT_MATRIX_SCENARIOS,
    seeds: Sequence[int] = (42,),
    settings: ExperimentSettings | None = None,
    modes: Sequence[EqpMode] = EQP_MODES,
    datasets: Sequence[str] = ("dataset_a",),
    dataset_a_n_samples: int = 1_000,
    dataset_b_raw_dir: Path | None = None,
    dataset_b_group_seed: int = 2024,
    exclude_invalid_auc: bool = True,
) -> dict[str, Any]:
    """Run the EQP ablation across dataset/scenario/seed combinations."""
    settings = ExperimentSettings() if settings is None else settings
    validated_modes: tuple[EqpMode, ...] = tuple(_validate_eqp_mode(mode) for mode in modes)
    generation = DatasetGenerationParams(
        dataset_a_n_samples=dataset_a_n_samples,
        dataset_b_raw_dir=dataset_b_raw_dir,
        dataset_b_group_seed=dataset_b_group_seed,
    )
    root = Path(output_dir)
    runs_dir = root / "runs"
    aggregate_dir = root / "aggregate"
    all_results: list[ExperimentRunResult] = []
    skipped: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    requested_datasets = tuple(dict.fromkeys(datasets))
    for dataset in requested_datasets:
        if dataset == "dataset_b" and not is_secom_raw_available(dataset_b_raw_dir):
            skipped.append(
                {
                    "dataset": dataset,
                    "reason": "secom_raw_unavailable",
                    "raw_dir": str(dataset_b_raw_dir or DEFAULT_RAW_DIR),
                }
            )
            continue
        for scenario in scenarios:
            canonical = normalize_scenario(scenario)
            for seed in seeds:
                run_id = _matrix_run_id(dataset, canonical, int(seed))
                run_dir = runs_dir / run_id
                bundle = _generate_matrix_bundle(
                    dataset=dataset,
                    scenario=canonical,
                    seed=int(seed),
                    generation=generation,
                )
                bundle_paths = write_dataset_bundle(bundle, run_dir, stem="bundle")
                mode_results: list[ExperimentRunResult] = []
                for mode in validated_modes:
                    run_settings = ExperimentSettings(
                        methods=settings.methods,
                        model_type=settings.model_type,
                        model_seed=settings.model_seed,
                        cv_seed=settings.cv_seed,
                        cv_splits=settings.cv_splits,
                        model_params=settings.model_params,
                        top_ks=settings.top_ks,
                        auc_threshold=settings.auc_threshold,
                        psi_bins=settings.psi_bins,
                        gads_distance=settings.gads_distance,
                        gads_bins=settings.gads_bins,
                        gads_epsilon=settings.gads_epsilon,
                        eqp_mode=mode,
                    )
                    mode_results.append(run_experiment_bundle(bundle, settings=run_settings))
                result = combine_run_results(mode_results)
                all_results.extend(mode_results)
                output_paths = write_experiment_outputs(result, run_dir)
                run_records.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "scenario": canonical,
                        "seed": int(seed),
                        "status_by_eqp_mode": {
                            str(mode_result.manifest["settings"]["eqp_mode"]): mode_result.manifest["status"]
                            for mode_result in mode_results
                        },
                        "eqp_modes": list(validated_modes),
                        "bundle_data": str(bundle_paths["data"]),
                        "bundle_manifest": str(bundle_paths["manifest"]),
                        "outputs": {name: str(path) for name, path in output_paths.items()},
                    }
                )

    if not all_results:
        raise ValueError("EQP ablation matrix produced no runs")

    aggregate_source = filter_valid_results(all_results) if exclude_invalid_auc else list(all_results)
    if not aggregate_source:
        raise ValueError("no valid runs remain after excluding status=invalid_auc")
    combined = combine_run_results(aggregate_source)
    aggregate_paths = write_experiment_outputs(combined, aggregate_dir)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "settings": asdict(settings),
        "eqp_modes": list(validated_modes),
        "datasets_requested": list(requested_datasets),
        "scenarios": [normalize_scenario(scenario) for scenario in scenarios],
        "seeds": [int(seed) for seed in seeds],
        "exclude_invalid_auc": exclude_invalid_auc,
        "n_runs_total": len(all_results),
        "n_runs_valid": len(aggregate_source),
        "n_runs_invalid_auc": len(all_results) - len(filter_valid_results(all_results)),
        "skipped_datasets": skipped,
        "runs": run_records,
        "aggregate": {name: str(path) for name, path in aggregate_paths.items()},
    }
    summary_path = root / "eqp_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary

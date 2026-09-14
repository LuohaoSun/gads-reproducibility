#!/usr/bin/env python3
"""Single-factor sensitivity sweeps over Dataset A generation parameters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from shap_diff_analysis.experiment_runner import (
    ExperimentRunResult,
    ExperimentSettings,
    filter_valid_results,
    git_commit,
    run_experiment_bundle,
    write_dataset_bundle,
    write_experiment_outputs,
)
from shap_diff_analysis.gads import GADS_DISTANCES
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.scenario import normalize_scenario

# Default single-factor grids from the experiment plan (G3/G5); each level is
# applied while every other generation parameter stays at its base value.
DEFAULT_SWEEP_LEVELS: dict[str, tuple[float | int, ...]] = {
    "noise_level": (0.1, 0.2, 0.4, 0.8),
    "n_samples": (1_000, 2_500, 5_000, 10_000),
    "n_devices": (10, 15, 20),
    "distribution_shift": (1.5, 3.0, 4.5),
    "mechanism_scale": (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    "harmless_drift": (2.0, 3.5, 5.0),
}
SWEEP_FACTORS: tuple[str, ...] = tuple(DEFAULT_SWEEP_LEVELS)

FACTOR_VALUE_TYPES: dict[str, type] = {
    "n_samples": int,
    "n_devices": int,
    "noise_level": float,
    "distribution_shift": float,
    "mechanism_scale": float,
    "harmless_drift": float,
}

# G5 keeps compute low: GADS plus the strongest raw-space/global baselines.
DEFAULT_SWEEP_METHODS: tuple[str, ...] = ("GADS", "Wasserstein-X", "Global-SHAP")

BASE_GENERATION_PARAMS: dict[str, Any] = {
    "n_samples": 1_000,
    "noise_level": 0.2,
    "n_devices": 10,
    "distribution_shift": 3.0,
    "mechanism_scale": 1.0,
    "harmless_drift": 3.5,
}

SWEEP_RESULT_COLUMNS: tuple[str, ...] = (
    "factor",
    "level",
    "dataset",
    "scenario",
    "seed",
    "method",
    "status",
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_5",
    "kpi_auc",
    "eqp_mode",
)


@dataclass(frozen=True)
class SweepRunSpec:
    """One sweep cell: a factor level applied to one scenario/seed run."""

    factor: str
    level: float | int
    scenario: str
    seed: int
    generation_kwargs: dict[str, Any]
    run_dir: str


def _sweep_run_dir(factor: str, level: float | int, scenario: str, seed: int) -> str:
    """Return the per-cell directory name under the sweep ``runs/`` tree."""
    return f"{factor}_{level}/dataset_a_{scenario}_seed{seed}"


def assemble_sweep_runs(
    *,
    factors: tuple[str, ...],
    levels_by_factor: dict[str, tuple[float | int, ...]],
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
    base_generation_params: dict[str, Any],
) -> tuple[SweepRunSpec, ...]:
    """Expand the single-factor sweep grid into concrete run specifications."""
    if not factors:
        raise ValueError("at least one sweep factor is required")
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = [factor for factor in factors if factor not in levels_by_factor]
    if unknown:
        raise ValueError(f"unknown sweep factors: {unknown}; expected one of: {list(levels_by_factor)}")
    for factor in levels_by_factor:
        if not levels_by_factor[factor]:
            raise ValueError(f"factor {factor!r} has no levels")
    uncovered = [factor for factor in factors if factor not in base_generation_params]
    if uncovered:
        raise ValueError(f"base generation params must cover every swept factor; missing: {uncovered}")

    canonical_scenarios = tuple(normalize_scenario(scenario) for scenario in scenarios)
    specs: list[SweepRunSpec] = []
    for factor in dict.fromkeys(factors):
        for level in levels_by_factor[factor]:
            for scenario in canonical_scenarios:
                for seed in seeds:
                    generation_kwargs = dict(base_generation_params)
                    generation_kwargs["scenario"] = scenario
                    generation_kwargs["seed"] = int(seed)
                    generation_kwargs[factor] = level
                    specs.append(
                        SweepRunSpec(
                            factor=factor,
                            level=level,
                            scenario=scenario,
                            seed=int(seed),
                            generation_kwargs=generation_kwargs,
                            run_dir=_sweep_run_dir(factor, level, scenario, int(seed)),
                        )
                    )
    return tuple(specs)


def parse_level_override(raw: str) -> tuple[str, tuple[float | int, ...]]:
    """Parse one ``FACTOR=v1,v2,...`` override into typed level values."""
    factor, separator, values_raw = raw.partition("=")
    if not separator or not factor or not values_raw:
        raise ValueError(f"level overrides must look like FACTOR=v1,v2,... ; got {raw!r}")
    if factor not in FACTOR_VALUE_TYPES:
        raise ValueError(f"unknown sweep factor {factor!r}; expected one of: {list(FACTOR_VALUE_TYPES)}")
    value_type = FACTOR_VALUE_TYPES[factor]
    levels = tuple(value_type(value.strip()) for value in values_raw.split(",") if value.strip())
    if not levels:
        raise ValueError(f"factor {factor!r} got no parseable levels from {raw!r}")
    return factor, levels


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="root directory for sweep runs and tables")
    parser.add_argument(
        "--factors",
        nargs="+",
        choices=SWEEP_FACTORS,
        default=list(SWEEP_FACTORS),
        help="generation parameters to sweep (default: all single-factor grids)",
    )
    parser.add_argument(
        "--levels",
        action="append",
        default=None,
        metavar="FACTOR=v1,v2,...",
        help="override the default level grid for one factor; repeatable",
    )
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2"], help="Dataset A scenarios (default: S1 S2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="random seeds per sweep cell")
    parser.add_argument(
        "--n-samples", type=int, default=1_000, help="base rows per bundle (n_samples factor overrides)"
    )
    parser.add_argument("--noise-level", type=float, default=0.2, help="base Dataset A label noise")
    parser.add_argument("--n-devices", type=int, default=10, help="base Dataset A equipment count")
    parser.add_argument("--distribution-shift", type=float, default=3.0, help="base Dataset A physical drift")
    parser.add_argument("--mechanism-scale", type=float, default=1.0, help="base Dataset A mechanism multiplier")
    parser.add_argument("--harmless-drift", type=float, default=3.5, help="base Dataset A harmless drift")
    parser.add_argument("--model-type", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--auc-threshold", type=float, default=None)
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument("--gads-distance", choices=GADS_DISTANCES, default="wasserstein")
    parser.add_argument("--gads-bins", type=int, default=50)
    parser.add_argument("--gads-epsilon", type=float, default=1e-10)
    parser.add_argument("--method", action="append", dest="methods", default=None)
    return parser


def build_settings(args: argparse.Namespace) -> ExperimentSettings:
    """Translate parsed CLI arguments into ExperimentSettings."""
    model_params = {
        key: value
        for key, value in {
            "min_child_weight": args.min_child_weight,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
        }.items()
        if value is not None
    }
    methods = tuple(args.methods) if args.methods else DEFAULT_SWEEP_METHODS
    return ExperimentSettings(
        methods=methods,
        model_type=args.model_type,
        model_seed=args.model_seed,
        cv_seed=args.cv_seed,
        cv_splits=args.cv_splits,
        model_params=model_params or None,
        auc_threshold=args.auc_threshold,
        psi_bins=args.psi_bins,
        gads_distance=args.gads_distance,
        gads_bins=args.gads_bins,
        gads_epsilon=args.gads_epsilon,
    )


def run_sensitivity_sweep(
    *,
    output_dir: str | Path,
    specs: tuple[SweepRunSpec, ...],
    settings: ExperimentSettings,
    levels_by_factor: dict[str, tuple[float | int, ...]],
    base_generation_params: dict[str, Any],
) -> dict[str, Any]:
    """Execute every sweep cell and write the long-format table plus manifest."""
    if not specs:
        raise ValueError("sensitivity sweep produced no runs")
    root = Path(output_dir)
    metric_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    results: list[ExperimentRunResult] = []
    for spec in specs:
        run_dir = root / "runs" / spec.run_dir
        bundle = generate_dataset_bundle(**spec.generation_kwargs)
        bundle_paths = write_dataset_bundle(bundle, run_dir, stem="bundle")
        result = run_experiment_bundle(bundle, settings=settings)
        output_paths = write_experiment_outputs(result, run_dir)
        results.append(result)
        metrics = result.metrics.copy()
        metrics.insert(0, "level", spec.level)
        metrics.insert(0, "factor", spec.factor)
        metric_frames.append(metrics)
        run_records.append(
            {
                "run_dir": spec.run_dir,
                "factor": spec.factor,
                "level": spec.level,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "status": result.manifest["status"],
                "bundle_data": str(bundle_paths["data"]),
                "bundle_manifest": str(bundle_paths["manifest"]),
                "outputs": {name: str(path) for name, path in output_paths.items()},
            }
        )

    long_table = pd.concat(metric_frames, ignore_index=True)
    for column in SWEEP_RESULT_COLUMNS:
        if column not in long_table.columns:
            raise ValueError(f"sweep long table is missing expected column {column!r}")
    long_table = long_table.loc[:, SWEEP_RESULT_COLUMNS]
    results_parquet = root / "sensitivity_results.parquet"
    results_csv = root / "sensitivity_results.csv"
    long_table.to_parquet(results_parquet, index=False)
    long_table.to_csv(results_csv, index=False)

    n_valid = len(filter_valid_results(results))
    swept_factors = dict.fromkeys(spec.factor for spec in specs)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "settings": asdict(settings),
        "grid": {factor: list(levels_by_factor[factor]) for factor in swept_factors},
        "scenarios": sorted({spec.scenario for spec in specs}),
        "seeds": sorted({spec.seed for spec in specs}),
        "base_generation_params": dict(base_generation_params),
        "n_runs_total": len(specs),
        "n_runs_valid": n_valid,
        "n_runs_invalid_auc": len(specs) - n_valid,
        "runs": run_records,
        "results_parquet": str(results_parquet),
        "results_csv": str(results_csv),
    }
    manifest_path = root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    """Expand the sweep grid, execute it and print a JSON summary."""
    args = build_parser().parse_args()
    levels_by_factor = {factor: tuple(levels) for factor, levels in DEFAULT_SWEEP_LEVELS.items()}
    if args.levels is not None:
        for raw_override in args.levels:
            factor, levels = parse_level_override(raw_override)
            levels_by_factor[factor] = levels
    base_generation_params = {
        "n_samples": args.n_samples,
        "noise_level": args.noise_level,
        "n_devices": args.n_devices,
        "distribution_shift": args.distribution_shift,
        "mechanism_scale": args.mechanism_scale,
        "harmless_drift": args.harmless_drift,
    }
    specs = assemble_sweep_runs(
        factors=tuple(args.factors),
        levels_by_factor=levels_by_factor,
        scenarios=tuple(args.scenarios),
        seeds=tuple(args.seeds),
        base_generation_params=base_generation_params,
    )
    summary = run_sensitivity_sweep(
        output_dir=args.output,
        specs=specs,
        settings=build_settings(args),
        levels_by_factor=levels_by_factor,
        base_generation_params=base_generation_params,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

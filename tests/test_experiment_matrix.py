"""Focused tests for the experiment matrix runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest

from shap_diff_analysis.experiment_runner import (
    DEFAULT_MATRIX_SCENARIOS,
    ExperimentSettings,
    MatrixRunSelection,
    assemble_experiment_matrix,
    combine_valid_run_results,
    filter_valid_results,
    is_secom_raw_available,
    run_experiment_bundle,
    run_experiment_matrix,
)
from shap_diff_analysis.generate_data import generate_dataset_bundle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

run_matrix_script: ModuleType = importlib.import_module("scripts.run_experiment_matrix")


def test_filter_valid_results_excludes_invalid_auc() -> None:
    """Aggregate helpers must drop runs tagged as invalid_auc."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=200, seed=1)
    invalid_settings = ExperimentSettings(methods=("GADS",), auc_threshold=1.01)
    valid_settings = ExperimentSettings(methods=("GADS",), auc_threshold=None)
    invalid = run_experiment_bundle(bundle, settings=invalid_settings)
    valid = run_experiment_bundle(bundle, settings=valid_settings)
    assert invalid.manifest["status"] == "invalid_auc"
    assert valid.manifest["status"] == "valid"
    kept = filter_valid_results([invalid, valid])
    assert len(kept) == 1
    assert kept[0].manifest["status"] == "valid"
    combined = combine_valid_run_results([invalid, valid])
    assert bool(combined.metrics["status"].eq("valid").all())


def test_run_experiment_matrix_writes_bundle_manifest_and_aggregate(tmp_path: Path) -> None:
    """Each matrix run persists bundle artifacts and a filtered aggregate."""
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S1", "S4-null"),
        seeds=(7,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        datasets=("dataset_a",),
        dataset_a_n_samples=180,
    )
    assert summary["n_runs_total"] == 2
    assert summary["n_runs_valid"] == 2
    assert summary["exclude_invalid_auc"] is True
    assert set(summary["scenarios"]) == {"S1", "S4-null"}

    run_dir = tmp_path / "runs" / "dataset_a_S4-null_seed7"
    assert (run_dir / "bundle.parquet").is_file()
    assert (run_dir / "bundle.manifest.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "rankings.parquet").is_file()
    assert (run_dir / "metrics.parquet").is_file()

    aggregate_metrics = pd.read_parquet(tmp_path / "aggregate" / "metrics.parquet")
    assert set(aggregate_metrics["scenario"]) == {"S1", "S4-null"}
    assert (tmp_path / "aggregate" / "metrics_table.tex").is_file()
    assert (tmp_path / "matrix_summary.json").is_file()


def test_s4_null_root_metrics_remain_nan_in_matrix_run(tmp_path: Path) -> None:
    """S4-null keeps root-cause metrics undefined while FAR remains defined."""
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S4-null",),
        seeds=(11,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        datasets=("dataset_a",),
        dataset_a_n_samples=200,
    )
    assert summary["n_runs_valid"] == 1
    equipment_metrics = pd.read_parquet(tmp_path / "runs" / "dataset_a_S4-null_seed11" / "equipment_metrics.parquet")
    gads = equipment_metrics.loc[equipment_metrics["method"] == "GADS"]
    assert gads["mrr"].isna().all()
    assert gads["hr_at_1"].isna().all()
    assert gads["far_at_1"].notna().all()


@pytest.mark.parametrize("scenario", DEFAULT_MATRIX_SCENARIOS)
def test_matrix_scenario_names_are_canonical(scenario: str) -> None:
    """The default matrix covers all five Dataset A scenarios."""
    assert scenario in {"S1", "S2", "S3", "S4-main", "S4-null"}


def test_assemble_experiment_matrix_rejects_duplicate_selection(tmp_path: Path) -> None:
    """Overlapping scenario/seed selections must fail explicitly."""
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_experiment_matrix(
        output_dir=first,
        scenarios=("S1",),
        seeds=(0,),
        settings=settings,
        datasets=("dataset_a",),
        dataset_a_n_samples=120,
    )
    run_experiment_matrix(
        output_dir=second,
        scenarios=("S1",),
        seeds=(0,),
        settings=settings,
        datasets=("dataset_a",),
        dataset_a_n_samples=120,
    )
    with pytest.raises(ValueError, match="duplicate \\(scenario, seed\\) selection rejected"):
        assemble_experiment_matrix(
            output_dir=tmp_path / "assembled",
            selections=[
                MatrixRunSelection(source_dir=first, scenarios=("S1",), seeds=(0,)),
                MatrixRunSelection(source_dir=second, scenarios=("S1",), seeds=(0,)),
            ],
            dataset="dataset_a",
        )


def test_assemble_experiment_matrix_merges_disjoint_slices(tmp_path: Path) -> None:
    """Disjoint partial matrices combine into one filtered aggregate."""
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_experiment_matrix(
        output_dir=first,
        scenarios=("S1",),
        seeds=(0,),
        settings=settings,
        datasets=("dataset_a",),
        dataset_a_n_samples=120,
    )
    run_experiment_matrix(
        output_dir=second,
        scenarios=("S4-null",),
        seeds=(1,),
        settings=settings,
        datasets=("dataset_a",),
        dataset_a_n_samples=120,
    )
    summary = assemble_experiment_matrix(
        output_dir=tmp_path / "assembled",
        selections=[
            MatrixRunSelection(source_dir=first, scenarios=("S1",), seeds=(0,)),
            MatrixRunSelection(source_dir=second, scenarios=("S4-null",), seeds=(1,)),
        ],
        dataset="dataset_a",
    )
    assert summary["n_unique_keys"] == 2
    metrics = pd.read_parquet(tmp_path / "assembled" / "aggregate" / "metrics.parquet")
    assert len(metrics) == 2
    assert set(metrics["scenario"]) == {"S1", "S4-null"}


def test_dataset_b_matrix_runs_when_raw_available(tmp_path: Path) -> None:
    """Dataset B is included only when SECOM raw files verify successfully."""
    if not is_secom_raw_available():
        pytest.skip("SECOM raw files are unavailable in this environment")
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S1",),
        seeds=(3,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        datasets=("dataset_b",),
    )
    assert summary["n_runs_total"] == 1
    assert (tmp_path / "runs" / "dataset_b_S1_seed3" / "bundle.manifest.json").is_file()
    manifest = json.loads((tmp_path / "matrix_summary.json").read_text(encoding="utf-8"))
    assert manifest["skipped_datasets"] == []


def test_run_experiment_matrix_records_generation_params(tmp_path: Path) -> None:
    """Every Dataset A generation parameter is recorded in summary and run manifest."""
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S1",),
        seeds=(5,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        datasets=("dataset_a",),
        dataset_a_n_samples=150,
        dataset_a_noise_level=0.4,
        dataset_a_n_devices=12,
        dataset_a_distribution_shift=2.0,
        dataset_a_mechanism_scale=0.5,
        dataset_a_harmless_drift=4.5,
    )
    record = summary["generation_params"]
    assert record["dataset_a_n_samples"] == 150
    assert record["dataset_a_noise_level"] == 0.4
    assert record["dataset_a_n_devices"] == 12
    assert record["dataset_a_distribution_shift"] == 2.0
    assert record["dataset_a_mechanism_scale"] == 0.5
    assert record["dataset_a_harmless_drift"] == 4.5

    run_dir = tmp_path / "runs" / "dataset_a_S1_seed5"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_params"]["noise_level"] == 0.4
    assert manifest["generation_params"]["n_devices"] == 12
    bundle_metadata = manifest["bundle"]["metadata"]
    assert bundle_metadata["generation_params"]["distribution_shift"] == 2.0
    assert bundle_metadata["generation_params"]["mechanism_scale"] == 0.5


def test_dataset_b_matrix_threads_protocol_params(tmp_path: Path) -> None:
    """Dataset B fault strengths and indicator switch reach the generated bundle."""
    if not is_secom_raw_available():
        pytest.skip("SECOM raw files are unavailable in this environment")
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S4-null",),
        seeds=(0,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        datasets=("dataset_b",),
        dataset_b_harmless_shift=5.0,
        dataset_b_include_missing_indicators=False,
    )
    assert summary["n_runs_total"] == 1
    assert summary["generation_params"]["dataset_b_harmless_shift"] == 5.0
    assert summary["generation_params"]["dataset_b_include_missing_indicators"] is False
    bundle_manifest = json.loads(
        (tmp_path / "runs" / "dataset_b_S4-null_seed0" / "bundle.manifest.json").read_text(encoding="utf-8")
    )
    assert not any(column.endswith("_missing") for column in bundle_manifest["process_features"])
    metadata = bundle_manifest["metadata"]
    assert metadata["preprocess"]["include_missing_indicators"] is False
    assert metadata["generation_params"]["harmless_shift_magnitude"] == 5.0
    assert metadata["physical_shifts"]["EQP_E"]["magnitude"] == 5.0


def test_experiment_settings_model_params_recorded_in_manifest() -> None:
    """ExperimentSettings.model_params overrides are recorded per run."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=200, seed=3)
    settings = ExperimentSettings(
        methods=("GADS",),
        cv_splits=3,
        model_params={"min_child_weight": 5.0, "max_depth": 4, "n_estimators": 40},
    )
    result = run_experiment_bundle(bundle, settings=settings)
    assert result.manifest["settings"]["model_params"] == {
        "min_child_weight": 5.0,
        "max_depth": 4,
        "n_estimators": 40,
    }
    assert result.manifest["status"] == "valid"


def test_cli_passes_generation_and_model_params_to_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The matrix CLI forwards every new flag into run_experiment_matrix."""
    captured: dict[str, object] = {}

    def fake_run_experiment_matrix(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(run_matrix_script, "run_experiment_matrix", fake_run_experiment_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix.py",
            "--output",
            str(tmp_path / "out"),
            "--scenarios",
            "S1",
            "--seeds",
            "0",
            "--dataset-a-n-samples",
            "500",
            "--noise-level",
            "0.4",
            "--n-devices",
            "12",
            "--distribution-shift",
            "2.0",
            "--mechanism-scale",
            "0.5",
            "--harmless-drift",
            "4.5",
            "--dataset-b-physical-shift",
            "2.5",
            "--dataset-b-mechanism-weight",
            "10.0",
            "--dataset-b-harmless-shift",
            "3.0",
            "--no-dataset-b-include-missing-indicators",
            "--model-type",
            "lightgbm",
            "--min-child-weight",
            "5.0",
            "--max-depth",
            "4",
            "--n-estimators",
            "100",
        ],
    )
    run_matrix_script.main()

    assert captured["dataset_a_n_samples"] == 500
    assert captured["dataset_a_noise_level"] == 0.4
    assert captured["dataset_a_n_devices"] == 12
    assert captured["dataset_a_distribution_shift"] == 2.0
    assert captured["dataset_a_mechanism_scale"] == 0.5
    assert captured["dataset_a_harmless_drift"] == 4.5
    assert captured["dataset_b_physical_shift"] == 2.5
    assert captured["dataset_b_mechanism_weight"] == 10.0
    assert captured["dataset_b_harmless_shift"] == 3.0
    assert captured["dataset_b_include_missing_indicators"] is False
    settings = captured["settings"]
    assert isinstance(settings, ExperimentSettings)
    assert settings.model_type == "lightgbm"
    assert settings.model_params == {"min_child_weight": 5.0, "max_depth": 4, "n_estimators": 100}


def test_cli_model_param_flags_default_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without regularization flags the estimator keeps the pipeline defaults."""
    captured: dict[str, object] = {}

    def fake_run_experiment_matrix(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(run_matrix_script, "run_experiment_matrix", fake_run_experiment_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment_matrix.py", "--output", str(tmp_path / "out"), "--scenarios", "S1", "--seeds", "0"],
    )
    run_matrix_script.main()

    settings = captured["settings"]
    assert isinstance(settings, ExperimentSettings)
    assert settings.model_params is None
    assert captured["dataset_b_include_missing_indicators"] is True

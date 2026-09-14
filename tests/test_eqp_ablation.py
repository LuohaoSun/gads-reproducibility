"""Focused tests for EQP ablation settings and outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shap_diff_analysis.experiment_runner import (
    ExperimentSettings,
    run_eqp_ablation,
    run_eqp_ablation_matrix,
    run_experiment_bundle,
)
from shap_diff_analysis.experiment_types import DEFAULT_EQP_MODE, EQP_MODES
from shap_diff_analysis.generate_data import generate_dataset_bundle


def test_default_eqp_mode_preserves_contextual_process_protocol() -> None:
    """Default settings keep EQP in the model and evaluate the process view."""
    settings = ExperimentSettings()
    assert settings.eqp_mode == DEFAULT_EQP_MODE
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=3)
    result = run_experiment_bundle(bundle, settings=ExperimentSettings(methods=("GADS",), cv_splits=3))
    manifest = result.manifest
    assert manifest["settings"]["eqp_mode"] == "contextual_process"
    assert "EQP" in manifest["model"]["feature_names"]
    assert set(result.rankings["view"]) == {"process", "all", "residual"}
    assert set(result.metrics["eqp_mode"]) == {"contextual_process"}
    assert "EQP" not in result.rankings.loc[result.rankings["view"] == "process", "feature"].tolist()


def test_no_eqp_mode_drops_eqp_from_model_and_residual_view() -> None:
    """No-EQP ablation fits only process features and ranks the process view."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=4)
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3, eqp_mode="no_eqp")
    result = run_experiment_bundle(bundle, settings=settings)
    assert "EQP" not in result.manifest["model"]["feature_names"]
    assert set(result.rankings["view"]) == {"process"}
    assert set(result.metrics["eqp_mode"]) == {"no_eqp"}
    assert "mrr" in result.metrics.columns
    assert "hr_at_1" in result.metrics.columns
    assert "far_at_1" in result.metrics.columns


def test_all_features_mode_ranks_eqp_in_primary_metrics() -> None:
    """All-feature ablation evaluates the GADS all view including EQP."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=6)
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3, eqp_mode="all_features")
    result = run_experiment_bundle(bundle, settings=settings)
    assert "EQP" in result.manifest["model"]["feature_names"]
    all_rankings = result.rankings.loc[result.rankings["view"] == "all"]
    assert "EQP" in all_rankings["feature"].tolist()
    assert set(result.metrics["eqp_mode"]) == {"all_features"}


def test_run_eqp_ablation_emits_comparable_metrics_for_all_modes() -> None:
    """Single-bundle ablation compares all configured EQP modes."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=8)
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3)
    result = run_eqp_ablation(bundle, modes=EQP_MODES, settings=settings)
    metrics = result.metrics
    assert set(metrics["eqp_mode"]) == set(EQP_MODES)
    for column in ("mrr", "hr_at_1", "hr_at_3", "hr_at_5", "far_at_1", "far_at_5"):
        assert column in metrics.columns
    manifest = result.manifest
    assert len(manifest["runs"]) == len(EQP_MODES)
    recorded_modes = {run["settings"]["eqp_mode"] for run in manifest["runs"]}
    assert recorded_modes == set(EQP_MODES)


def test_run_eqp_ablation_rejects_unknown_mode() -> None:
    """Unknown EQP mode names must fail during experiment setup."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=120, seed=1)
    with pytest.raises(ValueError, match="unsupported eqp_mode"):
        run_experiment_bundle(
            bundle,
            settings=ExperimentSettings(methods=("GADS",), cv_splits=3, eqp_mode="with_eqp"),  # type: ignore[arg-type]
        )


def test_eqp_ablation_matrix_writes_manifest_and_aggregate(tmp_path: Path) -> None:
    """Matrix ablation persists per-mode metrics and summary metadata."""
    summary = run_eqp_ablation_matrix(
        output_dir=tmp_path,
        scenarios=("S1",),
        seeds=(9,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        modes=EQP_MODES,
        datasets=("dataset_a",),
        dataset_a_n_samples=180,
    )
    assert summary["n_runs_total"] == len(EQP_MODES)
    assert summary["eqp_modes"] == list(EQP_MODES)

    run_dir = tmp_path / "runs" / "dataset_a_S1_seed9"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == len(EQP_MODES)
    assert {run["settings"]["eqp_mode"] for run in manifest["runs"]} == set(EQP_MODES)

    metrics = pd.read_parquet(tmp_path / "aggregate" / "metrics.parquet")
    assert set(metrics["eqp_mode"]) == set(EQP_MODES)
    assert (tmp_path / "eqp_ablation_summary.json").is_file()

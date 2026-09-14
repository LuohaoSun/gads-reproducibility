"""Focused tests for configurable GADS distances and ablation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shap_diff_analysis.experiment_runner import (
    ExperimentSettings,
    run_experiment_bundle,
    run_gads_distance_ablation,
    run_gads_distance_ablation_matrix,
)
from shap_diff_analysis.gads import (
    GADS_DISTANCES,
    _reference_binned_probabilities,
    compute_gads_distance,
    rank_gads_features,
)
from shap_diff_analysis.generate_data import generate_dataset_bundle


def _shap_frame() -> tuple[pd.DataFrame, pd.Series]:
    groups = pd.Series(["A", "A", "B", "B", "C", "C"], name="EQP")
    shap_values = pd.DataFrame(
        {
            "root_a": [10.0, 10.0, 100.0, 100.0, 0.0, 0.0],
            "noise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "EQP": [3.0, 3.0, 4.0, 4.0, 0.0, 0.0],
        }
    )
    return shap_values, groups


@pytest.mark.parametrize("distance", GADS_DISTANCES)
def test_compute_gads_distance_supports_all_named_variants(distance: str) -> None:
    """Every supported distance returns a finite non-negative score."""
    abnormal = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    normal = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    score = compute_gads_distance(abnormal, normal, distance=distance, bins=4, epsilon=1e-8)
    assert np.isfinite(score)
    assert score >= 0.0


def test_js_distance_uses_jensen_shannon_not_symmetric_kl() -> None:
    """JS must use mixture m=0.5*(P+Q), not the legacy symmetric KL formula."""
    abnormal = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    normal = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    bins = 4
    epsilon = 1e-8
    reference_probs, abnormal_probs = _reference_binned_probabilities(
        normal,
        abnormal,
        bins=bins,
        epsilon=epsilon,
    )
    mixture = 0.5 * (reference_probs + abnormal_probs)
    expected_js = 0.5 * np.sum(abnormal_probs * np.log(abnormal_probs / mixture)) + 0.5 * np.sum(
        reference_probs * np.log(reference_probs / mixture)
    )
    symmetric_kl = 0.5 * np.sum(abnormal_probs * np.log(abnormal_probs / reference_probs)) + 0.5 * np.sum(
        reference_probs * np.log(reference_probs / abnormal_probs)
    )
    actual = compute_gads_distance(abnormal, normal, distance="js", bins=bins, epsilon=epsilon)
    assert actual == pytest.approx(expected_js)
    assert symmetric_kl != pytest.approx(expected_js)


def test_histogram_edges_include_out_of_range_samples() -> None:
    """Tail bins extend to +/-inf so abnormal outliers are not silently dropped."""
    normal = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    abnormal = np.array([100.0], dtype=float)
    _, abnormal_probs = _reference_binned_probabilities(normal, abnormal, bins=4, epsilon=1e-8)
    assert abnormal_probs.sum() == pytest.approx(1.0)
    assert abnormal_probs[-1] > 1e-7


def test_unknown_distance_raises_without_fallback() -> None:
    """Invalid distance names must fail loudly."""
    with pytest.raises(ValueError, match="unsupported GADS distance"):
        compute_gads_distance(np.array([1.0]), np.array([0.0]), distance="cosine")


def test_rank_gads_process_view_excludes_eqp_for_all_distances() -> None:
    """Process view keeps EQP contextual out regardless of distance choice."""
    shap_values, groups = _shap_frame()
    for distance in GADS_DISTANCES:
        rankings = rank_gads_features(
            shap_values,
            groups,
            abnormal_devices=("A", "B"),
            normal_devices=("C",),
            process_features=("root_a", "noise"),
            distance=distance,
            bins=4,
            epsilon=1e-8,
        )
        assert "EQP" not in rankings["feature"].to_list()
        assert set(rankings["gads_distance"]) == {distance}


def test_residual_view_keeps_only_eqp_column() -> None:
    """Residual view ranks the contextual EQP SHAP column separately."""
    shap_values, groups = _shap_frame()
    residual = rank_gads_features(
        shap_values,
        groups,
        abnormal_devices=("A",),
        normal_devices=("C",),
        view="residual",
        distance="mean",
    )
    assert residual["feature"].tolist() == ["EQP"]
    assert residual["view"].tolist() == ["residual"]


def _assert_gads_distance_ablation_schema(
    *,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    equipment_metrics: pd.DataFrame,
    distances: tuple[str, ...],
) -> None:
    expected_methods = {f"GADS-{distance}" for distance in distances}
    for frame in (rankings, metrics, equipment_metrics):
        assert "gads_distance" in frame.columns
        assert set(frame["gads_distance"]) == set(distances)
        assert set(frame["method"]) == expected_methods
        for distance in distances:
            method = f"GADS-{distance}"
            subset = frame.loc[frame["method"] == method, "gads_distance"]
            assert set(subset) == {distance}


def test_run_gads_distance_ablation_labels_methods_per_distance() -> None:
    """Ablation helper compares all configured distances on one bundle."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=5)
    settings = ExperimentSettings(methods=("GADS",), cv_splits=3)
    distances = ("wasserstein", "mean")
    result = run_gads_distance_ablation(bundle, distances=distances, settings=settings)
    _assert_gads_distance_ablation_schema(
        rankings=result.rankings,
        metrics=result.metrics,
        equipment_metrics=result.equipment_metrics,
        distances=distances,
    )


def test_single_distance_run_keeps_default_metrics_schema() -> None:
    """Non-ablation runs keep metrics without an extra gads_distance column."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=120, seed=2)
    settings = ExperimentSettings(methods=("GADS",), gads_distance="wasserstein", cv_splits=3)
    result = run_experiment_bundle(bundle, settings=settings)
    assert "gads_distance" in result.rankings.columns
    assert set(result.rankings["gads_distance"]) == {"wasserstein"}
    assert "gads_distance" not in result.metrics.columns
    assert "gads_distance" not in result.equipment_metrics.columns
    assert set(result.metrics["method"]) == {"GADS"}


def test_experiment_runner_rejects_unknown_gads_distance() -> None:
    """Experiment settings propagate unsupported distances as errors."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=120, seed=2)
    settings = ExperimentSettings(methods=("GADS",), gads_distance="cosine", cv_splits=3)
    with pytest.raises(ValueError, match="unsupported gads_distance"):
        run_experiment_bundle(bundle, settings=settings)


def test_distance_ablation_matrix_writes_manifest_and_aggregate(tmp_path: Path) -> None:
    """Matrix ablation persists per-distance metrics and summary metadata."""
    distances = ("wasserstein", "mean")
    summary = run_gads_distance_ablation_matrix(
        output_dir=tmp_path,
        scenarios=("S1",),
        seeds=(9,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3),
        distances=distances,
        datasets=("dataset_a",),
        dataset_a_n_samples=180,
    )
    assert summary["n_runs_total"] == 2
    assert summary["distances"] == list(distances)

    run_dir = tmp_path / "runs" / "dataset_a_S1_seed9"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["runs"]) == 2
    assert {run["gads_distance"] for run in manifest["runs"]} == set(distances)

    rankings = pd.read_parquet(run_dir / "rankings.parquet")
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    equipment_metrics = pd.read_parquet(run_dir / "equipment_metrics.parquet")
    _assert_gads_distance_ablation_schema(
        rankings=rankings,
        metrics=metrics,
        equipment_metrics=equipment_metrics,
        distances=distances,
    )

    aggregate_metrics = pd.read_parquet(tmp_path / "aggregate" / "metrics.parquet")
    aggregate_equipment_metrics = pd.read_parquet(tmp_path / "aggregate" / "equipment_metrics.parquet")
    _assert_gads_distance_ablation_schema(
        rankings=rankings,
        metrics=aggregate_metrics,
        equipment_metrics=aggregate_equipment_metrics,
        distances=distances,
    )
    latex = (run_dir / "metrics_table.tex").read_text(encoding="utf-8")
    assert "gads_distance" in latex
    assert (tmp_path / "distance_ablation_summary.json").is_file()

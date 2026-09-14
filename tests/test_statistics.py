"""Tests for experiment-result summarization and paired tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shap_diff_analysis.experiment_runner import ExperimentSettings, run_experiment_matrix
from shap_diff_analysis.statistics import (
    compare_gads_vs_baselines,
    filter_valid_runs,
    holm_adjust,
    load_experiment_metrics,
    metric_applicable,
    summarize_experiment_results,
    summarize_grouped_metrics,
    write_stats_outputs,
)


def _metrics_frame() -> pd.DataFrame:
    rows = [
        ("dataset_a", "S1", 1, "GADS", 0.8, 1.0, 0.0, "valid"),
        ("dataset_a", "S1", 2, "GADS", 0.6, 1.0, 0.0, "valid"),
        ("dataset_a", "S1", 1, "KS-X", 0.2, 0.0, 1.0, "valid"),
        ("dataset_a", "S1", 2, "KS-X", 0.1, 0.0, 1.0, "valid"),
        ("dataset_a", "S4-null", 1, "GADS", np.nan, np.nan, 0.0, "valid"),
        ("dataset_a", "S4-null", 2, "GADS", np.nan, np.nan, 0.0, "valid"),
        ("dataset_a", "S4-null", 1, "KS-X", np.nan, np.nan, 1.0, "valid"),
        ("dataset_a", "S4-null", 2, "KS-X", np.nan, np.nan, 1.0, "valid"),
        ("dataset_a", "S1", 3, "GADS", 0.1, 0.0, 1.0, "invalid_auc"),
    ]
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "seed", "method", "mrr", "hr_at_1", "far_at_1", "status"],
        ),
    )


def test_filter_valid_runs_drops_invalid_status() -> None:
    """filter_valid_runs must keep only rows with status=valid."""
    frame = _metrics_frame()
    valid = filter_valid_runs(frame)
    assert len(valid) == 8
    assert set(valid["status"]) == {"valid"}


def test_summarize_grouped_metrics_handles_s4_null_root_metrics() -> None:
    """S4-null root metrics are marked inapplicable with the expected notes."""
    summary = summarize_grouped_metrics(
        _metrics_frame(),
        metric_columns=("mrr", "far_at_1"),
        n_resamples=200,
        seed=0,
    )
    s4_mrr = summary.loc[(summary["scenario"] == "S4-null") & (summary["metric"] == "mrr")]
    assert bool(s4_mrr["applicable"].eq(False).all())
    assert bool(s4_mrr["note"].eq("s4_null_no_root_cause").all())
    assert bool(s4_mrr["n"].eq(0).all())
    s4_far = summary.loc[(summary["scenario"] == "S4-null") & (summary["metric"] == "far_at_1")]
    assert bool(s4_far["applicable"].all())
    assert s4_far["n"].iloc[0] == 2
    s1_far = summary.loc[(summary["scenario"] == "S1") & (summary["metric"] == "far_at_1")]
    assert bool(s1_far["applicable"].eq(False).all())
    assert bool(s1_far["note"].eq("no_harmless_drift").all())


def test_metric_applicable_flags_s1_s3_far() -> None:
    """FAR at 1 is inapplicable for S1-S3 and applicable for S4-main."""
    for scenario in ("S1", "S2", "S3"):
        applicable, note = metric_applicable(scenario, "far_at_1")
        assert not applicable
        assert note == "no_harmless_drift"
    applicable, note = metric_applicable("S4-main", "far_at_1")
    assert applicable
    assert note == ""


def test_compare_gads_vs_baselines_skips_s4_null_mrr() -> None:
    """GADS baseline comparisons skip inapplicable S4-null MRR and S1 FAR."""
    comparisons = compare_gads_vs_baselines(
        _metrics_frame(),
        metric_columns=("mrr", "far_at_1"),
        test="wilcoxon",
        holm=False,
    )
    s4_mrr = comparisons.loc[(comparisons["scenario"] == "S4-null") & (comparisons["metric"] == "mrr")]
    assert bool(s4_mrr["applicable"].eq(False).all())
    assert bool(s4_mrr["note"].eq("s4_null_no_root_cause").all())
    s1_far = comparisons.loc[(comparisons["scenario"] == "S1") & (comparisons["metric"] == "far_at_1")]
    assert bool(s1_far["applicable"].eq(False).all())
    assert bool(s1_far["note"].eq("no_harmless_drift").all())
    s1_mrr = comparisons.loc[(comparisons["scenario"] == "S1") & (comparisons["metric"] == "mrr")].iloc[0]
    assert s1_mrr["n_pairs"] == 2
    assert np.isfinite(s1_mrr["p_value"])


def test_holm_adjust_is_monotone() -> None:
    """Holm-adjusted p-values must remain non-decreasing."""
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_write_stats_outputs_creates_expected_files(tmp_path: Path) -> None:
    """write_stats_outputs must create CSV, parquet, and LaTeX summary files."""
    summary, comparisons = summarize_experiment_results(
        _metrics_frame(),
        metric_columns=("mrr", "far_at_1"),
        bootstrap_samples=200,
        permutation_samples=200,
        seed=0,
    )
    paths = write_stats_outputs(summary, comparisons, tmp_path)
    assert (tmp_path / "stats.csv").is_file()
    assert (tmp_path / "stats.parquet").is_file()
    assert (tmp_path / "stats_summary.tex").is_file()
    assert paths["summary_csv"].name == "stats_summary.csv"


def test_load_experiment_metrics_from_matrix(tmp_path: Path) -> None:
    """Experiment matrix output loads and summarizes with S4-null MRR inapplicable."""
    run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S1", "S4-null"),
        seeds=(5, 6),
        settings=ExperimentSettings(methods=("GADS", "KS-X"), cv_splits=3),
        datasets=("dataset_a",),
        dataset_a_n_samples=160,
    )
    metrics = load_experiment_metrics(tmp_path / "aggregate" / "metrics.parquet")
    summary, comparisons = summarize_experiment_results(
        metrics,
        metric_columns=("mrr", "far_at_1"),
        bootstrap_samples=200,
        permutation_samples=200,
        seed=1,
    )
    assert not summary.empty
    assert not comparisons.empty
    s4 = summary.loc[(summary["scenario"] == "S4-null") & (summary["metric"] == "mrr")]
    assert bool(s4["applicable"].eq(False).all())


def test_filter_valid_runs_raises_on_empty() -> None:
    """filter_valid_runs must raise when no valid rows remain."""
    frame = _metrics_frame().loc[_metrics_frame()["status"] != "valid"]
    with pytest.raises(ValueError, match="no rows with status=valid"):
        filter_valid_runs(frame)

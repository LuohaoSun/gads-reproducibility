"""Tests for paper figure helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plot_paper_figs: ModuleType = importlib.import_module("scripts.plot_paper_figs")

# A full reproduction run populates this archive (see README "Reproducing the paper").
# The exact-value regression tests below only make sense once it exists; without it they
# are skipped instead of failing, since the archive is deliberately not shipped in git.
PAPER_FINAL_ARCHIVE = ROOT / "results" / "paper_final"
ARCHIVE_AVAILABLE = (
    (PAPER_FINAL_ARCHIVE / "matrix_a_xgb/aggregate/metrics.parquet").is_file()
    and (PAPER_FINAL_ARCHIVE / "matrix_a_xgb/stats/stats_summary.parquet").is_file()
    and (PAPER_FINAL_ARCHIVE / "matrix_b_xgb/stats/stats_summary.parquet").is_file()
)
requires_archive = pytest.mark.skipif(
    not ARCHIVE_AVAILABLE,
    reason="requires the results/paper_final archive produced by a full reproduction run",
)


def _summary() -> pd.DataFrame:
    rows = [
        ("dataset_a", "S1", "GADS", "mrr", 0.7, 0.5, 0.9, True),
        ("dataset_a", "S1", "KS-X", "mrr", 0.3, 0.1, 0.5, True),
        ("dataset_a", "S1", "GADS", "far_at_1", 0.0, 0.0, 0.0, True),
        ("dataset_a", "S1", "KS-X", "far_at_1", 1.0, 1.0, 1.0, True),
        ("dataset_a", "S4-null", "GADS", "far_at_1", 0.0, 0.0, 0.0, True),
    ]
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "method", "metric", "mean", "ci_low", "ci_high", "applicable"],
        ),
    )


def _rankings() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for rank, feature in enumerate(("f1", "f2", "f3"), start=1):
            rows.append(("dataset_a", "S1", seed, "GADS", "EQP_A", feature, rank, "process"))
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "seed", "method", "equipment", "feature", "rank", "view"],
        ),
    )


def test_require_columns_raises_on_missing() -> None:
    """Missing required columns must raise ValueError."""
    with pytest.raises(ValueError, match="missing required columns"):
        plot_paper_figs.require_columns(pd.DataFrame({"a": [1]}), ("a", "b"), label="frame")


def test_plot_grouped_metric_bars_writes_file(tmp_path: Path) -> None:
    """Grouped metric bar plots must write the requested output file."""
    output = plot_paper_figs.plot_grouped_metric_bars(
        _summary(),
        metrics=("mrr",),
        output_path=tmp_path / "bars.png",
        title="test",
    )
    assert output.is_file()


def test_plot_far_bars_writes_file(tmp_path: Path) -> None:
    """FAR bar plots must write the requested output file."""
    output = plot_paper_figs.plot_far_bars(_summary(), output_path=tmp_path / "far.png")
    assert output.is_file()


def test_seed_topk_jaccard_and_heatmap(tmp_path: Path) -> None:
    """Seed top-k Jaccard scores feed a non-empty heatmap output."""
    jaccard = plot_paper_figs.seed_topk_jaccard(_rankings(), top_k=2)
    assert not jaccard.empty
    output = plot_paper_figs.plot_seed_jaccard_heatmap(jaccard, output_path=tmp_path / "jaccard.png")
    assert output.is_file()


def test_plot_rank_stability_heatmap_writes_file(tmp_path: Path) -> None:
    """Rank stability heatmaps must write the requested output file."""
    output = plot_paper_figs.plot_rank_stability_heatmap(_rankings(), top_k=2, output_path=tmp_path / "rank.png")
    assert output.is_file()


def test_plot_fig2_gads_seed_stability_writes_file(tmp_path: Path) -> None:
    """Fig2 GADS seed-stability plots must write the requested output file."""
    output = plot_paper_figs.plot_fig2_gads_seed_stability(
        _rankings(),
        output_path=tmp_path / "fig2.png",
        dataset="dataset_a",
        method="GADS",
        top_k=2,
    )
    assert output.is_file()


def test_mean_seed_pair_jaccard_requires_matching_slice() -> None:
    """mean_seed_pair_jaccard must fail when the fig2 slice has no rankings."""
    with pytest.raises(ValueError, match="no rankings match fig2 slice"):
        plot_paper_figs.mean_seed_pair_jaccard(
            _rankings(),
            dataset="dataset_b",
            scenario=None,
            method="GADS",
            equipment=None,
            top_k=2,
        )


def _rankings_with_dataset_a_label() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for rank, feature in enumerate(("f1", "f2", "f3"), start=1):
            rows.append(("Dataset A", "S1", seed, "GADS", "EQP_A", feature, rank, "process"))
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "seed", "method", "equipment", "feature", "rank", "view"],
        ),
    )


def test_fig2_accepts_dataset_a_alias_for_aggregate_style_label(tmp_path: Path) -> None:
    """Fig2 helpers accept the dataset_a alias for aggregate-style Dataset A labels."""
    rankings = _rankings_with_dataset_a_label()
    summary = plot_paper_figs.mean_seed_pair_jaccard(
        rankings,
        dataset="dataset_a",
        scenario=None,
        method="GADS",
        equipment=None,
        top_k=2,
    )
    assert not summary.empty
    assert "mean_jaccard" in summary.columns
    output = plot_paper_figs.plot_fig2_gads_seed_stability(
        rankings,
        output_path=tmp_path / "fig2.png",
        top_k=2,
    )
    assert output.is_file()


def _rankings_with_dataset_b_label() -> pd.DataFrame:
    rows = []
    for seed in (1, 2):
        for rank, feature in enumerate(("f1", "f2", "f3"), start=1):
            rows.append(("Dataset B", "S1", seed, "GADS", "EQP_B", feature, rank, "process"))
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "seed", "method", "equipment", "feature", "rank", "view"],
        ),
    )


def test_resolve_fig2_dataset_selects_b_only_actual_style_label(tmp_path: Path) -> None:
    """resolve_fig2_dataset must pick the actual Dataset B label when only B is present."""
    rankings = _rankings_with_dataset_b_label()
    resolved = plot_paper_figs.resolve_fig2_dataset(rankings, None, explicit=False)
    assert resolved == "Dataset B"
    output = plot_paper_figs.plot_fig2_gads_seed_stability(
        rankings,
        output_path=tmp_path / "fig2.png",
        dataset=resolved,
        top_k=2,
    )
    assert output.is_file()


def test_resolve_fig2_dataset_explicit_missing_raises() -> None:
    """An explicit missing fig2 dataset must raise instead of falling back."""
    rankings = _rankings_with_dataset_b_label()
    with pytest.raises(ValueError, match="fig2 dataset 'dataset_a' not found in rankings"):
        plot_paper_figs.resolve_fig2_dataset(rankings, "dataset_a", explicit=True)


def test_load_summary_table_roundtrip(tmp_path: Path) -> None:
    """load_summary_table must round-trip parquet summary tables."""
    path = tmp_path / "summary.parquet"
    _summary().to_parquet(path, index=False)
    loaded = plot_paper_figs.load_summary_table(path)
    assert len(loaded) == 5


@requires_archive
def test_parse_tradeoff_points_from_paper_final() -> None:
    """parse_tradeoff_points must match the paper_final primary-cohort tradeoff values."""
    summary_a = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_a_xgb/stats/stats_summary.parquet",
    )
    points_a = plot_paper_figs.parse_tradeoff_points(summary_a)
    gads_a = points_a.loc[points_a["method"] == "GADS"].iloc[0]
    assert gads_a["s2_mrr"] == pytest.approx(1.0, abs=1e-3)
    assert gads_a["s4_null_far_at_1"] == pytest.approx(0.15, abs=1e-3)
    assert set(points_a["method"]) == set(plot_paper_figs.PRIMARY_COHORT_METHODS)

    summary_b = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_b_xgb/stats/stats_summary.parquet",
    )
    points_b = plot_paper_figs.parse_tradeoff_points(summary_b)
    gads_b = points_b.loc[points_b["method"] == "GADS"].iloc[0]
    assert gads_b["s2_mrr"] == pytest.approx(0.709, abs=1e-3)
    assert gads_b["s4_null_far_at_1"] == pytest.approx(0.5, abs=1e-3)


@requires_archive
def test_plot_primary_cohort_tradeoff_writes_file(tmp_path: Path) -> None:
    """Primary-cohort tradeoff plots must write the requested output file."""
    summary_a = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_a_xgb/stats/stats_summary.parquet",
    )
    summary_b = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_b_xgb/stats/stats_summary.parquet",
    )
    output = plot_paper_figs.plot_primary_cohort_tradeoff(
        {
            "Dataset A": plot_paper_figs.parse_tradeoff_points(summary_a),
            "Dataset B": plot_paper_figs.parse_tradeoff_points(summary_b),
        },
        output_path=tmp_path / "tradeoff.png",
    )
    assert output.is_file()


@requires_archive
def test_plot_primary_cohort_tradeoff_direct_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tradeoff panels must label all methods directly with no legend, title, or diagonal."""
    captured: list[Figure] = []
    original_close = plt.close

    def _capture_close(fig: Figure | None = None) -> None:
        if fig is not None:
            captured.append(fig)
        original_close(fig)

    monkeypatch.setattr(plt, "close", _capture_close)
    summary_a = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_a_xgb/stats/stats_summary.parquet",
    )
    summary_b = plot_paper_figs.load_summary_table(
        ROOT / "results/paper_final/matrix_b_xgb/stats/stats_summary.parquet",
    )
    plot_paper_figs.plot_primary_cohort_tradeoff(
        {
            "Dataset A": plot_paper_figs.parse_tradeoff_points(summary_a),
            "Dataset B": plot_paper_figs.parse_tradeoff_points(summary_b),
        },
        output_path=tmp_path / "tradeoff.pdf",
    )
    assert len(captured) == 1
    fig = captured[0]
    assert getattr(fig, "_suptitle", None) is None
    assert fig.legends == []
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_xlim() == (0.0, 1.0)
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.get_legend() is None
        assert len(ax.lines) == 0
        assert len(ax.collections) == len(plot_paper_figs.PRIMARY_COHORT_METHODS)
        annotations = {text.get_text() for text in ax.texts}
        assert annotations == set(plot_paper_figs.PRIMARY_COHORT_METHODS)
        assert all(cast(float, text.get_fontsize()) >= 8.5 for text in ax.texts)


def test_plot_primary_cohort_tradeoff_requires_annotation_layout(tmp_path: Path) -> None:
    """Tradeoff methods without a configured annotation position must raise."""
    points = pd.DataFrame(
        {
            "method": ["GADS", "Unregistered-Method"],
            "s2_mrr": [1.0, 0.5],
            "s4_null_far_at_1": [0.1, 0.9],
        },
    )
    with pytest.raises(ValueError, match="no annotation position configured"):
        plot_paper_figs.plot_primary_cohort_tradeoff(
            {"Dataset X": points},
            output_path=tmp_path / "tradeoff.pdf",
        )


def test_fig1_svg_structure_contract() -> None:
    """Fig1 SVG must keep five main nodes, one output node, and a distinct EQP branch."""
    svg_path = ROOT / plot_paper_figs.DEFAULT_FIG1_SVG
    content = svg_path.read_text(encoding="utf-8")
    assert content.count('class="node-main"') == 4
    assert content.count('class="node-output"') == 1
    assert content.count('class="node-eqp"') == 1
    assert content.count('class="arrow-main"') == 4
    assert content.count('class="arrow-branch"') == 1
    assert "MRR" not in content
    assert "benchmark" not in content
    assert "lane-label" not in content


def test_render_fig1_from_svg_writes_pdf(tmp_path: Path) -> None:
    """Fig1 SVG rendering must write the requested PDF output file."""
    svg_path = ROOT / plot_paper_figs.DEFAULT_FIG1_SVG
    output = plot_paper_figs.render_fig1_from_svg(svg_path=svg_path, output_path=tmp_path / "fig1.pdf")
    assert output.is_file()


@requires_archive
def test_fig2_jaccard_values_on_paper_final(tmp_path: Path) -> None:
    """Fig2 seed-pair Jaccard values must match the paper_final matrix_a_xgb aggregate."""
    from shap_diff_analysis.statistics import load_experiment_metrics, load_experiment_rankings

    root = ROOT / "results/paper_final/matrix_a_xgb"
    metrics = load_experiment_metrics(root / "aggregate/metrics.parquet")
    rankings = load_experiment_rankings(root / "aggregate/rankings.parquet", metrics=metrics)
    frame = rankings.loc[rankings["scenario"].isin(plot_paper_figs.FIG2_LOCALIZATION_SCENARIOS)]
    summary = plot_paper_figs.mean_seed_pair_jaccard(
        frame,
        dataset="dataset_a",
        scenario=None,
        method="GADS",
        equipment=None,
        top_k=5,
    )
    s2 = summary.loc[summary["scenario"] == "S2", "mean_jaccard"].iloc[0]
    assert s2 == pytest.approx(0.378, abs=1e-2)
    output = plot_paper_figs.plot_fig2_gads_seed_stability(
        rankings,
        output_path=tmp_path / "fig2.png",
        dataset="dataset_a",
        top_k=5,
    )
    assert output.is_file()

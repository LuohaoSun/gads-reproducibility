"""Tests for the bounded manuscript-figure build script."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

build_isa_figures: ModuleType = importlib.import_module("scripts.build_isa_figures")
plot_paper_figs: ModuleType = importlib.import_module("scripts.plot_paper_figs")

REQUIRED_OUTPUT_NAMES = ("fig1_framework.pdf", "tradeoff_primary_cohort.pdf", "fig2.png", "fig4.png")


def _tradeoff_summary(dataset: str, mrr_shift: float) -> pd.DataFrame:
    rows = []
    for index, method in enumerate(plot_paper_figs.PRIMARY_COHORT_METHODS):
        mrr = min(0.95, 0.2 + index * 0.15 + mrr_shift)
        far = max(0.0, 0.9 - index * 0.18)
        rows.append((dataset, "S2", method, "mrr", round(mrr, 3), 0.0, 1.0, True))
        rows.append((dataset, "S4-null", method, "far_at_1", round(far, 3), 0.0, 1.0, True))
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["dataset", "scenario", "method", "metric", "mean", "ci_low", "ci_high", "applicable"],
        ),
    )


def _g3_table() -> pd.DataFrame:
    rows = [
        (distance, level, "GADS", seed, 0.4 + level / 2)
        for distance in ("wasserstein", "mean")
        for level in (0.1, 0.5)
        for seed in (0, 1)
    ]
    return pd.DataFrame(rows, columns=pd.Index(["distance", "level", "method", "seed", "mrr"]))


def _rankings() -> pd.DataFrame:
    rows = []
    for seed in (0, 1):
        for rank, feature in enumerate(("f1", "f2"), start=1):
            rows.append(("Dataset A", "S2", seed, "GADS", "EQP_A", feature, rank, "process"))
    return pd.DataFrame(
        rows,
        columns=pd.Index(["dataset", "scenario", "seed", "method", "equipment", "feature", "rank", "view"]),
    )


def _synthetic_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "paper_final"
    for suffix in ("a", "b"):
        stats_dir = archive / f"matrix_{suffix}_xgb" / "stats"
        stats_dir.mkdir(parents=True)
        shift = 0.05 if suffix == "a" else 0.0
        _tradeoff_summary("dataset_x", shift).to_parquet(stats_dir / "stats_summary.parquet", index=False)
    g3_dir = archive / "g3_distance_intensity"
    g3_dir.mkdir(parents=True)
    _g3_table().to_parquet(g3_dir / "g3_distance_intensity_merged.parquet", index=False)
    return archive


def test_plan_build_maps_exactly_four_manuscript_assets(tmp_path: Path) -> None:
    """The planner must map exactly the four required asset names under the output dir."""
    repo_root = tmp_path / "repo"
    plan = build_isa_figures.plan_build(repo_root, tmp_path / "out")
    assert tuple(plan.output_dir / name for name in REQUIRED_OUTPUT_NAMES) == (
        plan.output_dir / "fig1_framework.pdf",
        plan.output_dir / "tradeoff_primary_cohort.pdf",
        plan.output_dir / "fig2.png",
        plan.output_dir / "fig4.png",
    )
    assert [name for name, _ in build_isa_figures.BUILD_STEPS] == list(REQUIRED_OUTPUT_NAMES)
    assert plan.archive_dir == repo_root / "results/paper_final"
    assert plan.fig1_svg == repo_root / "assets/fig1_framework.svg"


def test_plan_build_honors_explicit_overrides(tmp_path: Path) -> None:
    """Explicit --archive-dir and --fig1-svg values must override the defaults."""
    archive = tmp_path / "elsewhere"
    svg = tmp_path / "custom.svg"
    plan = build_isa_figures.plan_build(tmp_path, tmp_path / "out", archive_dir=archive, fig1_svg=svg)
    assert plan.archive_dir == archive
    assert plan.fig1_svg == svg
    assert plan.tradeoff_panels[0][1] == archive / "matrix_a_xgb/stats/stats_summary.parquet"
    assert plan.tradeoff_panels[1][1] == archive / "matrix_b_xgb/stats/stats_summary.parquet"
    assert plan.g3_parquet == archive / "g3_distance_intensity/g3_distance_intensity_merged.parquet"


def test_verify_build_environment_fails_without_rsvg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing rsvg-convert binary must fail the build before any work starts."""
    monkeypatch.setattr(build_isa_figures.shutil, "which", lambda _: None)
    plan = build_isa_figures.plan_build(tmp_path, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match=r"rsvg-convert.*not found on PATH"):
        build_isa_figures.verify_build_environment(plan)


def test_verify_build_environment_lists_missing_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All missing archived inputs must be reported at once with no fallback."""
    monkeypatch.setattr(build_isa_figures.shutil, "which", lambda _: "/usr/bin/rsvg-convert")
    archive = _synthetic_archive(tmp_path)
    plan = build_isa_figures.plan_build(tmp_path, tmp_path / "out", archive_dir=archive)
    with pytest.raises(FileNotFoundError) as raised:
        build_isa_figures.verify_build_environment(plan)
    message = str(raised.value)
    assert "matrix_a_xgb/aggregate/metrics.parquet" in message
    assert "matrix_a_xgb/aggregate/rankings.parquet" in message
    assert "fig1_framework.svg" in message


def test_run_build_produces_exactly_four_assets_with_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full build must write the four planned assets without rsvg or archives."""
    archive = _synthetic_archive(tmp_path)
    fig1_svg = tmp_path / "input_fig1_framework.svg"
    fig1_svg.write_text("<svg/>", encoding="utf-8")
    rendered: list[tuple[Path, Path]] = []

    def fake_render_fig1(*, svg_path: Path, output_path: Path, dpi: int = 300) -> Path:
        del dpi
        rendered.append((svg_path, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF-1.4 fake")
        return output_path

    monkeypatch.setattr(build_isa_figures, "render_fig1_from_svg", fake_render_fig1)
    monkeypatch.setattr(build_isa_figures, "load_fig2_rankings", lambda metrics, rankings: _rankings())

    plan = build_isa_figures.plan_build(tmp_path, tmp_path / "output", archive_dir=archive, fig1_svg=fig1_svg)
    figures = build_isa_figures.run_build(plan)

    assert set(figures) == set(REQUIRED_OUTPUT_NAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures.values())
    assert all(path.parent == plan.output_dir for path in figures.values())
    assert rendered == [(plan.fig1_svg, plan.output_dir / "fig1_framework.pdf")]
    tradeoff_pdf = figures["tradeoff_primary_cohort.pdf"]
    assert tradeoff_pdf.suffix == ".pdf"
    assert figures["fig2.png"].suffix == ".png"
    assert figures["fig4.png"].suffix == ".png"


def test_cli_parser_requires_output_dir(capsys: pytest.CaptureFixture[str]) -> None:
    """Running without --output-dir must exit with a usage error."""
    with pytest.raises(SystemExit):
        build_isa_figures.build_parser().parse_args([])
    assert "--output-dir" in capsys.readouterr().err

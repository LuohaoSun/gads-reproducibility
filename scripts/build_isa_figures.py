#!/usr/bin/env python3
"""Build the four generated CAS manuscript figures into an arbitrary output directory.

Produces exactly ``fig1_framework.pdf``, ``tradeoff_primary_cohort.pdf``, ``fig2.png``
and ``fig4.png`` by reusing the render/generation helpers in ``scripts.plot_paper_figs``
against the archived experiment outputs. The build fails explicitly (no silent
fallback) when ``rsvg-convert`` or any required input table is unavailable.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from shap_diff_analysis.statistics import load_experiment_metrics, load_experiment_rankings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_paper_figs import (  # noqa: E402
    DEFAULT_FIG1_SVG,
    load_summary_table,
    parse_tradeoff_points,
    plot_fig2_gads_seed_stability,
    plot_g3_intensity_curve,
    plot_primary_cohort_tradeoff,
    render_fig1_from_svg,
    resolve_fig2_dataset,
)

DEFAULT_ARCHIVE_RELPATH = Path("results/paper_final")
FIG1_OUTPUT_NAME = "fig1_framework.pdf"
TRADEOFF_OUTPUT_NAME = "tradeoff_primary_cohort.pdf"
FIG2_OUTPUT_NAME = "fig2.png"
FIG4_OUTPUT_NAME = "fig4.png"
REQUIRED_OUTPUT_NAMES: tuple[str, ...] = (
    FIG1_OUTPUT_NAME,
    TRADEOFF_OUTPUT_NAME,
    FIG2_OUTPUT_NAME,
    FIG4_OUTPUT_NAME,
)
TRADEOFF_PANEL_RELPATHS: tuple[tuple[str, Path], ...] = (
    ("Dataset A", Path("matrix_a_xgb/stats/stats_summary.parquet")),
    ("Dataset B", Path("matrix_b_xgb/stats/stats_summary.parquet")),
)
FIG2_METRICS_RELPATH = Path("matrix_a_xgb/aggregate/metrics.parquet")
FIG2_RANKINGS_RELPATH = Path("matrix_a_xgb/aggregate/rankings.parquet")
G3_PARQUET_RELPATH = Path("g3_distance_intensity/g3_distance_intensity_merged.parquet")
FIG2_TOP_K = 5
RSVG_CONVERT_BINARY = "rsvg-convert"


@dataclass(frozen=True)
class FigureBuildPlan:
    """Resolved input and output paths for one bounded manuscript-figure build."""

    output_dir: Path
    archive_dir: Path
    fig1_svg: Path
    tradeoff_panels: tuple[tuple[str, Path], ...]
    fig2_metrics: Path
    fig2_rankings: Path
    g3_parquet: Path

    @property
    def required_inputs(self) -> tuple[Path, ...]:
        """Return every file that must exist before building may start."""
        return (
            self.fig1_svg,
            *(summary for _, summary in self.tradeoff_panels),
            self.fig2_metrics,
            self.fig2_rankings,
            self.g3_parquet,
        )


def plan_build(
    repo_root: Path,
    output_dir: Path,
    *,
    archive_dir: Path | None = None,
    fig1_svg: Path | None = None,
) -> FigureBuildPlan:
    """Resolve all figure input/output paths without touching the filesystem."""
    archive = archive_dir or repo_root / DEFAULT_ARCHIVE_RELPATH
    svg_source = fig1_svg or repo_root / DEFAULT_FIG1_SVG
    return FigureBuildPlan(
        output_dir=output_dir,
        archive_dir=archive,
        fig1_svg=svg_source,
        tradeoff_panels=tuple((title, archive / relpath) for title, relpath in TRADEOFF_PANEL_RELPATHS),
        fig2_metrics=archive / FIG2_METRICS_RELPATH,
        fig2_rankings=archive / FIG2_RANKINGS_RELPATH,
        g3_parquet=archive / G3_PARQUET_RELPATH,
    )


def require_rsvg_convert() -> str:
    """Locate the rsvg-convert binary or raise with install guidance."""
    binary_path = shutil.which(RSVG_CONVERT_BINARY)
    if binary_path is None:
        raise FileNotFoundError(
            f"{RSVG_CONVERT_BINARY} is required to render {FIG1_OUTPUT_NAME} but was not found on PATH; "
            "install librsvg first (e.g. brew install librsvg)"
        )
    return binary_path


def verify_build_environment(plan: FigureBuildPlan) -> None:
    """Fail loudly unless every external dependency and input file is available."""
    require_rsvg_convert()
    missing = [input_path for input_path in plan.required_inputs if not input_path.is_file()]
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"required figure inputs are missing:\n{listed}\n"
            "pass --archive-dir / --fig1-svg to point at the archived experiment outputs"
        )


def build_fig1(plan: FigureBuildPlan) -> Path:
    """Render Fig. 1 from its tracked SVG source to a vector PDF."""
    return render_fig1_from_svg(svg_path=plan.fig1_svg, output_path=plan.output_dir / FIG1_OUTPUT_NAME)


def build_tradeoff(plan: FigureBuildPlan) -> Path:
    """Render the two-panel primary-cohort S2 MRR vs S4-null FAR@1 trade-off PDF."""
    points_by_dataset = {
        title: parse_tradeoff_points(load_summary_table(summary)) for title, summary in plan.tradeoff_panels
    }
    return plot_primary_cohort_tradeoff(points_by_dataset, output_path=plan.output_dir / TRADEOFF_OUTPUT_NAME)


def load_fig2_rankings(metrics_path: Path, rankings_path: Path) -> pd.DataFrame:
    """Load valid-run-filtered rankings backing the Fig. 2 stability chart."""
    metrics = load_experiment_metrics(metrics_path)
    return load_experiment_rankings(rankings_path, metrics=metrics)


def build_fig2(plan: FigureBuildPlan) -> Path:
    """Render the GADS process-view Top-K seed-stability bar chart PNG."""
    rankings = load_fig2_rankings(plan.fig2_metrics, plan.fig2_rankings)
    dataset = resolve_fig2_dataset(rankings, None, explicit=False)
    return plot_fig2_gads_seed_stability(
        rankings,
        output_path=plan.output_dir / FIG2_OUTPUT_NAME,
        dataset=dataset,
        top_k=FIG2_TOP_K,
    )


def build_fig4(plan: FigureBuildPlan) -> Path:
    """Render the GADS MRR versus mechanism-scale intensity curve PNG."""
    return plot_g3_intensity_curve(pd.read_parquet(plan.g3_parquet), output_path=plan.output_dir / FIG4_OUTPUT_NAME)


BUILD_STEPS: tuple[tuple[str, Callable[[FigureBuildPlan], Path]], ...] = (
    (FIG1_OUTPUT_NAME, build_fig1),
    (TRADEOFF_OUTPUT_NAME, build_tradeoff),
    (FIG2_OUTPUT_NAME, build_fig2),
    (FIG4_OUTPUT_NAME, build_fig4),
)


def run_build(plan: FigureBuildPlan) -> dict[str, Path]:
    """Run every build step in fixed order and confirm the four planned assets."""
    written: dict[str, Path] = {}
    for name, build_step in BUILD_STEPS:
        built = build_step(plan)
        expected = plan.output_dir / name
        if built != expected:
            raise RuntimeError(f"figure builder for {name} wrote {built} instead of the planned {expected}")
        if not built.is_file():
            raise RuntimeError(f"figure builder for {name} did not produce {built}")
        written[name] = built
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="directory that receives the four generated assets"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=f"archived paper_final root (default: <repo>/{DEFAULT_ARCHIVE_RELPATH})",
    )
    parser.add_argument(
        "--fig1-svg",
        type=Path,
        default=None,
        help=f"Fig. 1 SVG source (default: <repo>/{DEFAULT_FIG1_SVG})",
    )
    return parser


def main() -> None:
    """Verify availability of all inputs, then build exactly four manuscript assets."""
    args = build_parser().parse_args()
    plan = plan_build(REPO_ROOT, args.output_dir, archive_dir=args.archive_dir, fig1_svg=args.fig1_svg)
    verify_build_environment(plan)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    figures = run_build(plan)
    print(
        json.dumps(
            {
                "output_dir": str(plan.output_dir),
                "figures": {name: str(path) for name, path in figures.items()},
                "inputs": {"archive_dir": str(plan.archive_dir), "fig1_svg": str(plan.fig1_svg)},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

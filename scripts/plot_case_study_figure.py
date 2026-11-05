#!/usr/bin/env python3
"""Render manuscript Figure 5: the mechanism-shift case study (Dataset A, S2, seed 0).

All four panels are recomputed from the repository's own generator and model
entry points, never from a cached image or from the archived aggregate tables:

* raw ``feat_special_imp`` kernel densities for EQP_B versus the normal pool,
  annotated with the KS statistic and p-value;
* out-of-fold SHAP attribution densities for the same feature, annotated with
  the Wasserstein distance;
* the GADS process-feature scores for EQP_B (top six), the primary ranking;
* the separate EQP attribution channel, which is excluded from that ranking.

Font sizes are chosen for the printed figure: the PNG is placed at
``0.8\\linewidth`` (~315 pt) in the manuscript, so every panel label is set well
above the journal's 7 pt floor for figure lettering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import cast

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_case_study import (  # noqa: E402
    ABNORMAL_DEVICE,
    CASE_STUDY_FEATURE,
    N_SAMPLES,
    SEED,
    build_case_study_bundle,
    compute_case_study_statistics,
    fit_oof_attributions,
)
from shap_diff_analysis.gads import rank_gads_features  # noqa: E402

OUTPUT_NAME = "fig5_case_study.png"
FIGURE_SIZE = (7.8, 5.7)
DPI = 300
TOP_N_FEATURES = 6
KDE_POINTS = 400
EQP_BINS = 40
NORMAL_COLOR = "#356a9c"
ABNORMAL_COLOR = "#d1495b"
BAR_COLOR = "#2a9d8f"
FONT_TICK = 13.5
FONT_LABEL = 14.5
FONT_TITLE = 15.0
FONT_LEGEND = 13.5
FONT_ANNOTATION = 13.5


@dataclass(frozen=True)
class CaseStudyPanels:
    """Every array and statistic behind the four case-study panels."""

    seed: int
    raw_normal: np.ndarray
    raw_abnormal: np.ndarray
    shap_normal: np.ndarray
    shap_abnormal: np.ndarray
    eqp_normal: np.ndarray
    eqp_abnormal: np.ndarray
    ranking: pd.DataFrame
    ks_statistic: float
    ks_pvalue: float
    wasserstein: float


def build_case_study_panels(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    feature: str = CASE_STUDY_FEATURE,
    abnormal_device: str = ABNORMAL_DEVICE,
) -> CaseStudyPanels:
    """Generate the case-study bundle, fit OOF attributions and collect panel arrays."""
    bundle = build_case_study_bundle(n_samples=n_samples, seed=seed)
    attributions = fit_oof_attributions(bundle)
    statistics = compute_case_study_statistics(bundle, attributions, feature=feature, abnormal_device=abnormal_device)

    groups = bundle.data[bundle.group_column].astype(str)
    abnormal_mask = groups == abnormal_device
    normal_mask = groups.isin(bundle.normal_devices)
    if not bool(abnormal_mask.any()) or not bool(normal_mask.any()):
        raise ValueError("case-study cohort must contain both abnormal and normal observations")
    eqp_column = bundle.group_column
    ranking = rank_gads_features(
        attributions.shap_values,
        cast("pd.Series", groups),
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
    )
    selected_ranking = ranking.loc[ranking["equipment"] == abnormal_device].copy()
    if selected_ranking.empty:
        raise RuntimeError(f"GADS produced no ranking rows for equipment {abnormal_device!r}")
    return CaseStudyPanels(
        seed=int(seed),
        raw_normal=bundle.data.loc[normal_mask, feature].to_numpy(dtype=float),
        raw_abnormal=bundle.data.loc[abnormal_mask, feature].to_numpy(dtype=float),
        shap_normal=attributions.shap_values.loc[normal_mask, feature].to_numpy(dtype=float),
        shap_abnormal=attributions.shap_values.loc[abnormal_mask, feature].to_numpy(dtype=float),
        eqp_normal=attributions.shap_values.loc[normal_mask, eqp_column].to_numpy(dtype=float),
        eqp_abnormal=attributions.shap_values.loc[abnormal_mask, eqp_column].to_numpy(dtype=float),
        ranking=selected_ranking.sort_values(["rank", "feature"], kind="stable"),
        ks_statistic=statistics.ks_statistic,
        ks_pvalue=statistics.ks_pvalue,
        wasserstein=statistics.wasserstein,
    )


def kde_curve(values: np.ndarray, grid: np.ndarray, *, label: str) -> np.ndarray:
    """Evaluate a Gaussian KDE of ``values`` on ``grid``."""
    if values.size < 2:
        raise ValueError(f"{label} must contain at least two observations for a KDE")
    return np.asarray(gaussian_kde(values)(grid), dtype=float)


def _style_axis(axis: Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8, linespacing=1.25)
    axis.set_xlabel(xlabel, fontsize=FONT_LABEL)
    axis.set_ylabel(ylabel, fontsize=FONT_LABEL)
    axis.tick_params(axis="both", labelsize=FONT_TICK)
    sns.despine(ax=axis)


def _density_panel(
    axis: Axes,
    normal: np.ndarray,
    abnormal: np.ndarray,
    *,
    title: str,
    statistic: str,
    xlabel: str,
) -> None:
    """Draw the two kernel densities of one distributional panel."""
    low = float(min(normal.min(), abnormal.min()))
    high = float(max(normal.max(), abnormal.max()))
    padding = 0.05 * (high - low)
    grid = np.linspace(low - padding, high + padding, KDE_POINTS)
    normal_density = kde_curve(normal, grid, label="normal pool")
    abnormal_density = kde_curve(abnormal, grid, label=ABNORMAL_DEVICE)
    axis.plot(grid, normal_density, color=NORMAL_COLOR, linewidth=2.2, label="Normal pool")
    axis.plot(grid, abnormal_density, color=ABNORMAL_COLOR, linewidth=2.2, label=ABNORMAL_DEVICE)
    axis.set_ylim(bottom=0.0)
    _style_axis(axis, title=f"{title}\n{statistic}", xlabel=xlabel, ylabel="density")


def _score_panel(axis: Axes, ranking: pd.DataFrame) -> None:
    """Draw the GADS process-feature scores for the abnormal equipment."""
    top = ranking.head(TOP_N_FEATURES).sort_values(["score", "feature"], ascending=[True, False], kind="stable")
    axis.barh(
        top["feature"].astype(str),
        top["score"].to_numpy(dtype=float),
        color=BAR_COLOR,
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    _style_axis(
        axis,
        title=f"GADS scores ({ABNORMAL_DEVICE})",
        xlabel="GADS score",
        ylabel="process feature",
    )
    axis.tick_params(axis="y", labelsize=FONT_TICK)


def _eqp_panel(axis: Axes, normal: np.ndarray, abnormal: np.ndarray) -> None:
    """Draw the separate EQP attribution channel as overlapping densities."""
    low = float(min(normal.min(), abnormal.min()) - 0.05 * np.ptp(normal))
    high = float(max(normal.max(), abnormal.max()))
    bins = np.linspace(low, high, EQP_BINS)
    axis.hist(normal, bins=bins.tolist(), density=True, color=NORMAL_COLOR, alpha=0.60, label="Normal pool")
    axis.hist(
        abnormal,
        bins=bins.tolist(),
        density=True,
        color=ABNORMAL_COLOR,
        alpha=0.60,
        label=ABNORMAL_DEVICE,
    )
    _style_axis(
        axis,
        title="EQP residual channel",
        xlabel="EQP attribution",
        ylabel="density",
    )


def plot_case_study(panels: CaseStudyPanels, *, output_path: Path) -> Path:
    """Render the two-by-two case-study figure and write it to ``output_path``."""
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams["font.size"] = FONT_TICK
    mpl.rcParams["axes.edgecolor"] = "#64748b"
    mpl.rcParams["axes.linewidth"] = 0.9
    mpl.rcParams["grid.color"] = "#e2e8f0"
    mpl.rcParams["grid.linewidth"] = 0.7

    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=DPI)
    _density_panel(
        axes[0, 0],
        panels.raw_normal,
        panels.raw_abnormal,
        title="Raw feature values",
        statistic=f"KS = {panels.ks_statistic:.4f} (p = {panels.ks_pvalue:.4f})",
        xlabel="physical value",
    )
    _density_panel(
        axes[0, 1],
        panels.shap_normal,
        panels.shap_abnormal,
        title="OOF SHAP attributions",
        statistic=f"Wasserstein = {panels.wasserstein:.4f}",
        xlabel="SHAP attribution",
    )
    _score_panel(axes[1, 0], panels.ranking)
    _eqp_panel(axes[1, 1], panels.eqp_normal, panels.eqp_abnormal)

    figure.suptitle(
        f"Mechanism-shift case study: {CASE_STUDY_FEATURE} (Dataset A, S2, seed {panels.seed})",
        fontsize=FONT_TITLE,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=NORMAL_COLOR, linewidth=2.2),
            Line2D([], [], color=ABNORMAL_COLOR, linewidth=2.2),
        ],
        labels=["Normal pool", ABNORMAL_DEVICE],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=1.6,
        columnspacing=2.4,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="directory that receives the figure")
    parser.add_argument("--output-name", default=OUTPUT_NAME, help=f"output file name (default: {OUTPUT_NAME})")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help=f"bundle size (default: {N_SAMPLES})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"scenario seed (default: {SEED})")
    return parser


def main() -> None:
    """Recompute the case study and write the manuscript figure."""
    args = build_parser().parse_args()
    panels = build_case_study_panels(n_samples=args.n_samples, seed=args.seed)
    output_path = plot_case_study(panels, output_path=args.output_dir / args.output_name)
    print(
        json.dumps(
            {
                "figure": str(output_path),
                "ks_statistic": round(panels.ks_statistic, 4),
                "ks_pvalue": round(panels.ks_pvalue, 4),
                "wasserstein": round(panels.wasserstein, 4),
                "top_ranked": panels.ranking.head(TOP_N_FEATURES)["feature"].astype(str).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

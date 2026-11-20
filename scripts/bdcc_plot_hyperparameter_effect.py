"""Draw the hyperparameter effect figure requested by Reviewer 2.3.

The six-configuration Dataset B grid already exists under
``results/bdcc_revision/pack1_seeds_hyperparams/hyperparameter_grid`` and is the
only input; nothing is refit here.  Depth and minimum child weight form a 3x2
grid, so the figure is drawn as two line panels (AUC and MRR versus depth, one
line per minimum child weight) plus two bar panels (generalization gap and
false-alarm rate).  Lines show the depth trend directly instead of asking the
reader to compare six near-equal bars.

Usage:
    uv run python scripts/bdcc_plot_hyperparameter_effect.py \
        --output docs/bdcc/revision_r1/images/fig6_hyperparameter_effect.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd

# Pooled metrics are averaged over the scenarios in which each metric is defined.
LOCALIZATION_SCENARIOS = ("S1", "S2", "S3", "S4-main")
FALSE_ALARM_SCENARIOS = ("S4-main", "S4-null")
PAPER_DEPTH = 3
PAPER_MCW = 5

DEPTHS = (3, 4, 6)
MCW_VALUES = (1, 5, 10)
MCW_COLORS = {1: "#94a3b8", 5: "#1d4ed8", 10: "#0f766e"}
MCW_MARKERS = {1: "s", 5: "o", 10: "^"}

BAR_CONFIG_ORDER = ("d3_mcw1", "d3_mcw5", "d3_mcw10", "d4_mcw5", "d6_mcw1", "d6_mcw5")
BAR_CONFIG_DISPLAY = {
    "d3_mcw1": "3/1",
    "d3_mcw5": "3/5",
    "d3_mcw10": "3/10",
    "d4_mcw5": "4/5",
    "d6_mcw1": "6/1",
    "d6_mcw5": "6/5",
}

GADS_BLUE = "#1d4ed8"
NEUTRAL = "#475569"
HIGHLIGHT = "#dc2626"
# Shared vertical range for the two line panels so the AUC and MRR spreads are
# visually comparable instead of each auto-scaling to its own tiny variation.
# The upper limit leaves headroom above the top gridline so the corner
# annotation does not sit on it. Both line panels share this range.
LINE_PANEL_RANGE = (0.66, 1.02)
# Bar panels stop below 1.0 so the corner summary text sits above the bars.
BAR_PANEL_MAX = 1.0


def load_grid(path: Path) -> pd.DataFrame:
    """Read the per-configuration metric summary and drop the indicator variant.

    The missing-indicator row is a separate ablation, not a point on this grid,
    so including it would break the depth x minimum-child-weight structure.
    """
    frame = pd.read_csv(path)
    frame = frame.loc[~frame["include_missing_indicators"].astype(bool)]
    expected = {f"d{depth}_mcw{weight}" for depth in DEPTHS for weight in MCW_VALUES}
    unknown = sorted(set(frame["config_label"]) - expected)
    if unknown:
        raise SystemExit(f"hyperparameter effect: unexpected grid configurations {unknown}")
    return frame


def pooled_metric(frame: pd.DataFrame, metric: str, scenarios: tuple[str, ...]) -> pd.DataFrame:
    """Average one metric over the scenarios where it is defined, per configuration.

    Cells with no defined value (e.g. FAR in S1--S3) are dropped rather than
    treated as zero, which would understate the false-alarm rates.
    """
    block = frame.loc[frame["scenario"].isin(scenarios) & (frame["metric"] == metric)]
    block = block.dropna(subset=["mean"])
    if block.empty:
        raise SystemExit(f"hyperparameter effect: no defined values for {metric} in {scenarios}")
    return (
        block.groupby(["config_label", "max_depth", "min_child_weight"], as_index=False)["mean"]
        .mean()
        .sort_values(["min_child_weight", "max_depth"])
    )


def style_axis(axis: Axes, *, title: str, ylabel: str) -> None:
    """Apply the shared axis styling."""
    axis.set_title(title, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=8)
    axis.tick_params(labelsize=7.5)
    axis.grid(axis="y", linewidth=0.4, alpha=0.45)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def add_line_panel(axis: Axes, values: pd.DataFrame, *, title: str, ylabel: str, higher_is_better: bool) -> None:
    """Draw one depth-trend panel with a line per minimum child weight."""
    style_axis(axis, title=title, ylabel=ylabel)
    for weight in MCW_VALUES:
        block = values.loc[values["min_child_weight"] == weight]
        axis.plot(
            block["max_depth"],
            block["mean"],
            marker=MCW_MARKERS[weight],
            color=MCW_COLORS[weight],
            linewidth=1.4,
            markersize=4.5,
            # Two minimum-child-weight settings can coincide to within 0.002 at
            # a given depth; slight transparency keeps both markers visible
            # instead of letting the later one hide the earlier.
            alpha=0.85,
            label=f"MCW {weight}",
        )
        paper = block.loc[block["max_depth"] == PAPER_DEPTH]
        if not paper.empty:
            axis.plot(
                [PAPER_DEPTH],
                paper["mean"].to_numpy(),
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=HIGHLIGHT,
                markeredgewidth=1.2,
                zorder=5,
            )
    axis.set_xticks(list(DEPTHS))
    axis.set_xlabel("maximum depth", fontsize=7.5)
    axis.legend(
        fontsize=6.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        handlelength=1.2,
        columnspacing=1.0,
    )
    # The three minimum-child-weight values nearly coincide at depth 3, so the
    # spread is stated numerically instead of being left to visual inspection.
    at_paper_depth = values.loc[values["max_depth"] == PAPER_DEPTH, "mean"]
    chosen = values.loc[values["config_label"] == f"d{PAPER_DEPTH}_mcw{PAPER_MCW}", "mean"]
    lines = [f"spread at depth {PAPER_DEPTH}: {at_paper_depth.max() - at_paper_depth.min():.4f}"]
    if not chosen.empty:
        lines.insert(0, f"paper config: {float(chosen.iloc[0]):.3f}")
    axis.text(
        0.03,
        0.985,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=7,
        color=NEUTRAL,
        va="top",
        linespacing=1.25,
    )


def add_bar_panel(axis: Axes, values: pd.DataFrame, *, title: str, ylabel: str, better_is_low: bool) -> None:
    """Draw one bar panel over the six configurations."""
    style_axis(axis, title=title, ylabel=ylabel)
    ordered = values.set_index("config_label")["mean"].reindex(list(BAR_CONFIG_ORDER))
    positions = range(len(BAR_CONFIG_ORDER))
    colours = [GADS_BLUE if name == f"d{PAPER_DEPTH}_mcw{PAPER_MCW}" else NEUTRAL for name in BAR_CONFIG_ORDER]
    axis.bar(positions, ordered.to_numpy(), color=colours, width=0.62)
    axis.set_xticks(list(positions))
    axis.set_xticklabels([BAR_CONFIG_DISPLAY[name] for name in BAR_CONFIG_ORDER], fontsize=7.5)
    axis.set_xlabel("depth / MCW", fontsize=7.5)
    best = ordered.idxmin() if better_is_low else ordered.idxmax()
    chosen = ordered.reindex([f"d{PAPER_DEPTH}_mcw{PAPER_MCW}"]).iloc[0]
    axis.text(
        0.03,
        0.94,
        f"paper config: {chosen:.3f}\nbest: {ordered[best]:.3f}",
        transform=axis.transAxes,
        fontsize=7,
        color=NEUTRAL,
        va="top",
        linespacing=1.25,
    )


def build_figure(frame: pd.DataFrame, output: Path) -> None:
    """Render the four-panel hyperparameter sensitivity figure."""
    auc = pooled_metric(frame, "kpi_auc", LOCALIZATION_SCENARIOS)
    gap = pooled_metric(frame, "train_oof_auc_gap", LOCALIZATION_SCENARIOS)
    mrr = pooled_metric(frame, "mrr", LOCALIZATION_SCENARIOS)
    far1 = pooled_metric(frame, "far_at_1", FALSE_ALARM_SCENARIOS)

    # The canvas is drawn at close to the width it will be printed at (a
    # two-column span of a two-column page). Drawing it much wider and letting
    # LaTeX scale it down would shrink the tick labels and annotations below the
    # legible limit along with the panels.
    figure, grid = plt.subplots(2, 2, figsize=(7.0, 3.9))
    panels = grid.ravel()
    add_line_panel(panels[0], auc, title="Out-of-fold KPI AUC", ylabel="AUC", higher_is_better=True)
    add_line_panel(panels[1], mrr, title="Pooled MRR (S1--S4-main)", ylabel="MRR", higher_is_better=True)
    add_bar_panel(panels[2], gap, title="Train--OOF gap", ylabel="gap", better_is_low=True)
    add_bar_panel(panels[3], far1, title="Pooled FAR@1 (S4)", ylabel="FAR@1", better_is_low=True)
    # A small absolute spread must not read as a large effect: the line panels
    # share one scale so the AUC panel (spread 0.004) and the MRR panel
    # (spread 0.048) are directly comparable.
    for axis in panels[:2]:
        axis.set_ylim(LINE_PANEL_RANGE)
    for axis in panels[2:]:
        axis.set_ylim(0.0, BAR_PANEL_MAX)
    figure.suptitle(
        "Dataset B hyperparameter grid; circled point and blue bar mark the paper configuration",
        fontsize=7.5,
        y=1.005,
    )
    figure.tight_layout(h_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}")
    for name, series in (("kpi_auc", auc), ("gap", gap), ("mrr", mrr), ("far@1", far1)):
        rendered = ", ".join(
            f"{label}={mean:.3f}" for label, mean in zip(series["config_label"], series["mean"], strict=True)
        )
        print(f"{name}: {rendered}")


def main() -> None:
    """Render the figure from the persisted hyperparameter grid."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("results/bdcc_revision/pack1_seeds_hyperparams/hyperparameter_grid/metrics_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/bdcc/revision_r1/images/fig6_hyperparameter_effect.png"),
    )
    args = parser.parse_args()
    build_figure(load_grid(args.grid), args.output)


if __name__ == "__main__":
    main()

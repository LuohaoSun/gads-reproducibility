#!/usr/bin/env python3
"""Generate paper-ready figures from summarized experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import cast

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from shap_diff_analysis.statistics import S4_NULL_SCENARIO, load_experiment_metrics, load_experiment_rankings

MAIN_METRICS = ("mrr", "hr_at_1", "hr_at_3", "hr_at_5")
FAR_METRICS = ("far_at_1", "far_at_5")
SUMMARY_COLUMNS = ("dataset", "scenario", "method", "metric", "mean", "ci_low", "ci_high", "applicable")
RANKING_COLUMNS = ("dataset", "scenario", "seed", "method", "equipment", "feature", "rank")
FIG2_LOCALIZATION_SCENARIOS = ("S1", "S2", "S3", "S4-main")
FIG2_DEFAULT_DATASET = "dataset_a"
PRIMARY_COHORT_METHODS = ("GADS", "KS-X", "PSI-X", "Domain-SHAP", "Wasserstein-X", "Global-SHAP")
TRADEOFF_X_SCENARIO = "S2"
TRADEOFF_X_METRIC = "mrr"
TRADEOFF_Y_SCENARIO = "S4-null"
TRADEOFF_Y_METRIC = "far_at_1"
SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "S1": "S1\nParameter drift",
    "S2": "S2\nMechanism shift",
    "S3": "S3\nMixed faults",
    "S4-main": "S4-main\nRoot cause +\nharmless drift",
}
METHOD_COLORS: dict[str, str] = {
    "GADS": "#1d4ed8",
    "KS-X": "#d97706",
    "PSI-X": "#059669",
    "Domain-SHAP": "#dc2626",
    "Wasserstein-X": "#7c3aed",
    "Global-SHAP": "#475569",
}
METHOD_MARKERS: dict[str, str] = {
    "GADS": "o",
    "KS-X": "s",
    "PSI-X": "^",
    "Domain-SHAP": "D",
    "Wasserstein-X": "v",
    "Global-SHAP": "P",
}
DEFAULT_FIG1_SVG = Path("assets/fig1_framework.svg")
_DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "dataset_a": ("dataset_a", "dataset a", "a"),
    "dataset_b": ("dataset_b", "dataset b", "b"),
}


def normalize_dataset_label(label: str) -> str:
    """Normalize dataset labels for alias-tolerant matching."""
    return label.strip().lower().replace("-", " ").replace("_", " ")


def dataset_labels_match(requested: str, actual: str) -> bool:
    """Return True when two dataset labels refer to the same dataset."""
    requested_norm = normalize_dataset_label(requested)
    actual_norm = normalize_dataset_label(actual)
    if requested_norm == actual_norm:
        return True
    for aliases in _DATASET_ALIASES.values():
        alias_norms = {normalize_dataset_label(alias) for alias in aliases}
        if requested_norm in alias_norms and actual_norm in alias_norms:
            return True
    return False


def filter_rankings_by_dataset(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return ranking rows whose dataset label matches the requested slice."""
    mask = frame["dataset"].astype(str).map(lambda value: dataset_labels_match(dataset, value))
    return frame.loc[mask]


def scenario_display_name(scenario: str) -> str:
    """Map internal scenario codes to reader-facing axis labels."""
    return SCENARIO_DISPLAY_NAMES.get(scenario, scenario)


def dataset_display_name(dataset_key: str, frame: pd.DataFrame) -> str:
    """Resolve a reader-facing dataset label from a slice key and data frame."""
    subset = filter_rankings_by_dataset(frame, dataset_key) if "dataset" in frame.columns else frame
    if not subset.empty and "dataset" in subset.columns:
        return str(subset["dataset"].iloc[0])
    normalized = normalize_dataset_label(dataset_key)
    if normalized in {normalize_dataset_label("dataset_a"), "a"}:
        return "Dataset A"
    if normalized in {normalize_dataset_label("dataset_b"), "b"}:
        return "Dataset B"
    return dataset_key


def parse_tradeoff_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract S2 mean MRR and S4-null mean FAR@1 for primary-cohort methods."""
    require_columns(summary, SUMMARY_COLUMNS, label="summary table")
    rows: list[dict[str, object]] = []
    for method in PRIMARY_COHORT_METHODS:
        x_row = summary.loc[
            (summary["scenario"] == TRADEOFF_X_SCENARIO)
            & (summary["method"] == method)
            & (summary["metric"] == TRADEOFF_X_METRIC)
        ]
        y_row = summary.loc[
            (summary["scenario"] == TRADEOFF_Y_SCENARIO)
            & (summary["method"] == method)
            & (summary["metric"] == TRADEOFF_Y_METRIC)
        ]
        if x_row.empty or y_row.empty:
            raise ValueError(f"missing trade-off metrics for method {method!r}")
        x_mean = float(x_row.iloc[0]["mean"])
        y_mean = float(y_row.iloc[0]["mean"])
        if not np.isfinite(x_mean) or not np.isfinite(y_mean):
            raise ValueError(f"non-finite trade-off coordinates for method {method!r}: mrr={x_mean}, far={y_mean}")
        rows.append(
            {
                "method": method,
                "s2_mrr": x_mean,
                "s4_null_far_at_1": y_mean,
            }
        )
    extra = summary.loc[
        (summary["scenario"].isin((TRADEOFF_X_SCENARIO, TRADEOFF_Y_SCENARIO)))
        & (~summary["method"].isin(PRIMARY_COHORT_METHODS))
    ]
    if not extra.empty:
        excluded = sorted({str(method) for method in extra["method"].unique()})
        raise ValueError(f"unexpected non-primary methods in trade-off slice: {excluded}")
    return pd.DataFrame(rows)


def resolve_paper_final_dir(input_dir: Path, explicit: Path | None) -> Path:
    """Resolve the paper_final root from CLI input."""
    if explicit is not None:
        return explicit
    if input_dir.name.startswith("matrix_"):
        return input_dir.parent
    return input_dir


def resolve_fig2_dataset(
    rankings: pd.DataFrame,
    requested: str | None,
    *,
    explicit: bool,
) -> str:
    """Resolve the dataset label for Fig. 2 from rankings and CLI input."""
    require_columns(rankings, RANKING_COLUMNS, label="rankings")
    available_labels = sorted({str(label) for label in rankings["dataset"].unique()})

    if explicit:
        if requested is None:
            raise ValueError("fig2 dataset must be a non-empty string when --fig2-dataset is set")
        if filter_rankings_by_dataset(rankings, requested).empty:
            raise ValueError(f"fig2 dataset {requested!r} not found in rankings; available labels: {available_labels}")
        return requested

    for label in available_labels:
        if dataset_labels_match(FIG2_DEFAULT_DATASET, label):
            return FIG2_DEFAULT_DATASET

    if len(available_labels) == 1:
        return available_labels[0]

    raise ValueError(
        "fig2 dataset not specified and dataset_a not found in rankings; "
        f"available labels: {available_labels}. Pass --fig2-dataset explicitly."
    )


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    """Raise when a frame is empty or lacks required columns."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} is empty")


def load_summary_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet summary table for plotting."""
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("summary path must be .parquet or .csv")
    require_columns(frame, SUMMARY_COLUMNS, label="summary table")
    return frame


def filter_plot_metrics(summary: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Return applicable summary rows for the requested metrics."""
    subset = summary.loc[summary["metric"].isin(metrics)].copy()
    if subset.empty:
        raise ValueError(f"summary has no rows for metrics: {metrics}")
    applicable = subset["applicable"].astype(bool)
    if not applicable.all():
        subset = subset.loc[applicable]
    if subset.empty:
        raise ValueError(f"no applicable summary rows remain for metrics: {metrics}")
    return subset


def plot_grouped_metric_bars(
    summary: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    output_path: Path,
    title: str,
) -> Path:
    """Render grouped bar charts with bootstrap error bars."""
    data = filter_plot_metrics(summary, metrics)
    data = data.assign(
        label=data["scenario"].astype(str) + "\n" + data["metric"].astype(str),
        err_low=data["mean"] - data["ci_low"],
        err_high=data["ci_high"] - data["mean"],
    )
    sns.set_theme(style="whitegrid", context="paper")
    g = sns.catplot(
        data=data,
        kind="bar",
        x="label",
        y="mean",
        hue="method",
        col="dataset",
        errorbar=None,
        height=4.5,
        aspect=1.2,
        palette="tab10",
    )
    for ax in g.axes.flat:
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("scenario / metric")
        ax.set_ylabel("score")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    return output_path


def plot_far_bars(summary: pd.DataFrame, *, output_path: Path) -> Path:
    """Render false-alarm rate bar charts."""
    return plot_grouped_metric_bars(
        summary,
        metrics=FAR_METRICS,
        output_path=output_path,
        title="False-alarm rate by scenario",
    )


def top_k_feature_sets(rankings: pd.DataFrame, *, top_k: int, view: str = "process") -> pd.DataFrame:
    """Collect Top-K feature sets per seed, method and equipment."""
    require_columns(rankings, RANKING_COLUMNS, label="rankings")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    frame = rankings.copy()
    if "view" in frame.columns:
        frame = frame.loc[frame["view"] == view]
    if frame.empty:
        raise ValueError(f"no rankings remain after filtering view={view!r}")
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "scenario", "seed", "method", "equipment"]
    for key, group in frame.groupby(group_cols, sort=False):
        dataset, scenario, seed, method, equipment = cast(tuple[str, str, int, str, str], key)
        ordered = group.sort_values(["rank", "feature"], kind="stable").head(top_k)
        features = frozenset(ordered["feature"].astype(str))
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "seed": seed,
                "method": method,
                "equipment": equipment,
                "top_k": top_k,
                "features": features,
            }
        )
    return pd.DataFrame(rows)


def seed_topk_jaccard(rankings: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Compute pairwise Top-K Jaccard similarities across seeds."""
    sets = top_k_feature_sets(rankings, top_k=top_k)
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "scenario", "method", "equipment", "top_k"]
    for key, group in sets.groupby(group_cols, sort=False):
        dataset, scenario, method, equipment, k = cast(tuple[str, str, str, str, int], key)
        seeds = sorted(group["seed"].unique())
        if len(seeds) < 2:
            continue
        values = group.set_index("seed")["features"]
        for left, right in ((a, b) for i, a in enumerate(seeds) for b in seeds[i + 1 :]):
            left_set = values[left]
            right_set = values[right]
            union = left_set | right_set
            jaccard = float(len(left_set & right_set) / len(union)) if union else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "method": method,
                    "equipment": equipment,
                    "top_k": k,
                    "seed_left": left,
                    "seed_right": right,
                    "jaccard": jaccard,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("insufficient seeds to compute Top-K Jaccard similarities")
    return result


def mean_seed_pair_jaccard(
    rankings: pd.DataFrame,
    *,
    dataset: str,
    scenario: str | None,
    method: str,
    equipment: str | None,
    top_k: int,
    view: str = "process",
) -> pd.DataFrame:
    """Aggregate mean pairwise Top-K Jaccard similarities for one ranking slice."""
    frame = rankings.copy()
    if "view" in frame.columns:
        frame = frame.loc[frame["view"] == view]
    frame = filter_rankings_by_dataset(frame, dataset)
    frame = frame.loc[frame["method"] == method]
    if scenario is not None:
        frame = frame.loc[frame["scenario"] == scenario]
    if equipment is not None:
        frame = frame.loc[frame["equipment"] == equipment]
    if frame.empty:
        raise ValueError(
            "no rankings match fig2 slice "
            f"(dataset={dataset!r}, scenario={scenario!r}, method={method!r}, "
            f"equipment={equipment!r}, view={view!r})"
        )
    jaccard = seed_topk_jaccard(frame, top_k=top_k)
    grouped = jaccard.groupby(["scenario", "equipment"], as_index=False)["jaccard"].mean()
    if grouped.empty:
        raise ValueError("no seed-pair Jaccard values computed for fig2 slice")
    scenario_summary = grouped.groupby("scenario", as_index=False)["jaccard"].mean()
    return cast(pd.DataFrame, scenario_summary).rename(columns={"jaccard": "mean_jaccard"})


def plot_fig2_gads_seed_stability(
    rankings: pd.DataFrame,
    *,
    output_path: Path,
    dataset: str = "dataset_a",
    scenario: str | None = None,
    method: str = "GADS",
    equipment: str | None = None,
    top_k: int = 5,
    view: str = "process",
) -> Path:
    """Render paper Fig.~2: GADS process-view Top-K ranking overlap on Dataset A."""
    frame = rankings.copy()
    if scenario is not None:
        frame = frame.loc[frame["scenario"] == scenario]
    else:
        frame = frame.loc[frame["scenario"].isin(FIG2_LOCALIZATION_SCENARIOS)]
    summary = mean_seed_pair_jaccard(
        frame,
        dataset=dataset,
        scenario=None,
        method=method,
        equipment=equipment,
        top_k=top_k,
        view=view,
    )
    ordered = summary.sort_values("scenario", kind="stable").copy()
    ordered["scenario_label"] = ordered["scenario"].map(scenario_display_name)
    display_dataset = dataset_display_name(dataset, rankings)

    # Professional academic styling
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)

    # Harmonious professional color palette for localization scenarios
    scenario_colors = {
        "S1": "#205493",  # Deep Navy
        "S2": "#2874a6",  # Sapphire Blue (Highlight for mechanism shift)
        "S3": "#117a8b",  # Teal Blue
        "S4-main": "#47748b",  # Slate Cyan
    }
    bar_colors = [scenario_colors.get(sc, "#205493") for sc in ordered["scenario"]]

    # Draw bars with subtle outlines and ideal width
    bar_width = 0.52
    bars = ax.bar(
        range(len(ordered)),
        ordered["mean_jaccard"],
        width=bar_width,
        color=bar_colors,
        edgecolor="#0f172a",
        linewidth=1.1,
        zorder=3,
    )

    # Refined axes limits and grid
    max_val = float(ordered["mean_jaccard"].max()) if not ordered.empty else 0.4
    y_max = max(0.50, np.ceil((max_val + 0.10) * 10) / 10)
    ax.set_ylim(0.0, y_max)
    ax.set_yticks(np.arange(0.0, y_max + 0.01, 0.10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    # Soft horizontal grid lines only
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#cbd5e1", alpha=0.75, zorder=0)

    # X-axis ticks and labels
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["scenario_label"], fontsize=9.5, fontweight="500", color="#1e293b")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=9.5, colors="#334155")

    # Titles and labels
    ax.set_ylabel(
        f"Mean Pairwise Top-{top_k} Jaccard Overlap",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
        labelpad=8,
    )
    ax.set_xlabel("Fault Localization Scenario", fontsize=10.5, fontweight="bold", color="#0f172a", labelpad=10)
    ax.set_title(
        f"Top-{top_k} Process-Feature Ranking Overlap Across Seeds ({display_dataset})",
        fontsize=11.5,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )

    # Clean border (remove top and right spines, polish remaining spines)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color("#64748b")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["bottom"].set_linewidth(1.0)

    # Elegant data labels above bars with micro badge style
    for bar, val in zip(bars, ordered["mean_jaccard"], strict=True):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.3",
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
                "linewidth": 0.75,
            },
            zorder=4,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


TRADEOFF_PANEL_ANNOTATIONS: tuple[dict[str, tuple[float, float]], ...] = (
    {  # first panel (Dataset A): three classical methods cluster at (0.025, 1.0)
        "GADS": (0.76, 0.22),
        "KS-X": (0.16, 0.88),
        "PSI-X": (0.16, 0.80),
        "Domain-SHAP": (0.16, 0.96),
        "Wasserstein-X": (0.48, 0.90),
        "Global-SHAP": (0.18, 0.06),
    },
    {  # second panel (Dataset B): cluster at (0.006-0.034, 1.0); Wasserstein-X at (0.02, 0.8)
        "GADS": (0.78, 0.58),
        "KS-X": (0.16, 0.88),
        "PSI-X": (0.16, 0.80),
        "Domain-SHAP": (0.16, 0.96),
        "Wasserstein-X": (0.16, 0.70),
        "Global-SHAP": (0.22, 0.06),
    },
)


def plot_primary_cohort_tradeoff(
    points_by_dataset: dict[str, pd.DataFrame],
    *,
    output_path: Path,
) -> Path:
    """Render the full-width two-panel S2 MRR vs S4-null FAR@1 trade-off figure.

    Panels use direct labeled points (no legend, suptitle, or reference
    diagonal); label layouts are deterministic per panel position.
    """
    if not points_by_dataset:
        raise ValueError("points_by_dataset must contain at least one dataset panel")
    sns.set_theme(style="white", context="paper")
    n_panels = len(points_by_dataset)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4.0), sharex=True, sharey=True, dpi=300)
    if n_panels == 1:
        axes = [axes]
    for panel_index, (ax, (panel_title, points)) in enumerate(zip(axes, points_by_dataset.items(), strict=True)):
        require_columns(points, ("method", "s2_mrr", "s4_null_far_at_1"), label="trade-off points")
        if panel_index >= len(TRADEOFF_PANEL_ANNOTATIONS):
            raise ValueError(f"no annotation layout configured for trade-off panel index {panel_index}")
        label_positions = TRADEOFF_PANEL_ANNOTATIONS[panel_index]
        missing = {str(method) for method in points["method"]} - set(label_positions)
        if missing:
            raise ValueError(f"no annotation position configured for methods: {sorted(missing)}")

        # Desirable quadrant background patch (Lower-Right: High MRR, Low FAR)
        desirable_rect = Rectangle(
            (0.5, 0.0),
            0.5,
            0.5,
            facecolor="#f0fdf4",
            edgecolor="#86efac",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=0,
        )
        ax.add_patch(desirable_rect)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

        # Subtle grid
        ax.grid(True, linestyle="--", linewidth=0.6, color="#e2e8f0", alpha=0.8, zorder=1)

        # Panel title
        ax.set_title(panel_title, fontsize=11, fontweight="bold", color="#0f172a", pad=10)

        # Draw points and annotations
        for _, row in points.iterrows():
            method = str(row["method"])
            x_val = float(row["s2_mrr"])
            y_val = float(row["s4_null_far_at_1"])
            color = METHOD_COLORS.get(method, "#334e68")
            marker = METHOD_MARKERS.get(method, "o")
            is_ours = method == "GADS"

            # Point scatter with unclipped edges for boundary visibility
            ax.scatter(
                x_val,
                y_val,
                s=90 if is_ours else 70,
                color=color,
                marker=marker,
                edgecolor="#0f172a" if is_ours else "white",
                linewidth=1.3 if is_ours else 1.0,
                clip_on=False,
                zorder=4,
            )

            # Elegant badge annotation
            ax.annotate(
                method,
                (x_val, y_val),
                xytext=label_positions[method],
                textcoords="data",
                ha="left",
                va="center",
                fontsize=8.8,
                fontweight="bold" if is_ours else "500",
                color="#1e3a8a" if is_ours else color,
                bbox={
                    "boxstyle": "round,pad=0.25,rounding_size=0.3",
                    "facecolor": "#eff6ff" if is_ours else "#ffffff",
                    "edgecolor": "#93c5fd" if is_ours else "#cbd5e1",
                    "linewidth": 0.85,
                    "alpha": 0.95,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 0.75,
                    "color": color,
                    "shrinkA": 3,
                    "shrinkB": 4,
                },
                zorder=5,
            )

        # Border styling
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
            spine.set_linewidth(0.8)

    fig.supxlabel(
        "S2 Mean Reciprocal Rank (MRR)  [Higher is better $\\rightarrow$]",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
    )
    fig.supylabel(
        "S4-null Mean FAR@1  [$\\leftarrow$ Lower is better]",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
    )
    for ax in axes:
        ax.tick_params(labelsize=8.8, colors="#334155")
        ax.set_xlabel("")
        ax.set_ylabel("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_fig1_from_svg(*, svg_path: Path, output_path: Path, dpi: int = 300) -> Path:
    """Render the maintainable Fig.~1 SVG source to PDF (or PNG) output."""
    if not svg_path.is_file():
        raise FileNotFoundError(f"fig1 SVG source not found: {svg_path}")
    fmt = "pdf" if output_path.suffix.lower() == ".pdf" else "png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["rsvg-convert", "-f", fmt]
    if fmt == "png":
        command += ["-d", str(dpi), "-p", str(dpi)]
    command += ["-o", str(output_path), str(svg_path)]
    subprocess.run(command, check=True)
    return output_path


def plot_seed_jaccard_heatmap(jaccard: pd.DataFrame, *, output_path: Path) -> Path:
    """Plot mean seed-pair Jaccard similarities as a heatmap."""
    require_columns(
        jaccard,
        ("dataset", "scenario", "method", "equipment", "seed_left", "seed_right", "jaccard"),
        label="jaccard table",
    )
    grouped = jaccard.groupby(["dataset", "scenario", "method"], as_index=False)["jaccard"].mean()
    aggregated = cast(pd.DataFrame, grouped).rename(columns={"jaccard": "mean_jaccard"})
    pivot = aggregated.pivot_table(index="scenario", columns="method", values="mean_jaccard", aggfunc="first")
    if pivot.empty:
        raise ValueError("jaccard pivot table is empty")
    sns.set_theme(style="white", context="paper")
    plt.figure(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title(f"Seed stability (mean Top-{int(jaccard['top_k'].iloc[0])} Jaccard)")
    plt.xlabel("method")
    plt.ylabel("scenario")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_rank_stability_heatmap(rankings: pd.DataFrame, *, top_k: int, output_path: Path) -> Path:
    """Plot feature ranks across seeds for one representative slice."""
    require_columns(rankings, RANKING_COLUMNS, label="rankings")
    frame = rankings.copy()
    if "view" in frame.columns:
        frame = frame.loc[frame["view"] == "process"]
    if frame.empty:
        raise ValueError("no process-view rankings available for heatmap")
    group_sizes = cast(pd.Series, frame.groupby(["dataset", "scenario", "method"], sort=False).size())
    focus = cast(pd.DataFrame, group_sizes.reset_index(name="n_rows")).sort_values("n_rows", ascending=False).iloc[0]
    subset = frame.loc[
        (frame["dataset"] == focus["dataset"])
        & (frame["scenario"] == focus["scenario"])
        & (frame["method"] == focus["method"])
    ].copy()
    if subset.empty:
        raise ValueError("unable to select a ranking slice for heatmap")
    equipment = sorted(subset["equipment"].unique())[0]
    subset = subset.loc[subset["equipment"] == equipment]
    pivot = subset.pivot_table(index="feature", columns="seed", values="rank", aggfunc="first")
    top_features = subset.loc[subset["rank"] <= top_k, "feature"].value_counts().head(top_k).index.astype(str).tolist()
    pivot = pivot.loc[pivot.index.astype(str).isin(top_features)]
    if pivot.empty:
        raise ValueError("rank heatmap pivot is empty")
    sns.set_theme(style="white", context="paper")
    plt.figure(figsize=(max(6, pivot.shape[1] * 1.0), max(4, pivot.shape[0] * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r")
    plt.title(f"Rank stability ({focus['dataset']} / {focus['scenario']} / {focus['method']} / {equipment})")
    plt.xlabel("seed")
    plt.ylabel("feature")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_g3_intensity_curve(
    data: pd.DataFrame,
    *,
    output_path: Path,
) -> Path:
    """Render the G3 distance-by-intensity curve (GADS MRR vs mechanism scale)."""
    require_columns(data, ("distance", "level", "method", "mrr"), label="g3 table")
    gads = data.loc[data["method"] == "GADS"].copy()
    if gads.empty:
        raise ValueError("g3 table has no GADS rows")
    summary = gads.groupby(["distance", "level"], as_index=False)["mrr"].mean()
    distance_labels = {
        "wasserstein": "Wasserstein",
        "mean": "Mean difference",
        "kl": "KL divergence",
        "js": "JS divergence",
    }
    unknown = sorted(set(summary["distance"].unique()) - set(distance_labels))
    if unknown:
        raise ValueError(f"g3 table contains unknown distances: {unknown}")
    summary["distance_label"] = summary["distance"].map(distance_labels)
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(6.5, 4.2))
    ax = sns.lineplot(
        data=summary,
        x="level",
        y="mrr",
        hue="distance_label",
        marker="o",
        errorbar=None,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("mechanism scale (injection intensity)")
    ax.set_ylabel("MRR")
    ax.legend(title="distance", loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="experiment matrix root")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="stats summary table (default: <input-dir>/stats/stats_summary.parquet)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="aggregate metrics table for valid-run filtering (default: <input-dir>/aggregate/metrics.parquet)",
    )
    parser.add_argument(
        "--rankings-path",
        type=Path,
        default=None,
        help="aggregate rankings table (default: <input-dir>/aggregate/rankings.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="figure output directory (default: <input-dir>/figures)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff for stability figures")
    parser.add_argument(
        "--fig2-dataset",
        default=None,
        help=(
            "dataset slice for paper Fig. 2 (default: dataset_a when present, otherwise the sole dataset in rankings)"
        ),
    )
    parser.add_argument(
        "--fig2-scenario",
        default=None,
        help="optional scenario filter for Fig. 2; default aggregates localization scenarios",
    )
    parser.add_argument(
        "--fig2-method",
        default="GADS",
        help="method slice for paper Fig. 2 (default: GADS)",
    )
    parser.add_argument(
        "--fig2-equipment",
        default=None,
        help="optional equipment filter for Fig. 2; default averages over equipment",
    )
    parser.add_argument(
        "--fig2-view",
        default="process",
        help="ranking view for Fig. 2 (default: process)",
    )
    parser.add_argument(
        "--fig2-output",
        type=Path,
        default=None,
        help="Fig. 2 output path (default: <output-dir>/fig2_gads_seed_stability.png)",
    )
    parser.add_argument(
        "--paper-final-dir",
        type=Path,
        default=None,
        help="paper_final root for trade-off panels (default: parent of matrix input-dir)",
    )
    parser.add_argument(
        "--tradeoff-output",
        type=Path,
        default=None,
        help=(
            "primary-cohort trade-off output path (.pdf recommended; default: <output-dir>/tradeoff_primary_cohort.png)"
        ),
    )
    parser.add_argument(
        "--fig1-svg",
        type=Path,
        default=None,
        help=f"Fig. 1 SVG source (default: {DEFAULT_FIG1_SVG})",
    )
    parser.add_argument(
        "--fig1-output",
        type=Path,
        default=None,
        help="Fig. 1 output path (.pdf recommended; default: <output-dir>/fig1_framework.pdf)",
    )
    parser.add_argument(
        "--fig-types",
        nargs="+",
        choices=("bars", "far", "jaccard", "rank-heatmap", "fig2", "fig1", "tradeoff", "g3", "all"),
        default=("all",),
        help="figure groups to render (g3 is not part of 'all'; it requires --g3-parquet)",
    )
    parser.add_argument(
        "--g3-parquet",
        type=Path,
        default=None,
        help=(
            "merged G3 distance-by-intensity parquet (required when 'g3' is in --fig-types; "
            "see the README section on the G3 distance-by-intensity sweep for how to build it)"
        ),
    )
    parser.add_argument(
        "--g3-output",
        type=Path,
        default=None,
        help="G3 intensity-curve output path (default: <output-dir>/fig4_g3_intensity.png)",
    )
    return parser


def main() -> None:
    """Render selected paper figures from summary and ranking tables."""
    args = build_parser().parse_args()
    summary_path = args.summary_path or (args.input_dir / "stats" / "stats_summary.parquet")
    metrics_path = args.metrics_path or (args.input_dir / "aggregate" / "metrics.parquet")
    rankings_path = args.rankings_path or (args.input_dir / "aggregate" / "rankings.parquet")
    output_dir = args.output_dir or (args.input_dir / "figures")
    summary = load_summary_table(summary_path)
    metrics = load_experiment_metrics(metrics_path)
    rankings = load_experiment_rankings(rankings_path, metrics=metrics)

    selected = set(args.fig_types)
    if "all" in selected:
        selected = {"bars", "far", "jaccard", "rank-heatmap", "fig2"}

    outputs: dict[str, str] = {}
    if "bars" in selected:
        outputs["main_metrics"] = str(
            plot_grouped_metric_bars(
                summary,
                metrics=MAIN_METRICS,
                output_path=output_dir / "main_metrics_bars.png",
                title="Main localization metrics",
            )
        )
    if "far" in selected:
        far_summary = summary.loc[
            (summary["metric"].isin(FAR_METRICS))
            | ((summary["scenario"] == S4_NULL_SCENARIO) & summary["metric"].isin(FAR_METRICS))
        ]
        if far_summary.empty:
            raise ValueError("summary has no FAR metrics to plot")
        outputs["far"] = str(plot_far_bars(summary, output_path=output_dir / "far_bars.png"))
    if "jaccard" in selected:
        jaccard = seed_topk_jaccard(rankings, top_k=args.top_k)
        outputs["jaccard"] = str(plot_seed_jaccard_heatmap(jaccard, output_path=output_dir / "seed_topk_jaccard.png"))
    if "rank-heatmap" in selected:
        outputs["rank_heatmap"] = str(
            plot_rank_stability_heatmap(rankings, top_k=args.top_k, output_path=output_dir / "rank_stability.png")
        )
    if "fig2" in selected:
        fig2_path = args.fig2_output or (output_dir / "fig2_gads_seed_stability.png")
        fig2_dataset = resolve_fig2_dataset(
            rankings,
            args.fig2_dataset,
            explicit=args.fig2_dataset is not None,
        )
        outputs["fig2"] = str(
            plot_fig2_gads_seed_stability(
                rankings,
                output_path=fig2_path,
                dataset=fig2_dataset,
                scenario=args.fig2_scenario,
                method=args.fig2_method,
                equipment=args.fig2_equipment,
                top_k=args.top_k,
                view=args.fig2_view,
            )
        )

    if "fig1" in selected:
        repo_root = Path(__file__).resolve().parents[1]
        fig1_svg = args.fig1_svg or (repo_root / DEFAULT_FIG1_SVG)
        fig1_output = args.fig1_output or (output_dir / "fig1_framework.pdf")
        outputs["fig1"] = str(render_fig1_from_svg(svg_path=fig1_svg, output_path=fig1_output))

    if "tradeoff" in selected:
        paper_final = resolve_paper_final_dir(args.input_dir, args.paper_final_dir)
        tradeoff_specs = (
            ("Dataset A", paper_final / "matrix_a_xgb/stats/stats_summary.parquet"),
            ("Dataset B", paper_final / "matrix_b_xgb/stats/stats_summary.parquet"),
        )
        points_by_dataset: dict[str, pd.DataFrame] = {}
        for panel_title, summary_file in tradeoff_specs:
            if not summary_file.is_file():
                raise FileNotFoundError(f"trade-off summary not found: {summary_file}")
            points_by_dataset[panel_title] = parse_tradeoff_points(load_summary_table(summary_file))
        tradeoff_output = args.tradeoff_output or (output_dir / "tradeoff_primary_cohort.png")
        outputs["tradeoff"] = str(plot_primary_cohort_tradeoff(points_by_dataset, output_path=tradeoff_output))

    if "g3" in selected:
        if args.g3_parquet is None:
            raise ValueError("--g3-parquet is required when 'g3' is in --fig-types")
        if not args.g3_parquet.is_file():
            raise FileNotFoundError(f"g3 parquet not found: {args.g3_parquet}")
        g3_output = args.g3_output or (output_dir / "fig4_g3_intensity.png")
        outputs["g3"] = str(plot_g3_intensity_curve(pd.read_parquet(args.g3_parquet), output_path=g3_output))

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "rankings_path": str(rankings_path),
                "output_dir": str(output_dir),
                "figures": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

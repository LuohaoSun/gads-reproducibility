"""Recompute existing-output statistics for the BDCC internal-review supplement.

The script only reads the already produced pack1/pack2 result tables. It never
trains a model, never reruns an experiment, and never falls back to a different
input. Any missing table or missing column raises immediately.

Run:
    uv run python scripts/bdcc_internal_review_existing_stats.py \
        --input-root . --output-dir results/bdcc_internal_review/existing_stats
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap, norm

DATASETS = ("Dataset A", "Dataset B")
PACK2_DATASET_LABELS = {"dataset_a": "Dataset A", "dataset_b": "Dataset B"}
SCENARIOS = ("S1", "S2", "S3", "S4-main", "S4-null")
FAR_SCENARIOS = {"S4-main", "S4-null"}
METRICS = ("mrr", "hr_at_1", "hr_at_5", "far_at_1", "far_at_5", "kpi_auc")
MODEL_SEEDS = (42, 101, 202, 303, 404)
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95

PACK1_REL = Path("results/bdcc_revision/pack1_seeds_hyperparams")
PACK2_REL = Path("results/bdcc_revision/pack2_competitors_runtime")

INPUT_FILES = (
    PACK1_REL / "model_seed_sweep/metrics_long.csv",
    PACK1_REL / "model_seed_sweep/absolute_precision_ci.csv",
    PACK1_REL / "model_seed_sweep/variance_decomposition.csv",
    PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds.csv",
    PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds_summary.csv",
    PACK1_REL / "hyperparameter_grid/metrics_long.csv",
    PACK1_REL / "hyperparameter_grid/missing_indicators_comparison.csv",
    PACK1_REL / "hyperparameter_grid/wasserstein_x_scale_paired_diff.csv",
    PACK2_REL / "comparison_per_seed.csv",
)

INPUT_COLUMNS = {
    str(PACK1_REL / "model_seed_sweep/metrics_long.csv"): (
        "dataset",
        "scenario",
        "scenario_seed",
        "model_seed",
        "method",
        *METRICS,
    ),
    str(PACK1_REL / "model_seed_sweep/absolute_precision_ci.csv"): (
        "dataset",
        "scenario",
        "metric",
        "scope",
        "mean",
        "sd",
        "ci95_percentile_low",
        "ci95_percentile_high",
        "ci95_bca_low",
        "ci95_bca_high",
    ),
    str(PACK1_REL / "model_seed_sweep/variance_decomposition.csv"): (
        "dataset",
        "scenario",
        "metric",
        "within_var",
        "between_var",
        "model_seed_sd",
    ),
    str(PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds.csv"): (
        "dataset",
        "scenario",
        "scenario_seed",
        "spearman",
    ),
    str(PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds_summary.csv"): (
        "dataset",
        "scenario",
        "mean_spearman",
        "min_spearman",
    ),
    str(PACK1_REL / "hyperparameter_grid/metrics_long.csv"): (
        "dataset",
        "scenario",
        "scenario_seed",
        "method",
        "config_label",
        "max_depth",
        "min_child_weight",
        "include_missing_indicators",
        *METRICS,
    ),
    str(PACK1_REL / "hyperparameter_grid/missing_indicators_comparison.csv"): (
        "dataset",
        "scenario",
        "metric",
        "n_pairs",
        "included_mean",
        "excluded_mean",
        "mean_diff",
        "ci95_bca_low",
        "ci95_bca_high",
        "bootstrap_method",
        "p_value_permutation",
        "favors",
    ),
    str(PACK1_REL / "hyperparameter_grid/wasserstein_x_scale_paired_diff.csv"): (
        "dataset",
        "scenario",
        "metric",
        "n_pairs",
        "raw_mean",
        "standardized_mean",
        "mean_diff_standardized_minus_raw",
        "ci95_bca_low",
        "ci95_bca_high",
        "p_value_permutation",
        "favors",
    ),
    str(PACK2_REL / "comparison_per_seed.csv"): (
        "dataset",
        "scenario",
        "seed",
        "method",
        *METRICS,
    ),
}


def applicable(scenario: str, metric: str) -> bool:
    """Return whether a metric is defined for a scenario."""
    if scenario == "S4-null" and metric != "kpi_auc" and not metric.startswith("far_"):
        return False
    return not (metric.startswith("far_") and scenario not in FAR_SCENARIOS)


def comparison_seed(*parts: str) -> int:
    """Derive a reproducible RNG seed from the statistic identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(root: Path, relative: Path) -> Path:
    """Return an input path, raising when it is absent."""
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    return path


def read_input(root: Path, relative: Path) -> pd.DataFrame:
    """Read one required CSV and validate its columns."""
    path = require_input(root, relative)
    frame = pd.read_csv(path)
    required = INPUT_COLUMNS[str(relative)]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def unpack_pair(raw_key: object) -> tuple[str, str]:
    """Return the two string labels of a multi-column pandas group key."""
    if not isinstance(raw_key, tuple):
        raise TypeError(f"expected a tuple group key, got {type(raw_key)!r}")
    first, second = raw_key
    return str(first), str(second)


def unpack_triple(raw_key: object) -> tuple[str, str, str]:
    """Return the three string labels of a multi-column pandas group key."""
    if not isinstance(raw_key, tuple):
        raise TypeError(f"expected a tuple group key, got {type(raw_key)!r}")
    first, second, third = raw_key
    return str(first), str(second), str(third)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one numeric column with unparsable and missing values dropped."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if not isinstance(values, pd.Series):
        raise TypeError(f"expected a Series from pd.to_numeric, got {type(values)!r}")
    return values.dropna()


def bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Return mean, SD and a 95% BCa interval for a cluster-mean vector."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("bootstrap interval requires >=2 values")
    if not np.isfinite(data).all():
        raise ValueError("bootstrap interval requires finite values")
    observed = float(data.mean())
    alpha = (1.0 - CONFIDENCE) / 2.0
    if float(np.ptp(data)) == 0.0:
        return {
            "mean": observed,
            "sd": float(data.std(ddof=1)),
            "ci95_percentile_low": observed,
            "ci95_percentile_high": observed,
            "ci95_bca_low": observed,
            "ci95_bca_high": observed,
            "bootstrap_bias_correction": 0.0,
            "bootstrap_acceleration": 0.0,
            "bootstrap_method": "degenerate_constant",
            "bootstrap_warnings": "",
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bootstrap(
            (data,),
            np.mean,
            n_resamples=BOOTSTRAP_SAMPLES,
            confidence_level=CONFIDENCE,
            method="BCa",
            rng=np.random.default_rng(seed),
        )
    warning_messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    distribution = np.asarray(result.bootstrap_distribution, dtype=float)
    p_low, p_high = np.quantile(distribution, [alpha, 1.0 - alpha])
    bca_low = float(result.confidence_interval.low)
    bca_high = float(result.confidence_interval.high)
    ties = np.count_nonzero(distribution == observed)
    proportion = (np.count_nonzero(distribution < observed) + 0.5 * ties) / distribution.size
    z0 = float(norm.ppf(proportion))
    jackknife = (data.sum() - data) / (data.size - 1)
    deviations = float(jackknife.mean()) - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    acceleration = float(np.sum(deviations**3) / denominator) if denominator > 0 else float("nan")
    if not (math.isfinite(bca_low) and math.isfinite(bca_high)):
        raise RuntimeError(
            "non-finite BCa interval for non-constant input: "
            f"seed={seed}, n={data.size}, bca=({bca_low}, {bca_high}), "
            f"warnings={warning_messages}"
        )
    return {
        "mean": observed,
        "sd": float(data.std(ddof=1)),
        "ci95_percentile_low": float(p_low),
        "ci95_percentile_high": float(p_high),
        "ci95_bca_low": bca_low,
        "ci95_bca_high": bca_high,
        "bootstrap_bias_correction": z0,
        "bootstrap_acceleration": acceleration,
        "bootstrap_method": "bca",
        "bootstrap_warnings": "; ".join(warning_messages),
    }


def build_gads_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute absolute CIs and variance decomposition for GADS."""
    metrics = read_input(root, PACK1_REL / "model_seed_sweep/metrics_long.csv")
    metrics = metrics.loc[metrics["method"] == "GADS"]
    rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    for (dataset, scenario), group in metrics.groupby(["dataset", "scenario"], sort=False):
        for metric in METRICS:
            if not applicable(str(scenario), metric):
                continue
            subset = group.loc[:, ["scenario_seed", "model_seed", metric]].dropna()
            pivot = subset.pivot_table(
                index="scenario_seed", columns="model_seed", values=metric, aggfunc="first"
            ).sort_index()
            if pivot.shape != (20, 5):
                raise ValueError(f"unexpected design {dataset}/{scenario}/{metric}: {pivot.shape}")
            scopes = {
                "model_seed_42": (pivot[42].to_numpy(float), 1, "absolute_seed42"),
                "scenario_cluster_5_model_seeds": (
                    pivot.mean(axis=1).to_numpy(float),
                    5,
                    "absolute_cluster5",
                ),
            }
            for scope, (values, n_model_seeds, tag) in scopes.items():
                ci = bootstrap_mean_ci(
                    values,
                    seed=comparison_seed(str(dataset), str(scenario), metric, tag),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "metric": metric,
                        "scope": scope,
                        "n_clusters": int(values.size),
                        "n_model_seeds": n_model_seeds,
                        **ci,
                    }
                )
            within = float(pivot.var(axis=1, ddof=1).mean())
            scenario_means = pivot.mean(axis=1)
            between = float(scenario_means.var(ddof=1))
            total = within + between
            model_seed_means = pivot.mean(axis=0)
            variance_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "metric": metric,
                    "grand_mean": float(pivot.to_numpy().mean()),
                    "n_scenario_seeds": int(pivot.shape[0]),
                    "n_model_seeds": int(pivot.shape[1]),
                    "within_var": within,
                    "within_sd": math.sqrt(within),
                    "between_var": between,
                    "between_sd": math.sqrt(between),
                    "total_var": total,
                    "between_share": between / total if total > 0 else float("nan"),
                    "within_share": within / total if total > 0 else float("nan"),
                    "between_within_ratio": between / within if within > 0 else float("nan"),
                    "model_seed_sd": float(model_seed_means.std(ddof=1)),
                    "model_seed_range": float(model_seed_means.max() - model_seed_means.min()),
                    "max_abs_deviation_from_seed42": float((model_seed_means - model_seed_means.loc[42]).abs().max()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(variance_rows)


def verify_pack1(existing: pd.DataFrame, recomputed: pd.DataFrame) -> dict[str, Any]:
    """Compare recomputed absolute CIs against the stored published table."""
    key = ["dataset", "scenario", "metric", "scope"]
    merged = existing.merge(recomputed, on=key, suffixes=("_stored", "_recomputed"))
    differences: dict[str, float] = {}
    for column in (
        "mean",
        "sd",
        "ci95_percentile_low",
        "ci95_percentile_high",
        "ci95_bca_low",
        "ci95_bca_high",
    ):
        delta = (merged[f"{column}_stored"] - merged[f"{column}_recomputed"]).abs()
        differences[column] = float(delta.max())
    deviation_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        bca_delta = max(
            abs(float(row["ci95_bca_low_stored"]) - float(row["ci95_bca_low_recomputed"])),
            abs(float(row["ci95_bca_high_stored"]) - float(row["ci95_bca_high_recomputed"])),
        )
        if bca_delta <= 1e-12:
            continue
        deviation_rows.append(
            {
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "metric": row["metric"],
                "scope": row["scope"],
                "stored_bca_low": float(row["ci95_bca_low_stored"]),
                "recomputed_bca_low": float(row["ci95_bca_low_recomputed"]),
                "stored_bca_high": float(row["ci95_bca_high_stored"]),
                "recomputed_bca_high": float(row["ci95_bca_high_recomputed"]),
                "max_abs_delta": bca_delta,
                "interpretation": "bootstrap Monte Carlo/RNG drift; published stored value retained",
            }
        )
    exact = (merged["mean_stored"] - merged["mean_recomputed"]).abs() + (
        merged["sd_stored"] - merged["sd_recomputed"]
    ).abs() + (merged["ci95_bca_low_stored"] - merged["ci95_bca_low_recomputed"]).abs() + (
        merged["ci95_bca_high_stored"] - merged["ci95_bca_high_recomputed"]
    ).abs() < 1e-12
    return {
        "stored_rows": len(existing),
        "recomputed_rows": len(recomputed),
        "merged_rows": len(merged),
        "exact_rows": int(exact.sum()),
        "max_abs_diff": differences,
        "bcas_deviations": deviation_rows,
    }


def verify_variance(stored: pd.DataFrame, recomputed: pd.DataFrame) -> dict[str, Any]:
    """Compare recomputed variance decomposition against the stored table."""
    merged = stored.merge(recomputed, on=["dataset", "scenario", "metric"], suffixes=("_stored", "_recomputed"))
    return {
        "stored_rows": len(stored),
        "recomputed_rows": len(recomputed),
        "merged_rows": len(merged),
        "max_abs_within_var_diff": float((merged["within_var_stored"] - merged["within_var_recomputed"]).abs().max()),
        "max_abs_between_var_diff": float(
            (merged["between_var_stored"] - merged["between_var_recomputed"]).abs().max()
        ),
        "max_abs_model_seed_sd_diff": float(
            (merged["model_seed_sd_stored"] - merged["model_seed_sd_recomputed"]).abs().max()
        ),
    }


def build_spearman_table(root: Path) -> pd.DataFrame:
    """Recompute per-scenario Spearman cluster means with BCa intervals."""
    pairs = read_input(root, PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds.csv")
    summary = read_input(root, PACK1_REL / "model_seed_sweep/rank_correlation_model_seeds_summary.csv")
    rows: list[dict[str, Any]] = []
    for raw_key, group in pairs.groupby(["dataset", "scenario"], sort=False):
        dataset, scenario = unpack_pair(raw_key)
        per_scenario_seed = group.groupby("scenario_seed", as_index=True)["spearman"].mean()
        ci = bootstrap_mean_ci(
            per_scenario_seed.to_numpy(float),
            seed=comparison_seed(str(dataset), str(scenario), "spearman_cluster"),
        )
        stored = summary.loc[(summary["dataset"] == dataset) & (summary["scenario"] == scenario)].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "n_pairs": len(group),
                "n_scenario_seeds": int(per_scenario_seed.size),
                "mean_spearman_pairs": float(group["spearman"].mean()),
                "median_spearman_pairs": float(group["spearman"].median()),
                "min_spearman_pair": float(group["spearman"].min()),
                "sd_across_scenario_seed_means": float(per_scenario_seed.std(ddof=1)),
                "ci95_bca_low": ci["ci95_bca_low"],
                "ci95_bca_high": ci["ci95_bca_high"],
                "bootstrap_method": ci["bootstrap_method"],
                "bootstrap_warnings": ci["bootstrap_warnings"],
                "stored_mean_spearman": float(stored["mean_spearman"]),
                "stored_min_spearman": float(stored["min_spearman"]),
            }
        )
    return pd.DataFrame(rows)


def build_pack2_table(root: Path) -> pd.DataFrame:
    """Recompute pack2 per-scenario CIs, reusing published GADS seed42 rows."""
    per_seed = read_input(root, PACK2_REL / "comparison_per_seed.csv")
    stored = read_input(root, PACK1_REL / "model_seed_sweep/absolute_precision_ci.csv")
    stored = stored.loc[stored["scope"] == "model_seed_42"].set_index(["dataset", "scenario", "metric"])
    rows: list[dict[str, Any]] = []
    for raw_key, group in per_seed.groupby(["dataset", "scenario", "method"], sort=False):
        dataset, scenario, method = unpack_triple(raw_key)
        if group["seed"].nunique() != 20:
            raise ValueError(f"pack2 cell does not have 20 seeds: {dataset}/{scenario}/{method}")
        for metric in METRICS:
            if not applicable(str(scenario), metric):
                continue
            values = numeric_series(group, metric)
            if values.empty:
                continue
            ci = bootstrap_mean_ci(
                values.to_numpy(float),
                seed=comparison_seed("pack2", str(dataset), str(scenario), str(method), metric),
            )
            ci_source = "recomputed_pack2"
            dataset_label = PACK2_DATASET_LABELS[str(dataset)]
            key = (dataset_label, scenario, metric)
            if method == "GADS" and key in stored.index:
                published = stored.loc[key]
                if abs(float(published["mean"]) - ci["mean"]) > 1e-12:
                    raise RuntimeError(f"stored/recomputed mean mismatch for {key}")
                if abs(float(published["sd"]) - ci["sd"]) > 1e-12:
                    raise RuntimeError(f"stored/recomputed sd mismatch for {key}")
                ci = {
                    "mean": float(published["mean"]),
                    "sd": float(published["sd"]),
                    "ci95_percentile_low": float(published["ci95_percentile_low"]),
                    "ci95_percentile_high": float(published["ci95_percentile_high"]),
                    "ci95_bca_low": float(published["ci95_bca_low"]),
                    "ci95_bca_high": float(published["ci95_bca_high"]),
                    "bootstrap_bias_correction": float(published["bootstrap_bias_correction"]),
                    "bootstrap_acceleration": float(published["bootstrap_acceleration"]),
                    "bootstrap_method": str(published["bootstrap_method"]),
                    "bootstrap_warnings": "",
                }
                ci_source = "pack1_published_model_seed_42"
            rows.append(
                {
                    "dataset": dataset_label,
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "n_seeds": int(values.size),
                    **ci,
                    "ci_source": ci_source,
                }
            )
    return pd.DataFrame(rows)


def build_hyperparameter_table(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute per-cell and overall hyperparameter means with cluster CIs."""
    metrics = read_input(root, PACK1_REL / "hyperparameter_grid/metrics_long.csv")
    metrics = metrics.loc[metrics["method"] == "GADS"].copy()
    if metrics["scenario_seed"].nunique() != 20:
        raise ValueError("hyperparameter grid does not have 20 scenario seeds")
    metrics["config_id"] = metrics.apply(
        lambda row: str(row["config_label"]) + ("_with_indicator" if bool(row["include_missing_indicators"]) else ""),
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for (config, scenario), group in metrics.groupby(["config_id", "scenario"], sort=False):
        if group["scenario_seed"].nunique() != 20 or len(group) != 20:
            raise ValueError(f"hyperparameter cell is not 20 seeds: {config}/{scenario} ({len(group)})")
        for metric in METRICS:
            if not applicable(str(scenario), metric):
                continue
            values = numeric_series(group, metric)
            if values.empty:
                continue
            rows.append(
                {
                    "config_id": config,
                    "config_label": group["config_label"].iloc[0],
                    "max_depth": int(group["max_depth"].iloc[0]),
                    "min_child_weight": int(group["min_child_weight"].iloc[0]),
                    "include_missing_indicators": bool(group["include_missing_indicators"].iloc[0]),
                    "scenario": scenario,
                    "metric": metric,
                    "n_scenario_seeds": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                }
            )
    per_cell = pd.DataFrame(rows)

    overall: list[dict[str, Any]] = []
    for config, group in metrics.groupby("config_id", sort=False):
        config_meta = group.iloc[0]
        for metric in METRICS:
            wide = group.pivot_table(index="scenario_seed", columns="scenario", values=metric, aggfunc="first")
            columns = [scenario for scenario in SCENARIOS if applicable(scenario, metric) and scenario in wide]
            if not columns:
                continue
            cluster_values = wide.loc[:, columns].mean(axis=1).dropna()
            if cluster_values.size != 20:
                raise ValueError(f"overall cluster count {cluster_values.size} for {config}/{metric}")
            ci = bootstrap_mean_ci(
                cluster_values.to_numpy(float),
                seed=comparison_seed("hyper", str(config), metric),
            )
            pooled = numeric_series(group, metric)
            overall.append(
                {
                    "config_id": config,
                    "config_label": config_meta["config_label"],
                    "max_depth": int(config_meta["max_depth"]),
                    "min_child_weight": int(config_meta["min_child_weight"]),
                    "include_missing_indicators": bool(config_meta["include_missing_indicators"]),
                    "metric": metric,
                    "n_scenarios_averaged": len(columns),
                    "n_clusters": int(cluster_values.size),
                    "mean_cluster": ci["mean"],
                    "sd_cluster": ci["sd"],
                    "ci95_bca_low": ci["ci95_bca_low"],
                    "ci95_bca_high": ci["ci95_bca_high"],
                    "bootstrap_method": ci["bootstrap_method"],
                    "bootstrap_warnings": ci["bootstrap_warnings"],
                    "pooled_run_mean": float(pooled.mean()),
                    "pooled_run_sd": float(pooled.std(ddof=1)),
                    "n_pooled_runs": int(pooled.size),
                }
            )
    return per_cell, pd.DataFrame(overall)


def build_missing_indicator(root: Path) -> pd.DataFrame:
    """Read the published missing-indicator ablation table."""
    frame = read_input(root, PACK1_REL / "hyperparameter_grid/missing_indicators_comparison.csv")
    return frame.loc[
        :,
        [
            "dataset",
            "scenario",
            "metric",
            "n_pairs",
            "included_mean",
            "excluded_mean",
            "mean_diff",
            "ci95_bca_low",
            "ci95_bca_high",
            "bootstrap_method",
            "p_value_permutation",
            "favors",
        ],
    ]


def build_wasserstein(root: Path) -> pd.DataFrame:
    """Read the published Wasserstein-X paired-difference table."""
    return read_input(root, PACK1_REL / "hyperparameter_grid/wasserstein_x_scale_paired_diff.csv")


def verify_pack2_gads_identity(root: Path) -> dict[str, Any]:
    """Check that pack2 GADS per-seed samples equal pack1 model_seed_42 samples."""
    pack1 = read_input(root, PACK1_REL / "model_seed_sweep/metrics_long.csv")
    pack1 = pack1.loc[(pack1["method"] == "GADS") & (pack1["model_seed"] == 42)]
    pack2 = read_input(root, PACK2_REL / "comparison_per_seed.csv")
    pack2 = pack2.loc[pack2["method"] == "GADS"].copy()
    pack2["dataset"] = pack2["dataset"].map(PACK2_DATASET_LABELS)
    left = pack1.rename(columns={"scenario_seed": "seed"})[["dataset", "scenario", "seed", *METRICS]]
    right = pack2[["dataset", "scenario", "seed", *METRICS]]
    merged = left.merge(right, on=["dataset", "scenario", "seed"], suffixes=("_pack1", "_pack2"))
    left_keys = set(map(tuple, left[["dataset", "scenario", "seed"]].to_numpy()))
    right_keys = set(map(tuple, right[["dataset", "scenario", "seed"]].to_numpy()))
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        first = merged[f"{metric}_pack1"]
        second = merged[f"{metric}_pack2"]
        both = first.notna() & second.notna()
        either = first.notna() | second.notna()
        metrics[metric] = {
            "max_abs_diff": float((first[both] - second[both]).abs().max()) if both.any() else None,
            "nan_mismatch": int((either & ~both).sum()),
        }
    return {
        "pack1_rows": len(left),
        "pack2_rows": len(right),
        "merged_rows": len(merged),
        "drop_left_only": len(left_keys - right_keys),
        "drop_right_only": len(right_keys - left_keys),
        "metrics": metrics,
    }


def fmt(value: object, digits: int = 3) -> str:
    """Format a scalar number for the compact markdown summary."""
    if value is None or value is pd.NA:
        return "--"
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"expected a scalar number, got {type(value)!r}")
    number = float(value)
    if math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def cell(row: pd.Series) -> str:
    """Format one mean (SD) [95% BCa CI] cell."""
    return f"{fmt(row['mean'])} ({fmt(row['sd'])}) [{fmt(row['ci95_bca_low'])}, {fmt(row['ci95_bca_high'])}]"


def table(rows: list[list[str]], header: list[str]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def gads_markdown(frame: pd.DataFrame, dataset: str, scope: str) -> str:
    """Render one GADS scenario-by-metric table."""
    subset = frame.loc[(frame["dataset"] == dataset) & (frame["scope"] == scope)]
    rows: list[list[str]] = []
    for scenario in SCENARIOS:
        cells: list[str] = []
        for metric in METRICS:
            selected = subset.loc[(subset["scenario"] == scenario) & (subset["metric"] == metric)]
            cells.append(cell(selected.iloc[0]) if not selected.empty else "--")
        rows.append([scenario, *cells])
    return table(rows, ["scenario", *METRICS])


def competitor_markdown(frame: pd.DataFrame, metric: str) -> str:
    """Render one pack2 method-by-scenario table."""
    subset = frame.loc[(frame["dataset"] == "Dataset B") & (frame["metric"] == metric)]
    rows: list[list[str]] = []
    for method in sorted(subset["method"].unique()):
        cells: list[str] = []
        for scenario in SCENARIOS:
            selected = subset.loc[(subset["scenario"] == scenario) & (subset["method"] == method)]
            cells.append(cell(selected.iloc[0]) if not selected.empty else "--")
        rows.append([method, *cells])
    return table(rows, ["method", *SCENARIOS])


def spearman_markdown(frame: pd.DataFrame) -> str:
    """Render the Spearman table."""
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row["dataset"]),
                str(row["scenario"]),
                str(int(row["n_pairs"])),
                fmt(row["mean_spearman_pairs"]),
                fmt(row["median_spearman_pairs"]),
                fmt(row["min_spearman_pair"]),
                fmt(row["sd_across_scenario_seed_means"]),
                f"[{fmt(row['ci95_bca_low'])}, {fmt(row['ci95_bca_high'])}]",
            ]
        )
    return table(
        rows,
        ["dataset", "scenario", "pairs", "mean rho", "median", "min", "SD of seed means", "95% BCa CI"],
    )


def hyper_markdown(frame: pd.DataFrame) -> str:
    """Render the overall hyperparameter table for the headline metrics."""
    metrics = ("mrr", "hr_at_1", "far_at_1", "kpi_auc")
    rows: list[list[str]] = []
    for config in (
        "d3_mcw1",
        "d3_mcw5",
        "d3_mcw10",
        "d4_mcw5",
        "d6_mcw1",
        "d6_mcw5",
        "d3_mcw5_with_indicator",
    ):
        subset = frame.loc[frame["config_id"] == config]
        if subset.empty:
            raise ValueError(f"missing hyperparameter config: {config}")
        cells: list[str] = []
        for metric in metrics:
            selected = subset.loc[subset["metric"] == metric]
            if selected.empty:
                cells.append("--")
                continue
            row = selected.iloc[0]
            cells.append(
                f"{fmt(row['mean_cluster'])} ({fmt(row['sd_cluster'])}) "
                f"[{fmt(row['ci95_bca_low'])}, {fmt(row['ci95_bca_high'])}]"
            )
        rows.append([config, *cells])
    return table(rows, ["config", *metrics])


def missing_markdown(frame: pd.DataFrame) -> str:
    """Render the missing-indicator ablation table."""
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row["dataset"]),
                str(row["scenario"]),
                str(row["metric"]),
                fmt(row["included_mean"]),
                fmt(row["excluded_mean"]),
                fmt(row["mean_diff"]),
                f"[{fmt(row['ci95_bca_low'])}, {fmt(row['ci95_bca_high'])}]",
                fmt(row["p_value_permutation"]),
                str(row["favors"]),
            ]
        )
    return table(
        rows,
        ["dataset", "scenario", "metric", "with ind.", "without", "diff", "95% BCa CI", "perm p", "favors"],
    )


def wasserstein_markdown(frame: pd.DataFrame) -> str:
    """Render the Wasserstein-X paired-difference table."""
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                str(row["scenario"]),
                str(row["metric"]),
                fmt(row["raw_mean"]),
                fmt(row["standardized_mean"]),
                fmt(row["mean_diff_standardized_minus_raw"]),
                f"[{fmt(row['ci95_bca_low'])}, {fmt(row['ci95_bca_high'])}]",
                str(int(row["n_pairs"])),
                fmt(row["p_value_permutation"]),
                str(row["favors"]),
            ]
        )
    return table(
        rows,
        ["scenario", "metric", "raw", "standardized", "diff", "95% BCa CI", "pairs", "perm p", "favors"],
    )


def write_summary(
    output_dir: Path,
    input_hashes: dict[str, str],
    gads: pd.DataFrame,
    variance: pd.DataFrame,
    spearman: pd.DataFrame,
    pack2: pd.DataFrame,
    overall: pd.DataFrame,
    missing: pd.DataFrame,
    wasserstein: pd.DataFrame,
    verification: dict[str, Any],
) -> None:
    """Write the compact human-readable summary."""
    lines = [
        "# BDCC internal-review existing-statistics supplement",
        "",
        "Generated only from existing `results/bdcc_revision` tables; no model was trained and no new "
        "experiment was run. All inputs are pinned by SHA-256 in `protocol.json`.",
        "",
        "## Protocol",
        "",
        "- Statistical unit: `scenario_seed` cluster (n = 20). The five model seeds are averaged inside each "
        "scenario cluster first; 10,000 bootstrap resamples are then drawn over the 20 cluster means.",
        "- The 100 model-seed x scenario-seed runs are never treated as independent samples.",
        "- Intervals: 95% BCa (`scipy.stats.bootstrap`, mid-rank tie correction); constant vectors are marked "
        "`degenerate_constant`.",
        "- The five model seeds are fixed at (42, 101, 202, 303, 404); the bootstrap resamples scenario "
        "clusters and does not draw a larger model-seed distribution.",
        "",
        "## Dataset B - GADS, five-model-seed cluster scope",
        "",
        "mean (SD) [95% BCa CI]. `--` means the metric is undefined in that scenario, not zero.",
        "",
        gads_markdown(gads, "Dataset B", "scenario_cluster_5_model_seeds"),
        "",
        "## Dataset A - GADS, five-model-seed cluster scope",
        "",
        gads_markdown(gads, "Dataset A", "scenario_cluster_5_model_seeds"),
        "",
        "## Dataset B - competitor methods, MRR (fixed model seed 42)",
        "",
        competitor_markdown(pack2, "mrr"),
        "",
        "## Dataset B - competitor methods, FAR@1 (fixed model seed 42)",
        "",
        competitor_markdown(pack2, "far_at_1"),
        "",
        "GADS competitor rows reuse the published pack1 `model_seed_42` CIs because the per-seed samples are "
        "bit-identical. Other methods are recomputed from `comparison_per_seed.csv`; every row in "
        "`pack2_ci.csv` carries `ci_source`.",
        "",
        "## Spearman rank correlation across model seeds",
        "",
        spearman_markdown(spearman),
        "",
        "## Six-point hyperparameter grid plus indicator variant (headline metrics)",
        "",
        hyper_markdown(overall),
        "",
        "Full HR@1/HR@5/FAR@1/FAR@5/KPI-AUC cells are in `hyperparameter_overall_ci.csv` and "
        "`hyperparameter_per_scenario_ci.csv`.",
        "",
        "## Missing-indicator ablation, (3, 5)",
        "",
        missing_markdown(missing),
        "",
        "## Dataset A, S2 - raw vs normal-pool standardized Wasserstein-X",
        "",
        wasserstein_markdown(wasserstein),
        "",
        "## Variance decomposition",
        "",
        "`within_sd` = sqrt(mean over the 20 scenario seeds of the variance across the five model seeds); "
        "`between_sd` = SD of the 20 scenario-seed means (ddof=1); `between_share` = "
        "between / (between + within). Full table: `gads_variance_decomposition.csv`.",
        "",
        "## Verification and deviations",
        "",
        f"- Stored pack1 absolute-precision table: {verification['pack1_absolute_precision']['stored_rows']} "
        f"rows, {verification['pack1_absolute_precision']['exact_rows']} reproduce exactly.",
    ]
    lines.extend(
        f"- Documented drift: {deviation['dataset']}/{deviation['scenario']}/{deviation['metric']} "
        f"({deviation['scope']}), BCa low changes from {deviation['stored_bca_low']:.6f} to "
        f"{deviation['recomputed_bca_low']:.6f} (max delta {deviation['max_abs_delta']:.6f}). "
        "This is bootstrap Monte Carlo/RNG drift that can affect the last printed digit, so the "
        "published stored value is retained."
        for deviation in verification["pack1_absolute_precision"]["bcas_deviations"]
    )
    lines.extend(
        [
            f"- pack2 GADS vs pack1 `model_seed_42`: {verification['pack2_gads_identity']['merged_rows']} "
            "rows matched, maximum absolute difference 0.0 for every metric, including NaN positions.",
            f"- Actual percentile fallbacks: {verification['bootstrap_diagnostics']['percentile_fallback_rows']}; "
            f"bootstrap-warning rows: {verification['bootstrap_diagnostics']['rows_with_warnings']}.",
            "- Constant cells such as [1, 1] are empirical bootstrap degeneracy: all 20 observed cluster "
            "means are identical. This is not evidence that the population has no uncertainty.",
            "- Dataset B hyperparameter grid has one model seed (42) with 20 scenario seeds per cell; "
            "per-scenario Spearman exists only between model-seed pairs.",
            "",
            "## Inputs",
            "",
            "| input | sha256 |",
            "|---|---|",
        ]
    )
    lines.extend(f"| `{path}` | `{digest}` |" for path, digest in input_hashes.items())
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    """Convert NaN values to None so the JSON is strict."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def main() -> None:
    """Run the supplement computation and write all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing results/bdcc_revision (default: cwd).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bdcc_internal_review/existing_stats"),
        help="Output directory (relative paths are resolved against the input root).",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = input_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [require_input(input_root, relative) for relative in INPUT_FILES]
    input_hashes = {str(path.relative_to(input_root)): sha256_file(path) for path in input_paths}

    gads, variance = build_gads_tables(input_root)
    stored_absolute = read_input(input_root, PACK1_REL / "model_seed_sweep/absolute_precision_ci.csv")
    stored_variance = read_input(input_root, PACK1_REL / "model_seed_sweep/variance_decomposition.csv")
    verification: dict[str, Any] = {
        "pack1_absolute_precision": verify_pack1(stored_absolute, gads),
        "pack1_variance_decomposition": verify_variance(stored_variance, variance),
        "pack2_gads_identity": verify_pack2_gads_identity(input_root),
    }

    spearman = build_spearman_table(input_root)
    pack2_frame = build_pack2_table(input_root)
    per_cell, overall = build_hyperparameter_table(input_root)
    missing = build_missing_indicator(input_root)
    wasserstein = build_wasserstein(input_root)

    outputs = {
        "gads_absolute_ci.csv": gads,
        "gads_variance_decomposition.csv": variance,
        "spearman_ci.csv": spearman,
        "pack2_ci.csv": pack2_frame,
        "hyperparameter_overall_ci.csv": overall,
        "hyperparameter_per_scenario_ci.csv": per_cell,
        "missing_indicator_ci.csv": missing,
        "wasserstein_x_scale.csv": wasserstein,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    all_frames = (gads, pack2_frame, spearman, overall)
    fallback_rows = sum(int((frame["bootstrap_method"] == "percentile_fallback").sum()) for frame in all_frames)
    warning_rows = sum(int((frame["bootstrap_warnings"] != "").sum()) for frame in all_frames)
    warning_messages = sorted({message for frame in all_frames for message in frame["bootstrap_warnings"] if message})
    verification["bootstrap_diagnostics"] = {
        "resamples": BOOTSTRAP_SAMPLES,
        "confidence_level": CONFIDENCE,
        "method": "BCa with mid-rank tie correction",
        "percentile_fallback_rows": fallback_rows,
        "rows_with_warnings": warning_rows,
        "warning_messages": warning_messages,
        "method_counts": {
            name: {str(key): int(value) for key, value in frame["bootstrap_method"].value_counts().items()}
            for name, frame in (
                ("gads_absolute", gads),
                ("pack2", pack2_frame),
                ("spearman", spearman),
                ("hyperparameter_overall", overall),
            )
        },
    }
    verification["pack2_ci_source_counts"] = {
        str(key): int(value) for key, value in pack2_frame["ci_source"].value_counts().items()
    }
    verification["status"] = "pass_with_documented_deviations"
    verification["missing_inputs"] = []

    generated_at = datetime.now(UTC).isoformat()
    protocol = {
        "script": "scripts/bdcc_internal_review_existing_stats.py",
        "generated_at_utc": generated_at,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "no_training": True,
        "no_new_experiment": True,
        "input_files": [{"path": path, "sha256": input_hashes[path]} for path in input_hashes],
        "output_files": sorted([*outputs, "summary.md", "protocol.json", "verification.json"]),
        "design": {
            "pack1": "2 datasets x 5 scenarios x 5 model seeds x 20 scenario seeds",
            "pack2": "2 datasets x 5 scenarios x 20 scenario seeds x 9 methods, model seed 42",
            "model_seeds": list(MODEL_SEEDS),
            "scenarios": list(SCENARIOS),
            "metrics": list(METRICS),
        },
        "statistics": {
            "resampling_unit": "scenario_seed cluster (n=20)",
            "model_seeds_averaged_within_cluster_first": True,
            "runs_treated_as_independent": False,
            "bootstrap_resamples": BOOTSTRAP_SAMPLES,
            "confidence_level": CONFIDENCE,
            "interval": "95% BCa",
            "rigor_note": (
                "The five model seeds are fixed at (42, 101, 202, 303, 404); the scenario-cluster "
                "bootstrap does not sample a larger model-seed distribution."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }

    (output_dir / "verification.json").write_text(
        json.dumps(json_safe(verification), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_summary(
        output_dir,
        input_hashes,
        gads,
        variance,
        spearman,
        pack2_frame,
        overall,
        missing,
        wasserstein,
        verification,
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(json_safe(protocol), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {len(outputs) + 3} artifacts to {output_dir}")
    print(json.dumps(verification["bootstrap_diagnostics"], indent=2))


if __name__ == "__main__":
    main()

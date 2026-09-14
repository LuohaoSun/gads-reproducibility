"""Paired uncertainty estimates and significance tests for experiment runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

S4_NULL_SCENARIO = "S4-null"
FAR_SCENARIOS: frozenset[str] = frozenset({"S4-main", S4_NULL_SCENARIO})
DEFAULT_SUMMARY_METRICS: tuple[str, ...] = (
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_5",
)
REQUIRED_METRICS_CONTEXT = ("dataset", "scenario", "seed", "method", "status")
PairedTestName = Literal["permutation", "wilcoxon"]


def is_root_cause_metric(metric: str) -> bool:
    """Return whether a metric requires annotated root causes."""
    return metric == "mrr" or metric.startswith("hr_at_") or metric.startswith("root_cause_coverage_at_")


def is_far_metric(metric: str) -> bool:
    """Return whether a metric measures false-alarm rate on harmless features."""
    return metric.startswith("far_at_")


def metric_applicable(scenario: str, metric: str) -> tuple[bool, str]:
    """Return whether a scenario/metric pair is defined and its exclusion note."""
    if scenario == S4_NULL_SCENARIO and is_root_cause_metric(metric):
        return False, "s4_null_no_root_cause"
    if is_far_metric(metric) and scenario not in FAR_SCENARIOS:
        return False, "no_harmless_drift"
    return True, ""


def filter_valid_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose experiment status is ``valid``."""
    if "status" not in frame.columns:
        raise ValueError("frame must contain a status column")
    valid = frame.loc[frame["status"] == "valid"].copy()
    if valid.empty:
        raise ValueError("no rows with status=valid")
    return valid


def load_experiment_metrics(path: str | Path) -> pd.DataFrame:
    """Load a metrics table and retain only valid runs."""
    metrics_path = Path(path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    frame = pd.read_parquet(metrics_path)
    missing = set(REQUIRED_METRICS_CONTEXT) - set(frame.columns)
    if missing:
        raise ValueError(f"metrics missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("metrics table is empty")
    return filter_valid_runs(frame)


def valid_run_keys(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return unique dataset/scenario/seed keys for valid metric rows."""
    frame = metrics.loc[metrics["status"] == "valid"] if "status" in metrics.columns else metrics
    return frame.loc[:, ["dataset", "scenario", "seed"]].drop_duplicates()


def load_experiment_rankings(path: str | Path, *, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load a rankings table, optionally restricting to valid metric runs."""
    rankings_path = Path(path)
    if not rankings_path.is_file():
        raise FileNotFoundError(f"rankings file not found: {rankings_path}")
    frame = pd.read_parquet(rankings_path)
    required = {"dataset", "scenario", "seed", "method", "equipment", "feature", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"rankings missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("rankings table is empty")
    if "status" in frame.columns:
        return filter_valid_runs(frame)
    if metrics is not None:
        keys = valid_run_keys(metrics)
        filtered = frame.merge(keys, on=["dataset", "scenario", "seed"], how="inner")
        if filtered.empty:
            raise ValueError("no ranking rows match valid metrics runs")
        return filtered
    return frame


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be non-empty and finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, array.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(array.mean()), float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def paired_permutation_test(
    left: Iterable[float],
    right: Iterable[float],
    *,
    n_resamples: int = 20_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sided paired randomization test of a mean difference."""
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("paired samples must have equal length >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired samples must be finite")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    observed = float(np.mean(x - y))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, x.size))
    null = np.mean(signs * (x - y), axis=1)
    p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_resamples + 1))
    return observed, p_value


def paired_wilcoxon_test(left: Iterable[float], right: Iterable[float]) -> tuple[float, float]:
    """Return Wilcoxon signed-rank statistic and two-sided p-value."""
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("paired samples must have equal length >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired samples must be finite")
    result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    return float(cast(Any, result).statistic), float(cast(Any, result).pvalue)


def friedman_test(values_by_method: dict[str, Iterable[float]]) -> tuple[float, float]:
    """Run a Friedman test across methods measured on the same seeds."""
    if len(values_by_method) < 3:
        raise ValueError("Friedman test requires at least three methods")
    arrays = [np.asarray(list(values), dtype=float) for values in values_by_method.values()]
    if len({array.size for array in arrays}) != 1 or arrays[0].size < 2:
        raise ValueError("methods must have equal-length paired samples >= 2")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("method samples must be finite")
    result = friedmanchisquare(*arrays)
    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down correction for a sequence of p-values."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be finite values in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float((p.size - rank) * p[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def _summarize_values(
    values: np.ndarray,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[int, float, float, float, float]:
    if values.size == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size == 1:
        return 1, mean, std, mean, mean
    mean, low, high = bootstrap_mean_ci(values, confidence=confidence, n_resamples=n_resamples, seed=seed)
    return values.size, mean, std, low, high


def summarize_metric_runs(
    metrics: pd.DataFrame,
    *,
    metric: str,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize a long-form metric table by method with bootstrap intervals."""
    if "method" not in metrics or metric not in metrics:
        raise ValueError("metrics must contain method and requested metric columns")
    rows: list[dict[str, object]] = []
    for method, subset in metrics.groupby("method", sort=False):
        values = subset[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            rows.append(
                {"method": method, "metric": metric, "n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            )
            continue
        mean, low, high = bootstrap_mean_ci(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
        rows.append(
            {"method": method, "metric": metric, "n": values.size, "mean": mean, "ci_low": low, "ci_high": high}
        )
    return pd.DataFrame(rows)


def summarize_grouped_metrics(
    metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize metrics by dataset×scenario×method×metric."""
    missing = set(metric_columns) - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics missing requested metric columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby(["dataset", "scenario", "method"], sort=False):
        dataset, scenario, method = (str(part) for part in cast(tuple[object, ...], key))
        for metric in metric_columns:
            applicable, note = metric_applicable(scenario, metric)
            values = group[metric].dropna().to_numpy(dtype=float) if applicable else np.array([], dtype=float)
            n, mean, std, ci_low, ci_high = _summarize_values(
                values,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "applicable": applicable,
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def _run_paired_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    test: PairedTestName,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    mean_diff = float(np.mean(left - right))
    if test == "permutation":
        observed, p_value = paired_permutation_test(left, right, n_resamples=n_resamples, seed=seed)
        return observed, mean_diff, p_value
    statistic, p_value = paired_wilcoxon_test(left, right)
    return statistic, mean_diff, p_value


def compare_gads_vs_baselines(
    metrics: pd.DataFrame,
    *,
    reference_method: str = "GADS",
    metric_columns: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    test: PairedTestName = "permutation",
    holm: bool = True,
    n_resamples: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare GADS against each baseline with seed-paired tests."""
    if reference_method not in set(metrics["method"]):
        raise ValueError(f"reference method {reference_method!r} is absent from metrics")
    missing = set(metric_columns) - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics missing requested metric columns: {sorted(missing)}")
    if test not in {"permutation", "wilcoxon"}:
        raise ValueError("test must be 'permutation' or 'wilcoxon'")

    rows: list[dict[str, object]] = []
    for key, scenario_frame in metrics.groupby(["dataset", "scenario"], sort=False):
        dataset, scenario = (str(part) for part in cast(tuple[object, ...], key))
        baselines = sorted(method for method in scenario_frame["method"].unique() if method != reference_method)
        for metric in metric_columns:
            applicable, exclusion_note = metric_applicable(scenario, metric)
            group_rows: list[dict[str, object]] = []
            for baseline in baselines:
                if not applicable:
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "metric": metric,
                            "reference_method": reference_method,
                            "baseline": baseline,
                            "n_pairs": 0,
                            "mean_diff": np.nan,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "p_value_holm": np.nan,
                            "test": test,
                            "applicable": False,
                            "note": exclusion_note,
                        }
                    )
                    continue
                paired = scenario_frame.loc[
                    scenario_frame["method"].isin((reference_method, baseline)),
                    ["seed", "method", metric],
                ]
                pivot = paired.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
                if reference_method not in pivot.columns or baseline not in pivot.columns:
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "metric": metric,
                            "reference_method": reference_method,
                            "baseline": baseline,
                            "n_pairs": 0,
                            "mean_diff": np.nan,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "p_value_holm": np.nan,
                            "test": test,
                            "applicable": applicable,
                            "note": "missing_method",
                        }
                    )
                    continue
                aligned = pivot[[reference_method, baseline]].dropna()
                left = aligned[reference_method].to_numpy(dtype=float)
                right = aligned[baseline].to_numpy(dtype=float)
                finite_mask = np.isfinite(left) & np.isfinite(right)
                n_aligned = left.size
                left = left[finite_mask]
                right = right[finite_mask]
                if left.size < 2:
                    note = "non_finite_pairs" if n_aligned >= 2 and left.size < n_aligned else "insufficient_pairs"
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "metric": metric,
                            "reference_method": reference_method,
                            "baseline": baseline,
                            "n_pairs": int(left.size),
                            "mean_diff": np.nan,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "p_value_holm": np.nan,
                            "test": test,
                            "applicable": False,
                            "note": note,
                        }
                    )
                    continue
                statistic, mean_diff, p_value = _run_paired_test(
                    left,
                    right,
                    test=test,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                group_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "metric": metric,
                        "reference_method": reference_method,
                        "baseline": baseline,
                        "n_pairs": int(left.size),
                        "mean_diff": mean_diff,
                        "statistic": statistic,
                        "p_value": p_value,
                        "p_value_holm": p_value,
                        "test": test,
                        "applicable": True,
                        "note": "",
                    }
                )
            if holm and group_rows:
                testable = [row for row in group_rows if row["applicable"] and np.isfinite(cast(float, row["p_value"]))]
                if len(testable) >= 2:
                    adjusted = holm_adjust([cast(float, row["p_value"]) for row in testable])
                    for row, p_adj in zip(testable, adjusted, strict=True):
                        row["p_value_holm"] = float(p_adj)
            rows.extend(group_rows)
    return pd.DataFrame(rows)


def summarize_experiment_results(
    metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    reference_method: str = "GADS",
    test: PairedTestName = "permutation",
    holm: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 20_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary and paired-comparison tables from valid metrics rows."""
    summary = summarize_grouped_metrics(
        metrics,
        metric_columns=metric_columns,
        confidence=confidence,
        n_resamples=bootstrap_samples,
        seed=seed,
    )
    comparisons = compare_gads_vs_baselines(
        metrics,
        reference_method=reference_method,
        metric_columns=metric_columns,
        test=test,
        holm=holm,
        n_resamples=permutation_samples,
        seed=seed,
    )
    return summary, comparisons


def format_summary_latex(summary: pd.DataFrame) -> str:
    """Render grouped summary statistics as a LaTeX table."""
    columns = ["dataset", "scenario", "method", "metric", "n", "mean", "std", "ci_low", "ci_high"]
    missing = set(columns) - set(summary.columns)
    if missing:
        raise ValueError(f"summary missing columns for LaTeX export: {sorted(missing)}")
    return summary.loc[:, columns].to_latex(index=False, float_format="%.4f", na_rep="--")


def format_comparisons_latex(comparisons: pd.DataFrame) -> str:
    """Render paired comparison statistics as a LaTeX table."""
    columns = [
        "dataset",
        "scenario",
        "metric",
        "baseline",
        "n_pairs",
        "mean_diff",
        "p_value",
        "p_value_holm",
        "test",
        "note",
    ]
    missing = set(columns) - set(comparisons.columns)
    if missing:
        raise ValueError(f"comparisons missing columns for LaTeX export: {sorted(missing)}")
    return comparisons.loc[:, columns].to_latex(index=False, float_format="%.4f", na_rep="--")


def write_stats_outputs(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist summary and comparison tables to CSV, Parquet and LaTeX."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in (("summary", summary), ("comparisons", comparisons)):
        csv_path = directory / f"stats_{name}.csv"
        parquet_path = directory / f"stats_{name}.parquet"
        frame.to_csv(csv_path, index=False)
        frame.to_parquet(parquet_path, index=False)
        paths[f"{name}_csv"] = csv_path
        paths[f"{name}_parquet"] = parquet_path
    summary_tex = directory / "stats_summary.tex"
    comparisons_tex = directory / "stats_comparisons.tex"
    summary_tex.write_text(format_summary_latex(summary), encoding="utf-8")
    comparisons_tex.write_text(format_comparisons_latex(comparisons), encoding="utf-8")
    paths["summary_latex"] = summary_tex
    paths["comparisons_latex"] = comparisons_tex
    combined_csv = directory / "stats.csv"
    combined_parquet = directory / "stats.parquet"
    combined = pd.concat(
        [summary.assign(table="summary"), comparisons.assign(table="comparisons")],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(combined_csv, index=False)
    combined.to_parquet(combined_parquet, index=False)
    paths["combined_csv"] = combined_csv
    paths["combined_parquet"] = combined_parquet
    return paths

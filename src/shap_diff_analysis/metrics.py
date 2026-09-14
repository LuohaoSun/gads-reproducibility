"""Ranking metrics for equipment-level root-cause evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import numpy as np
import pandas as pd


def _validate_rankings(rankings: pd.DataFrame) -> None:
    required = {"method", "equipment", "feature", "rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings missing required columns: {sorted(missing)}")
    if rankings.empty:
        raise ValueError("rankings must not be empty")
    if rankings[["method", "equipment", "feature"]].isna().to_numpy().any():
        raise ValueError("ranking identifiers must not be missing")
    ranks = cast(pd.Series, pd.to_numeric(rankings["rank"], errors="coerce"))
    rank_values = ranks.to_numpy(dtype=float)
    if np.isnan(rank_values).any() or (rank_values < 1).any() or np.any(rank_values % 1 != 0):
        raise ValueError("rank must contain positive integers")


def evaluate_rankings_per_equipment(
    rankings: pd.DataFrame,
    *,
    root_causes: Mapping[str, Iterable[str]],
    harmless_features: Iterable[str] = (),
    top_ks: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Compute MRR, HR@K, coverage@K and false-alarm@K per device.

    If a device has no annotated root cause (the S4-null calibration scenario),
    all root-cause metrics are ``NaN`` rather than incorrectly reporting zeros.
    False-alarm metrics remain defined whenever ``harmless_features`` is given.
    """
    _validate_rankings(rankings)
    ks = tuple(dict.fromkeys(int(k) for k in top_ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("top_ks must contain positive integers")
    harmless = set(harmless_features)
    root_map = {str(device): set(features) for device, features in root_causes.items()}
    rows: list[dict[str, object]] = []
    for group_key, subset in rankings.groupby(["method", "equipment"], sort=False):
        method, equipment = cast(tuple[str, str], group_key)
        ordered = subset.sort_values(["rank", "feature"], kind="stable")["feature"].tolist()
        roots = root_map.get(str(equipment), set())
        if roots:
            hit_ranks = [idx + 1 for idx, feature in enumerate(ordered) if feature in roots]
            reciprocal_rank = 1.0 / min(hit_ranks) if hit_ranks else 0.0
        else:
            reciprocal_rank = np.nan
        row: dict[str, object] = {
            "method": method,
            "equipment": equipment,
            "mrr": reciprocal_rank,
            "n_root_causes": len(roots),
            "n_ranked_features": len(ordered),
        }
        for k in ks:
            top = set(ordered[:k])
            row[f"hr_at_{k}"] = float(bool(top & roots)) if roots else np.nan
            row[f"root_cause_coverage_at_{k}"] = float(len(top & roots) / len(roots)) if roots else np.nan
            row[f"far_at_{k}"] = float(bool(top & harmless)) if harmless else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rankings(
    rankings: pd.DataFrame,
    *,
    root_causes: Mapping[str, Iterable[str]],
    harmless_features: Iterable[str] = (),
    top_ks: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Aggregate equipment-level ranking metrics by method."""
    per_equipment = evaluate_rankings_per_equipment(
        rankings,
        root_causes=root_causes,
        harmless_features=harmless_features,
        top_ks=top_ks,
    )
    aggregations: dict[str, str] = {"mrr": "mean", "n_root_causes": "sum", "n_ranked_features": "mean"}
    metric_columns = [
        column
        for column in per_equipment.columns
        if column.startswith("hr_at_") or column.startswith("root_cause_coverage_at_") or column.startswith("far_at_")
    ]
    aggregations.update(dict.fromkeys(metric_columns, "mean"))
    result = cast(pd.DataFrame, per_equipment.groupby("method", as_index=False).agg(aggregations))
    result["n_equipment"] = per_equipment.groupby("method")["equipment"].nunique().to_numpy()
    return cast(pd.DataFrame, result)


def add_result_context(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    """Attach long-run identifiers without mutating the source frame."""
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    result = metrics.copy()
    result.insert(0, "dataset", dataset)
    result.insert(1, "scenario", scenario)
    result.insert(2, "seed", int(seed))
    return result

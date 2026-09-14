"""Grouped attribution-distribution differences used by GADS."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

GadsDistance = Literal["wasserstein", "mean", "kl", "js"]
GADS_DISTANCES: tuple[GadsDistance, ...] = ("wasserstein", "mean", "kl", "js")
DEFAULT_GADS_BINS = 50
DEFAULT_GADS_EPSILON = 1e-10


def _validate_groups(
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series")
    if groups.isna().any():
        raise ValueError("groups must not contain missing values")
    abnormal = tuple(dict.fromkeys(str(value) for value in abnormal_devices))
    normal = tuple(dict.fromkeys(str(value) for value in normal_devices))
    if not abnormal:
        raise ValueError("abnormal_devices must not be empty")
    if not normal:
        raise ValueError("normal_devices must not be empty")
    if set(abnormal) & set(normal):
        raise ValueError("abnormal_devices and normal_devices must be disjoint")
    present = set(groups.astype(str).unique())
    missing = (set(abnormal) | set(normal)) - present
    if missing:
        raise ValueError(f"devices are absent from groups: {sorted(missing)}")
    return abnormal, normal


def _validate_distance(distance: str) -> GadsDistance:
    if distance not in GADS_DISTANCES:
        raise ValueError(f"unsupported GADS distance {distance!r}; expected one of: {GADS_DISTANCES}")
    return distance


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def _reference_binned_probabilities(
    reference: np.ndarray,
    sample: np.ndarray,
    *,
    bins: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("bins must be at least 2 for KL/JS distances")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite value")
    reference_values = np.asarray(reference, dtype=float)
    sample_values = np.asarray(sample, dtype=float)
    if reference_values.size == 0 or sample_values.size == 0:
        raise ValueError("reference and sample arrays must be non-empty")
    if not np.isfinite(reference_values).all() or not np.isfinite(sample_values).all():
        raise ValueError("reference and sample arrays must contain finite values")
    edges = np.histogram_bin_edges(reference_values, bins=bins)
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    sample_counts, _ = np.histogram(sample_values, bins=edges)
    reference_probs = (reference_counts.astype(float) + epsilon) / (
        reference_counts.sum() + epsilon * reference_counts.size
    )
    sample_probs = (sample_counts.astype(float) + epsilon) / (sample_counts.sum() + epsilon * sample_counts.size)
    return reference_probs, sample_probs


def compute_gads_distance(
    abnormal_values: np.ndarray,
    normal_values: np.ndarray,
    *,
    distance: str = "wasserstein",
    bins: int = DEFAULT_GADS_BINS,
    epsilon: float = DEFAULT_GADS_EPSILON,
) -> float:
    """Compare abnormal and normal SHAP samples with a named distance."""
    selected = _validate_distance(distance)
    abnormal = np.asarray(abnormal_values, dtype=float)
    normal = np.asarray(normal_values, dtype=float)
    if abnormal.size == 0 or normal.size == 0:
        raise ValueError("abnormal and normal arrays must be non-empty")
    if not np.isfinite(abnormal).all() or not np.isfinite(normal).all():
        raise ValueError("abnormal and normal arrays must contain finite values")

    if selected == "wasserstein":
        return float(wasserstein_distance(abnormal, normal))
    if selected == "mean":
        return float(abs(np.mean(abnormal) - np.mean(normal)))
    reference_probs, abnormal_probs = _reference_binned_probabilities(
        normal,
        abnormal,
        bins=bins,
        epsilon=epsilon,
    )
    if selected == "kl":
        return _kl_divergence(abnormal_probs, reference_probs)
    mixture = 0.5 * (abnormal_probs + reference_probs)
    return 0.5 * _kl_divergence(abnormal_probs, mixture) + 0.5 * _kl_divergence(reference_probs, mixture)


def rank_gads_features(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    distance: str = "wasserstein",
    bins: int = DEFAULT_GADS_BINS,
    epsilon: float = DEFAULT_GADS_EPSILON,
    method_name: str = "GADS",
) -> pd.DataFrame:
    """Rank features by a configurable attribution-distribution distance.

    One global KPI model supplies ``shap_values``.  For every abnormal device,
    the reference distribution is the union of *only* ``normal_devices``;
    abnormal devices are never used as references, including in multi-fault
    scenarios such as S3.

    ``view='process'`` excludes the equipment contextual covariate, while
    ``view='all'`` includes every column and ``view='residual'`` returns only
    the equipment column (if present).
    """
    if not isinstance(shap_values, pd.DataFrame):
        raise TypeError("shap_values must be a pandas DataFrame")
    if shap_values.empty:
        raise ValueError("shap_values must contain at least one row and column")
    if len(shap_values) != len(groups) or not shap_values.index.equals(groups.index):
        raise ValueError("shap_values and groups must have identical indexes")
    if not np.isfinite(shap_values.to_numpy(dtype=float)).all():
        raise ValueError("shap_values must contain finite values")
    selected_distance = _validate_distance(distance)
    abnormal, normal = _validate_groups(groups, abnormal_devices, normal_devices)
    if view not in {"process", "all", "residual"}:
        raise ValueError("view must be one of 'process', 'all', or 'residual'")

    feature_names = list(shap_values.columns)
    equipment_names = [name for name in feature_names if name.lower() in {"eqp", "equipment", "device"}]
    if view == "process":
        if process_features is None:
            selected = [name for name in feature_names if name not in equipment_names]
        else:
            selected = list(process_features)
            missing = set(selected) - set(feature_names)
            if missing:
                raise ValueError(f"process features are absent from SHAP values: {sorted(missing)}")
            selected = [name for name in selected if name not in equipment_names]
        if not selected:
            raise ValueError("process view has no features after excluding equipment context")
    elif view == "residual":
        if not equipment_names:
            raise ValueError("residual view requires an EQP/equipment/device SHAP column")
        selected = equipment_names
    else:
        selected = feature_names

    group_strings = groups.astype(str)
    normal_mask = group_strings.isin(normal)
    rows: list[dict[str, object]] = []
    for device in abnormal:
        abnormal_mask = group_strings == device
        if int(abnormal_mask.sum()) == 0 or int(normal_mask.sum()) == 0:
            raise ValueError(f"device {device!r} has no abnormal or reference samples")
        scores = {
            feature: compute_gads_distance(
                shap_values.loc[abnormal_mask, feature].to_numpy(dtype=float),
                shap_values.loc[normal_mask, feature].to_numpy(dtype=float),
                distance=selected_distance,
                bins=bins,
                epsilon=epsilon,
            )
            for feature in selected
        }
        ordered = sorted(scores, key=lambda name: (-scores[name], name))
        for rank, feature in enumerate(ordered, start=1):
            rows.append(
                {
                    "equipment": device,
                    "feature": feature,
                    "score": scores[feature],
                    "rank": rank,
                    "view": view,
                    "method": method_name,
                    "gads_distance": selected_distance,
                }
            )
    return pd.DataFrame(rows)


def grouped_shap_wasserstein(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Alias with an explicit distance name for downstream callers."""
    return rank_gads_features(
        shap_values,
        groups,
        abnormal_devices,
        normal_devices,
        process_features=process_features,
        distance="wasserstein",
    )

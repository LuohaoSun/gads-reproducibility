"""Protocol-level process-feature filters computed on the normal pool only.

These filters deliberately sit outside the GADS method: they change which
process features the whole experiment is allowed to consider (KPI-model input
and/or every method's ranking surface) using statistics computed on
normal-pool rows only, so the abnormal device under investigation can never
leak into the screening decision.  Two explicit levers are provided:

- a KPI-association pre-filter (point-biserial |corr| with the KPI target on
  normal-pool rows, keeping either Benjamini-Hochberg FDR survivors or the
  top-K strongest associations), applied to the bundle before any model fit;
- an attribution-mass floor (drop features whose normal-pool out-of-fold SHAP
  std is below a percentile threshold), applied between the cross-fitted
  model and the ranking stage.

Both filters fail loudly: zero survivors, ambiguous specifications or invalid
parameters raise instead of silently keeping or dropping features.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .experiment_types import DatasetBundle


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values (step-up, monotone enforced)."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pvalues must be a non-empty 1-D array")
    if np.isnan(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must contain values in [0, 1] without missing entries")
    n = values.size
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(n, dtype=float)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _pool_mask(groups: pd.Series, normal_devices: Iterable[str]) -> np.ndarray:
    normal = tuple(dict.fromkeys(str(device) for device in normal_devices))
    if not normal:
        raise ValueError("normal_devices must not be empty")
    mask = groups.astype(str).isin(normal).to_numpy()
    if not mask.any():
        raise ValueError("no rows belong to the normal-pool devices")
    return mask


@dataclass(frozen=True)
class PoolKpiFilterSpec:
    """Explicit specification for the normal-pool KPI-association pre-filter.

    Exactly one of ``fdr_alpha`` (Benjamini-Hochberg FDR cut-off) and
    ``top_k`` (keep the K strongest |point-biserial| associations) must be
    set; leaving both or neither set is an error.
    """

    fdr_alpha: float | None = None
    top_k: int | None = None

    def __post_init__(self) -> None:
        if (self.fdr_alpha is None) == (self.top_k is None):
            raise ValueError("exactly one of fdr_alpha and top_k must be provided")
        if self.fdr_alpha is not None:
            if not isinstance(self.fdr_alpha, (int, float)) or isinstance(self.fdr_alpha, bool):
                raise TypeError("fdr_alpha must be a real number")
            if not 0.0 < float(self.fdr_alpha) <= 1.0:
                raise ValueError("fdr_alpha must lie in (0, 1]")
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
                raise TypeError("top_k must be an integer")
            if self.top_k < 1:
                raise ValueError("top_k must be a positive integer")

    def mode(self) -> str:
        """Return the selected filter mode ('fdr' or 'top_k')."""
        return "fdr" if self.fdr_alpha is not None else "top_k"


def pool_kpi_association_filter(
    data: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    normal_devices: Sequence[str],
    process_features: Sequence[str],
    spec: PoolKpiFilterSpec,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Rank process features by |point-biserial| association with the KPI on normal-pool rows.

    Returns the surviving features (ordered by descending |correlation|, then
    name) and a JSON-serialisable record with every per-feature statistic.
    Zero-variance pool features cannot associate with anything and are
    assigned correlation 0 with p-value 1 deterministically.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    features = tuple(dict.fromkeys(str(name) for name in process_features))
    if not features:
        raise ValueError("process_features must not be empty")
    required = {target_column, group_column, *features}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")

    groups = data[group_column]
    if not isinstance(groups, pd.Series):
        raise TypeError("group column must form a pandas Series")
    pool = _pool_mask(groups, normal_devices)
    target_series = pd.Series(data.loc[pool, target_column])
    target = cast(pd.Series, pd.to_numeric(target_series, errors="coerce")).to_numpy(dtype=float)
    if target.size < 2 or np.unique(target).size != 2:
        raise ValueError("normal-pool target must be binary with at least two samples")

    statistics: dict[str, dict[str, float]] = {}
    pvalues = np.empty(len(features), dtype=float)
    for index, feature in enumerate(features):
        feature_series = pd.Series(data.loc[pool, feature])
        values = cast(pd.Series, pd.to_numeric(feature_series, errors="coerce")).to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError(f"feature {feature!r} contains non-finite values on the normal pool")
        if float(np.std(values)) == 0.0:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = cast(tuple[float, float], pearsonr(values, target))
            if not (np.isfinite(correlation) and np.isfinite(p_value)):
                raise ValueError(f"point-biserial correlation failed for feature {feature!r}")
        statistics[feature] = {"corr": float(correlation), "p_value": float(p_value)}
        pvalues[index] = float(p_value)

    adjusted = benjamini_hochberg(pvalues)
    for feature, adjusted_p in zip(features, adjusted, strict=True):
        statistics[feature]["adjusted_p"] = float(adjusted_p)

    if spec.fdr_alpha is not None:
        survivors = {
            feature for feature, adjusted_p in zip(features, adjusted, strict=True) if adjusted_p < spec.fdr_alpha
        }
    else:
        order = sorted(features, key=lambda name: (-abs(statistics[name]["corr"]), name))
        survivors = set(order[: spec.top_k])  # type: ignore[index]
    if not survivors:
        raise ValueError(
            f"pool KPI-association filter kept no process features "
            f"(mode={spec.mode()}, fdr_alpha={spec.fdr_alpha}, top_k={spec.top_k})"
        )

    kept = tuple(name for name in sorted(survivors, key=lambda name: (-abs(statistics[name]["corr"]), name)))
    record: dict[str, Any] = {
        "method": "point_biserial_on_normal_pool",
        "mode": spec.mode(),
        "fdr_alpha": spec.fdr_alpha,
        "top_k": spec.top_k,
        "n_pool_rows": int(pool.sum()),
        "n_process_features": len(features),
        "n_kept": len(kept),
        "kept": list(kept),
        "statistics": statistics,
    }
    return kept, record


def apply_pool_kpi_filter_to_bundle(
    bundle: DatasetBundle,
    spec: PoolKpiFilterSpec,
) -> DatasetBundle:
    """Return a new bundle whose process features survive the pool KPI filter.

    Ground-truth annotations (root causes, harmless features) are preserved
    verbatim: a filtered-out root cause or harmless feature simply can no
    longer be ranked, which is the honest protocol outcome.
    """
    if not isinstance(bundle, DatasetBundle):
        raise TypeError("bundle must be a DatasetBundle")
    kept, record = pool_kpi_association_filter(
        bundle.data,
        target_column=bundle.target_column,
        group_column=bundle.group_column,
        normal_devices=bundle.normal_devices,
        process_features=bundle.process_features,
        spec=spec,
    )
    if bundle.group_column in kept:
        raise ValueError("pool KPI filter must not keep the group column")
    metadata = dict(bundle.metadata)
    protocol_filters = dict(metadata.get("protocol_filters", {}))
    protocol_filters["pool_kpi_association"] = record
    metadata["protocol_filters"] = protocol_filters
    return DatasetBundle(
        data=bundle.data,
        scenario=bundle.scenario,
        seed=bundle.seed,
        group_column=bundle.group_column,
        target_column=bundle.target_column,
        process_features=kept,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        metadata=metadata,
    )


def pool_shap_std_floor(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    normal_devices: Iterable[str],
    process_features: Sequence[str],
    floor_percentile: float,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Drop process features whose normal-pool OOF SHAP std is below a percentile.

    The threshold is the ``floor_percentile`` percentile of the normal-pool
    SHAP standard deviations over ``process_features``; features at or above
    the threshold survive.  This removes features the KPI model effectively
    never uses, whose attribution distributions cannot carry signal.
    """
    if not isinstance(shap_values, pd.DataFrame):
        raise TypeError("shap_values must be a pandas DataFrame")
    if isinstance(floor_percentile, bool) or not isinstance(floor_percentile, (int, float)):
        raise TypeError("floor_percentile must be a real number")
    if not np.isfinite(floor_percentile) or not 0.0 <= float(floor_percentile) < 100.0:
        raise ValueError("floor_percentile must lie in [0, 100)")
    features = tuple(dict.fromkeys(str(name) for name in process_features))
    if not features:
        raise ValueError("process_features must not be empty")
    missing = set(features) - set(shap_values.columns)
    if missing:
        raise ValueError(f"process features are absent from SHAP values: {sorted(missing)}")
    if len(shap_values) != len(groups) or not shap_values.index.equals(groups.index):
        raise ValueError("shap_values and groups must have identical indexes")

    pool = _pool_mask(groups, normal_devices)
    pool_shap = shap_values.loc[pool, list(features)]
    if not np.isfinite(pool_shap.to_numpy(dtype=float)).all():
        raise ValueError("normal-pool SHAP values must be finite")
    stds = pool_shap.std(ddof=1)
    if stds.isna().any():
        raise ValueError("normal-pool SHAP standard deviations must not be missing")
    threshold = float(np.percentile(stds.to_numpy(dtype=float), float(floor_percentile)))
    kept = tuple(name for name in features if float(stds[name]) >= threshold)
    if not kept:
        raise ValueError(f"SHAP-std floor kept no process features (floor_percentile={floor_percentile})")
    record: dict[str, Any] = {
        "method": "normal_pool_oof_shap_std_floor",
        "floor_percentile": float(floor_percentile),
        "threshold": threshold,
        "n_pool_rows": int(pool.sum()),
        "n_process_features": len(features),
        "n_kept": len(kept),
        "kept": list(kept),
        "pool_shap_std": {name: float(stds[name]) for name in features},
    }
    return kept, record

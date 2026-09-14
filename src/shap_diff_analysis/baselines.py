"""Raw-space and model-based baselines for the paper experiment matrix."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .gads import _validate_groups
from .modeling import CrossFittedAttributions, ModelType, fit_cross_fitted_model
from .vendor.m2oe import TabularGroupExplainer
from .xpe import xpe_feature_scores


def _selected_features(
    columns: Iterable[str],
    process_features: Iterable[str] | None,
    *,
    view: str,
) -> list[str]:
    names = list(columns)
    equipment = [name for name in names if name.lower() in {"eqp", "equipment", "device"}]
    if view not in {"process", "all", "residual"}:
        raise ValueError("view must be one of 'process', 'all', or 'residual'")
    if view == "process":
        selected = list(process_features) if process_features is not None else [n for n in names if n not in equipment]
        selected = [n for n in selected if n not in equipment]
    elif view == "residual":
        selected = equipment
    else:
        selected = names
    missing = set(selected) - set(names)
    if missing:
        raise ValueError(f"requested features are absent: {sorted(missing)}")
    if not selected:
        raise ValueError("no features selected for ranking")
    return selected


def _validate_inputs(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if len(X) != len(groups) or not X.index.equals(groups.index):
        raise ValueError("X and groups must have identical indexes")
    return _validate_groups(groups, abnormal_devices, normal_devices)


def _ranking_rows(
    scores: dict[str, float],
    *,
    equipment: str,
    method: str,
    view: str,
) -> list[dict[str, object]]:
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    return [
        {
            "equipment": equipment,
            "feature": feature,
            "score": float(scores[feature]),
            "rank": rank,
            "view": view,
            "method": method,
        }
        for rank, feature in enumerate(ordered, start=1)
    ]


def _numeric_values(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = X.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("raw-space baselines require finite numeric process features")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("raw-space baselines require finite numeric process features")
    return numeric


def rank_ks_x(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
) -> pd.DataFrame:
    """Rank raw features by two-sample KS statistic (descending)."""
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    features = _selected_features(X.columns, process_features, view=view)
    values = _numeric_values(X, features)
    groups_str = groups.astype(str)
    normal_mask = groups_str.isin(normal)
    rows: list[dict[str, object]] = []
    for device in abnormal:
        mask = groups_str == device
        scores = {
            feature: float(cast(Any, ks_2samp(values.loc[mask, feature], values.loc[normal_mask, feature])).statistic)
            for feature in features
        }
        rows.extend(_ranking_rows(scores, equipment=device, method="KS-X", view=view))
    return pd.DataFrame(rows)


def _psi_score(reference: np.ndarray, observed: np.ndarray, bins: int) -> float:
    if bins < 2:
        raise ValueError("PSI bins must be at least 2")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    observed_counts = np.histogram(observed, bins=edges)[0].astype(float)
    epsilon = 1e-12
    reference_prob = np.maximum(reference_counts / len(reference), epsilon)
    observed_prob = np.maximum(observed_counts / len(observed), epsilon)
    return float(np.sum((observed_prob - reference_prob) * np.log(observed_prob / reference_prob)))


def rank_psi_x(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    bins: int = 10,
) -> pd.DataFrame:
    """Rank raw features by PSI using bins fixed on the normal pool."""
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    features = _selected_features(X.columns, process_features, view=view)
    values = _numeric_values(X, features)
    groups_str = groups.astype(str)
    normal_mask = groups_str.isin(normal)
    rows: list[dict[str, object]] = []
    for device in abnormal:
        mask = groups_str == device
        scores = {
            feature: _psi_score(values.loc[normal_mask, feature].to_numpy(), values.loc[mask, feature].to_numpy(), bins)
            for feature in features
        }
        rows.extend(_ranking_rows(scores, equipment=device, method="PSI-X", view=view))
    return pd.DataFrame(rows)


def rank_wasserstein_x(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
) -> pd.DataFrame:
    """Rank raw process features by one-dimensional Wasserstein distance."""
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    features = _selected_features(X.columns, process_features, view=view)
    values = _numeric_values(X, features)
    groups_str = groups.astype(str)
    normal_mask = groups_str.isin(normal)
    rows: list[dict[str, object]] = []
    for device in abnormal:
        mask = groups_str == device
        scores = {
            feature: float(
                wasserstein_distance(values.loc[mask, feature].to_numpy(), values.loc[normal_mask, feature].to_numpy())
            )
            for feature in features
        }
        rows.extend(_ranking_rows(scores, equipment=device, method="Wasserstein-X", view=view))
    return pd.DataFrame(rows)


def rank_global_shap(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
) -> pd.DataFrame:
    """Rank every device by the same global mean absolute SHAP baseline."""
    abnormal, _ = _validate_inputs(shap_values, groups, abnormal_devices, normal_devices)
    features = _selected_features(shap_values.columns, process_features, view=view)
    values = shap_values.loc[:, features]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("SHAP values must be finite")
    scores = values.abs().mean(axis=0).to_dict()
    rows: list[dict[str, object]] = []
    for device in abnormal:
        rows.extend(
            _ranking_rows(
                {name: float(scores[name]) for name in features}, equipment=device, method="Global-SHAP", view=view
            )
        )
    return pd.DataFrame(rows)


def _domain_shap_classifier_columns(
    columns: Iterable[str],
    process_features: Iterable[str] | None,
    *,
    include_eqp_in_classifier: bool,
) -> list[str]:
    if include_eqp_in_classifier:
        return list(columns)
    return _selected_features(columns, process_features, view="process")


def rank_domain_shap(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    include_eqp_in_classifier: bool = False,
    model_type: ModelType = "xgboost",
    model_seed: int = 42,
    cv_seed: int | None = None,
    cv_splits: int = 5,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Domain-classifier+SHAP baseline, trained one-vs-rest per device."""
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    ranking_features = _selected_features(X.columns, process_features, view=view)
    classifier_columns = _domain_shap_classifier_columns(
        X.columns,
        process_features,
        include_eqp_in_classifier=include_eqp_in_classifier,
    )
    groups_str = groups.astype(str)
    normal_mask = groups_str.isin(normal)
    rows: list[dict[str, object]] = []
    for device_idx, device in enumerate(abnormal):
        domain_mask = (groups_str == device) | normal_mask
        X_domain = X.loc[domain_mask, classifier_columns]
        y_domain = (groups_str.loc[domain_mask] == device).astype(int)
        if y_domain.nunique() != 2:
            raise ValueError(f"domain classifier for {device!r} does not contain both classes")
        result = fit_cross_fitted_model(
            X_domain,
            y_domain,
            model_type=model_type,
            model_seed=model_seed + device_idx,
            cv_seed=cv_seed,
            cv_splits=cv_splits,
            model_params=model_params,
        )
        values = result.shap_values.loc[:, ranking_features]
        # Domain SHAP is an importance ranking; using all one-vs-rest samples
        # avoids privileging a particular class's sample count.
        scores = values.abs().mean(axis=0).to_dict()
        rows.extend(
            _ranking_rows(
                {name: float(scores[name]) for name in ranking_features},
                equipment=device,
                method="Domain-SHAP",
                view=view,
            )
        )
    return pd.DataFrame(rows)


def rank_m2oe_group(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    seed: int = 42,
    epochs: int = 30,
    n_neighbors: int = 30,
    batch_size: int = 16,
    max_group_rows: int | None = 128,
    num_tries: int = 3,
) -> pd.DataFrame:
    """Rank features by the M2OE anomalous-group masking model, one device at a time.

    Wraps the vendored ``TabularGroupExplainer`` (Angiulli et al., Machine
    Learning 113:7565, 2024; NumPy port in ``vendor/m2oe``). For every
    abnormal device the method learns, without ever touching the KPI target,
    a shared per-feature choice weight over the subspace density contrastive
    loss that separates the device's rows from the pooled normal rows. The
    deterministic ranking score is the learned mask-model weight per feature:
    soft choice weight (in ``[0, 1]``, the continuous relaxation of the
    paper's binary explaining subspace) times the mean absolute learned patch
    magnitude over the group's training pairs. Features outside the chosen
    subspace therefore score ~0, and inside the subspace features that need
    larger counterfactual patches to normalise the group rank higher; ties
    are broken alphabetically by the shared ranking machinery.
    """
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    features = _selected_features(X.columns, process_features, view=view)
    values = _numeric_values(X, features)
    groups_str = groups.astype(str)
    normal_array = values.loc[groups_str.isin(normal)].to_numpy()
    rows: list[dict[str, object]] = []
    for device_idx, device in enumerate(abnormal):
        device_array = values.loc[groups_str == device].to_numpy()
        explainer = TabularGroupExplainer(
            epochs=epochs,
            n_neighbors=n_neighbors,
            batch_size=batch_size,
            max_group_rows=max_group_rows,
            num_tries=num_tries,
            seed=seed + device_idx,
        )
        explanation = explainer.explain_group(device_array, normal_array)
        scores = {
            feature: float(explanation.choice[position] * explanation.mean_abs_mask[position])
            for position, feature in enumerate(features)
        }
        rows.extend(_ranking_rows(scores, equipment=device, method="M2OE-Group", view=view))
    return pd.DataFrame(rows)


def rank_xpe(
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    *,
    y: pd.Series | None = None,
    attribution: CrossFittedAttributions | None = None,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    permutations: int = 8,
    seed: int = 42,
    max_target_rows: int | None = None,
    max_source_rows: int | None = None,
) -> pd.DataFrame:
    """Rank features by XPE performance-change attribution, one device at a time.

    Decker et al., KDD 2024 (arXiv:2408.13648); compact adaptation in
    ``xpe.py``. For every abnormal device the method couples the device's
    rows (current data) to the pooled normal rows (reference data) with exact
    optimal transport, then attributes the logloss change between
    reference-matched and current inputs to features via seeded permutation
    Shapley over splice sets. The monitored model is the pipeline's
    cross-fitted KPI model, passed through ``attribution``; ``y`` is the KPI
    target. See the ``xpe`` module docstring for the documented deviations
    from the paper. The numeric-only transport means the optional
    ``view="all"`` protocol (EQP included) is rejected, exactly like the
    other raw-space baselines.
    """
    if y is None or attribution is None:
        raise ValueError("XPE requires the KPI target y and the pipeline's cross-fitted model attribution")
    abnormal, normal = _validate_inputs(X, groups, abnormal_devices, normal_devices)
    features = _selected_features(X.columns, process_features, view=view)
    # Validate numeric finiteness of the ranking features up front (mirrors the
    # other raw-space baselines); the transport itself re-reads X.
    _numeric_values(X, features)
    if list(attribution.feature_names) != list(X.columns):
        raise ValueError("XPE requires X columns to match the cross-fitted model's training columns")
    y_binary = cast(pd.Series, pd.to_numeric(y, errors="coerce"))
    if y_binary.isna().any() or not set(y_binary.unique()) <= {0, 1}:
        raise ValueError("XPE requires a binary 0/1 target")
    y_binary = y_binary.astype(int)
    if len(y_binary) != len(X) or not y_binary.index.equals(X.index):
        raise ValueError("XPE requires y aligned with X (same length and index)")
    groups_str = groups.astype(str)
    normal_positions = np.flatnonzero(groups_str.isin(normal).to_numpy())
    rows: list[dict[str, object]] = []
    for device_idx, device in enumerate(abnormal):
        target_positions = np.flatnonzero((groups_str == device).to_numpy())
        scores = xpe_feature_scores(
            attribution.model,
            X,
            y_binary,
            target_positions,
            normal_positions,
            features,
            permutations=permutations,
            seed=seed + device_idx,
            max_target_rows=max_target_rows,
            max_source_rows=max_source_rows,
        )
        rows.extend(_ranking_rows(scores, equipment=device, method="XPE", view=view))
    return pd.DataFrame(rows)


def rank_baseline_features(
    method: str,
    *,
    X: pd.DataFrame,
    groups: pd.Series,
    abnormal_devices: Iterable[str],
    normal_devices: Iterable[str],
    shap_values: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    attribution: CrossFittedAttributions | None = None,
    process_features: Iterable[str] | None = None,
    view: str = "process",
    **kwargs: Any,
) -> pd.DataFrame:
    """Dispatch a named baseline through one stable ranking interface."""
    methods = {
        "KS-X": rank_ks_x,
        "PSI-X": rank_psi_x,
        "Wasserstein-X": rank_wasserstein_x,
        "Global-SHAP": rank_global_shap,
        "Domain-SHAP": rank_domain_shap,
        "M2OE-Group": rank_m2oe_group,
    }
    if method == "GADS":
        if shap_values is None:
            raise ValueError("GADS requires shap_values")
        from .gads import rank_gads_features

        return rank_gads_features(
            shap_values,
            groups,
            abnormal_devices,
            normal_devices,
            process_features=process_features,
            view=view,
        )
    if method == "XPE":
        return rank_xpe(
            X,
            groups,
            abnormal_devices,
            normal_devices,
            y=y,
            attribution=attribution,
            process_features=process_features,
            view=view,
            **kwargs,
        )
    if method not in methods:
        raise ValueError(f"unknown ranking method: {method!r}")
    function = methods[method]
    if method == "Global-SHAP":
        if shap_values is None:
            raise ValueError("Global-SHAP requires shap_values")
        return function(
            shap_values,
            groups,
            abnormal_devices,
            normal_devices,
            process_features=process_features,
            view=view,
            **kwargs,
        )
    if method == "Domain-SHAP":
        return function(
            X,
            groups,
            abnormal_devices,
            normal_devices,
            process_features=process_features,
            view=view,
            **kwargs,
        )
    return function(
        X,
        groups,
        abnormal_devices,
        normal_devices,
        process_features=process_features,
        view=view,
        **kwargs,
    )


ALL_METHODS = ("GADS", "KS-X", "PSI-X", "Domain-SHAP", "Wasserstein-X", "Global-SHAP")
MODERN_BASELINE_METHODS = ("M2OE-Group", "XPE")
RANKING_METHODS: tuple[str, ...] = (*ALL_METHODS, *MODERN_BASELINE_METHODS)

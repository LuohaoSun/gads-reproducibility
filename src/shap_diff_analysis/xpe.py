"""Explanatory Performance Estimation (XPE) feature-shift attribution.

Source
------
Decker, Koebler, Lebacher, Thon, Tresp, Buettner. "Explanatory Model
Monitoring to Understand the Effects of Feature Shifts on Performance."
KDD 2024, arXiv:2408.13648. No official code is published; this module is a
compact faithful adaptation.

Algorithm (paper)
-----------------
XPE attributes the change in a monitored model's performance between a
reference dataset (source) and a current dataset (target) to input features:

1. one exact optimal-transport coupling between source and target samples is
   computed with a squared-Euclidean ground cost (the paper uses the EMD
   solver of the POT library on equally sized sample sets and derives a
   deterministic matching from the coupling);
2. for a feature subset ``K`` the value function splices samples: each target
   sample keeps its own values on ``K`` and receives the matched source
   sample's values on the complement (the paper's deterministic
   ``v_T(K) = L(f(T^-1_{K^c}(x_t)), y_hat_t)`` with label transport
   ``y_hat_t`` from the matched source row);
3. Shapley values over features attribute ``v(full) - v(empty)`` - the
   performance change between reference-matched and current inputs - to
   individual features. The paper uses KernelSHAP (3000 samples); this
   implementation uses permutation Shapley with a fixed seed.

Deviations from the paper (all deliberate, documented for review)
-----------------------------------------------------------------
1. Shapley attribution uses seeded permutation sampling instead of KernelSHAP
   (both are standard unbiased/consistent estimators of the same values);
   the number of permutations is explicit and the estimator is deterministic
   given the seed.
2. The OT coupling is solved as an exact linear program via
   ``scipy.optimize.linprog`` (HiGHS) instead of POT's EMD solver; the
   discrete Kantorovich problem is identical.
3. The monitored model is the experiment pipeline's cross-fitted KPI model
   (the final estimator fitted on all rows, reused from the pipeline's
   :class:`~shap_diff_analysis.modeling.CrossFittedAttributions`). The paper
   monitors a deployed model; out-of-fold predictions cannot be reused
   because spliced samples are synthetic rows, so the fitted model predicts
   on them.
4. The performance metric is binary cross-entropy (logloss) against
   transported labels, matching the paper's experiments.
5. OT costs are computed on features standardised by reference-pool
   statistics so that heterogeneous feature scales contribute comparably to
   the squared-Euclidean cost; the splicing itself happens in the original
   feature space.
6. Equal sample counts are obtained by deterministically subsampling the
   larger of the two pools (seeded), as the paper matches equally sized sets.
7. Transport and attribution operate on the numeric process features only;
   the contextual EQP covariate is kept at each target row's own value
   (ranking view is the process view, consistent with the other baselines).
8. Optional explicit row caps (``max_target_rows`` / ``max_source_rows``)
   subsample the device rows and the normal pool before the OT solve; the
   subsampling is seeded and stratified by the binary KPI label
   (largest-remainder proportional allocation), and caps only ever apply
   when the caller passes them - there is no implicit default cap. Standard
   EMD sizes grow quadratically with row counts, so caps keep the LP tractable
   at experiment-matrix scale.
9. Features that are exactly constant in the equal-count reference subsample
   (zero variance - frequent for median-imputed sparse SECOM sensors once
   the subsample is drawn) cannot be standardised and cannot influence the
   OT coupling or the splice value function in a meaningful way. They enter
   neither the cost nor the Shapley value function and instead receive a
   deterministic bottom score (``-inf``; the attribution scale is signed, so
   ``0.0`` would not guarantee the bottom), tie-broken by feature name. They
   keep their ranking position: XPE always ranks exactly the full
   process-feature set like every other method, because dropping them would
   shrink the candidate space and bias MRR/FAR in XPE's favour. The
   degenerate feature list is emitted at DEBUG level on the
   ``shap_diff_analysis.xpe`` logger (the ranking-row contract carries no
   metadata slot). Genuinely invalid inputs - empty pools, non-finite
   values, non-binary targets, or a reference in which every feature is
   constant - still raise explicit errors.

Usage contract: one call per abnormal equipment (one-vs-rest protocol);
the reference pool is the pooled normal-device rows.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

_EPSILON_VARIANCE = 1e-12
_DEGENERATE_SCORE = float("-inf")
_LOGGER = logging.getLogger(__name__)


def _stratified_subsample(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample ``positions`` to at most ``cap`` rows, stratified by binary label.

    Per-class allocation uses proportional shares with the largest-remainder
    rule; a class smaller than its share contributes everything it has and the
    surplus moves to the other class (always feasible because the cap is
    smaller than the total row count in this branch). The union of the
    per-class draws is sorted, so the result is a deterministic function of
    ``(positions, labels, cap, rng state)``.
    """
    if cap < 1:
        raise ValueError("row cap must be at least 1")
    if len(positions) != len(labels):
        raise ValueError("positions and labels must have the same length")
    if len(positions) <= cap:
        return positions
    positive = positions[labels == 1]
    negative = positions[labels == 0]
    ideal_positive = cap * len(positive) / len(positions)
    ideal_negative = cap * len(negative) / len(positions)
    allocation_positive = int(np.floor(ideal_positive))
    allocation_negative = int(np.floor(ideal_negative))
    remainder = cap - allocation_positive - allocation_negative
    if remainder and (ideal_positive - allocation_positive) >= (ideal_negative - allocation_negative):
        allocation_positive += 1
    elif remainder:
        allocation_negative += 1
    if allocation_positive > len(positive):
        allocation_negative += allocation_positive - len(positive)
        allocation_positive = len(positive)
    if allocation_negative > len(negative):
        allocation_positive += allocation_negative - len(negative)
        allocation_negative = len(negative)
    if (
        allocation_positive + allocation_negative != cap
        or allocation_positive > len(positive)
        or allocation_negative > len(negative)
    ):
        raise RuntimeError(f"stratified allocation failed: {allocation_positive}+{allocation_negative} != {cap}")
    selected_positive = (
        rng.choice(positive, size=allocation_positive, replace=False) if allocation_positive else positive[:0]
    )
    selected_negative = (
        rng.choice(negative, size=allocation_negative, replace=False) if allocation_negative else negative[:0]
    )
    return np.sort(np.concatenate([selected_positive, selected_negative]))


def _validate_xpe_inputs(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
) -> None:
    if len(y) != len(X) or not y.index.equals(X.index):
        raise ValueError("XPE requires y aligned with X (same length and index)")
    if not set(feature_names) <= set(X.columns):
        raise ValueError("XPE ranking features must be columns of X")
    encoded = tuple(getattr(model, "feature_names_in_", ()))
    if encoded and list(encoded) != list(X.columns):
        raise ValueError("XPE requires X columns to match the monitored model's training columns")
    y_numeric = cast(pd.Series, pd.to_numeric(y, errors="coerce"))
    if y_numeric.isna().any() or not set(y_numeric.unique()) <= {0, 1}:
        raise ValueError("XPE requires a binary 0/1 target")


def _emd_coupling(cost: np.ndarray) -> np.ndarray:
    """Solve the exact discrete OT problem and return the coupling matrix."""
    n_sources, n_targets = cost.shape
    variable_index = np.arange(n_sources * n_targets)
    source_of_var = np.repeat(np.arange(n_sources), n_targets)
    target_of_var = np.tile(np.arange(n_targets), n_sources)
    rows = np.concatenate([source_of_var, n_sources + target_of_var])
    columns = np.concatenate([variable_index, variable_index])
    data = np.ones(2 * n_sources * n_targets, dtype=float)
    constraints = coo_matrix(
        (data, (rows, columns)),
        shape=(n_sources + n_targets, n_sources * n_targets),
    ).tocsr()
    marginals = np.concatenate([np.full(n_sources, 1.0 / n_sources), np.full(n_targets, 1.0 / n_targets)])
    result = linprog(
        cost.ravel(),
        A_eq=constraints,
        b_eq=marginals,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"XPE optimal-transport problem failed: {result.message}")
    return np.asarray(result.x, dtype=float).reshape(n_sources, n_targets)


def _standardised_cost(
    source_values: np.ndarray,
    target_values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    source_centred = (source_values - mean[np.newaxis, :]) / scale[np.newaxis, :]
    target_centred = (target_values - mean[np.newaxis, :]) / scale[np.newaxis, :]
    differences = source_centred[:, np.newaxis, :] - target_centred[np.newaxis, :, :]
    return np.einsum("ijk,ijk->ij", differences, differences)


def _permutation_shapley(
    value_function: Callable[[tuple[int, ...]], float],
    n_features: int,
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Estimate Shapley values of ``value_function`` by seeded permutations."""
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    rng = np.random.default_rng(seed)
    baseline = value_function(())
    shapley = np.zeros(n_features)
    for _ in range(permutations):
        order = rng.permutation(n_features)
        previous = baseline
        included: list[int] = []
        for feature in order:
            included.append(int(feature))
            current = value_function(tuple(sorted(included)))
            shapley[feature] += current - previous
            previous = current
    return shapley / permutations


def xpe_feature_scores(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    feature_names: list[str],
    *,
    permutations: int = 8,
    seed: int = 42,
    max_target_rows: int | None = None,
    max_source_rows: int | None = None,
) -> dict[str, float]:
    """Attribute the source-to-target logloss change to ``feature_names``.

    model: Monitored binary classifier exposing ``predict_proba`` (the
        pipeline's cross-fitted KPI model).
    X: Model input frame containing every training column of ``model``.
    y: Binary target aligned with ``X``; also the stratification variable
        for the optional row caps.
    target_positions: Positional row indices of the current (abnormal) rows.
    source_positions: Positional row indices of the reference (normal) rows.
    feature_names: Numeric ranking features; transport and attribution are
        restricted to these columns.
    permutations: Permutation-Shapley resamples (fixed-seed).
    seed: Seed for every subsampling step and the Shapley permutations.
    max_target_rows: Optional cap on device (target) rows applied before the
        OT solve; seeded, label-stratified, only active when passed.
    max_source_rows: Optional cap on normal-pool (source) rows applied
        before the OT solve; seeded, label-stratified, only active when
        passed.

    Returns:
        Per-feature Shapley attribution of the logloss change; a higher
        score means the feature's shift contributes more to the performance
        change and is therefore ranked as more anomalous. Features that are
        exactly constant in the reference subsample receive the
        deterministic bottom score ``-inf`` (see module docstring, deviation
        9) but keep their ranking position, so the returned mapping always
        covers every entry of ``feature_names``.
    """
    _validate_xpe_inputs(model, X, y, feature_names)
    if max_target_rows is not None and max_target_rows < 1:
        raise ValueError("max_target_rows must be at least 1 when provided")
    if max_source_rows is not None and max_source_rows < 1:
        raise ValueError("max_source_rows must be at least 1 when provided")
    subsample_rng = np.random.default_rng(seed)
    label_values = cast(pd.Series, pd.to_numeric(y)).astype(int).to_numpy()
    if max_target_rows is not None:
        target_positions = _stratified_subsample(
            target_positions, label_values[target_positions], cap=max_target_rows, rng=subsample_rng
        )
    if max_source_rows is not None:
        source_positions = _stratified_subsample(
            source_positions, label_values[source_positions], cap=max_source_rows, rng=subsample_rng
        )
    if len(target_positions) == 0 or len(source_positions) == 0:
        raise ValueError("XPE requires non-empty target and reference row sets")
    n = min(len(target_positions), len(source_positions))
    if len(target_positions) > n:
        target_positions = np.sort(subsample_rng.choice(target_positions, size=n, replace=False))
    if len(source_positions) > n:
        source_positions = np.sort(subsample_rng.choice(source_positions, size=n, replace=False))

    source_values = X.iloc[source_positions].loc[:, feature_names].to_numpy(dtype=float)
    target_values = X.iloc[target_positions].loc[:, feature_names].to_numpy(dtype=float)
    mean = source_values.mean(axis=0)
    scale = source_values.std(axis=0, ddof=1) if len(source_positions) > 1 else source_values.std(axis=0, ddof=0)
    degenerate_mask = scale <= _EPSILON_VARIANCE
    informative_positions = [position for position, degenerate in enumerate(degenerate_mask) if not degenerate]
    if not informative_positions:
        raise ValueError("XPE requires at least one process feature with positive reference variance")
    if degenerate_mask.any():
        _LOGGER.debug(
            "XPE reference subsample has %d/%d exactly constant features scored at the bottom: %s",
            int(degenerate_mask.sum()),
            len(feature_names),
            [feature_names[position] for position in np.flatnonzero(degenerate_mask)],
        )
    informative_names = [feature_names[position] for position in informative_positions]

    coupling = _emd_coupling(
        _standardised_cost(
            source_values[:, informative_positions],
            target_values[:, informative_positions],
            mean[informative_positions],
            scale[informative_positions],
        )
    )
    matched_source_positions = source_positions[np.argmax(coupling, axis=0)]

    base_frame = X.iloc[matched_source_positions].reset_index(drop=True)
    target_frame = X.iloc[target_positions].reset_index(drop=True)
    matched_labels = cast(pd.Series, pd.to_numeric(y)).astype(int).iloc[matched_source_positions].reset_index(drop=True)

    def spliced_logloss(subset: tuple[int, ...]) -> float:
        frame = base_frame
        if subset:
            names = [informative_names[position] for position in subset]
            frame = base_frame.copy()
            frame.loc[:, names] = target_frame.loc[:, names].to_numpy()
        probabilities = model.predict_proba(frame)[:, 1]
        return float(log_loss(matched_labels, probabilities, labels=[0, 1]))

    shapley = _permutation_shapley(spliced_logloss, len(informative_names), permutations=permutations, seed=seed)
    scores = {name: float(value) for name, value in zip(informative_names, shapley, strict=True)}
    for name, degenerate in zip(feature_names, degenerate_mask, strict=True):
        if degenerate:
            scores[name] = _DEGENERATE_SCORE
    return scores

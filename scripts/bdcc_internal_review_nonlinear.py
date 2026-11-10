#!/usr/bin/env python3
"""Internal-review validation: Dataset A with a deliberately nonlinear label.

The published Dataset A generator draws the KPI from a linear-logistic model.
This script keeps its independent feature/EQP bundle and replaces only the
label-generating conditional probability with a fixed, documented nonlinear
function.  The goal is to check whether the GADS advantage survives when the
true conditional probability is not a linear GLM and is not the same function
family that XGBoost is fitted to.

Design summary
--------------
* Features, EQP assignment and all scenario injections come from
  ``generate_dataset_bundle`` unchanged.  The returned manifest
  (``abnormal_devices``, ``normal_devices``, ``root_causes``,
  ``harmless_features``) is therefore the original ground truth.
* The normal logit is

  ``eta = -2.5 + sum_j w_j * tanh(1.2 * x_j)
          + 0.35 * tanh(0.8*x_1) * tanh(0.8*x_2)
          + sum_s w_s * tanh(c_s * x_s / sigma_s)``

  where ``x_j`` are the ten useful features and ``x_s`` the three special
  features.  The pairwise term makes the label non-additive as well as
  nonlinear; a linear GLM cannot reproduce it, and an additive GAM can only
  approximate it.  Dividing by ``sigma_s`` puts each special feature on its
  generation scale (``feat_special_imp`` is drawn with SD 2, the other two
  with SD 1), so the coefficients are comparable response-surface slopes.
* A mechanism-shift fault follows a textbook response-surface (RSM)
  regime-coupled bilinear form: for the targeted device only, the response
  gains

  ``2.0 * x_load * (x_s / sigma_s)``  with ``x_load = feat_useful_1``.

  The raw feature marginals are untouched: only the conditional probability
  given EQP changes.  Because ``x_load`` and ``x_s`` are independent and
  centered, this bilinear term has zero population linear slope in ``x_s``
  (``cov(x_s, x_load * x_s) = 0``); it changes the conditional probability
  through a load-dependent slope, i.e. a regime-coupled interaction rather
  than a location or scale shift of the raw feature.  This is a statistical
  benchmark mechanism, not a physical claim.
* S1 uses the original generator's location shift on ``feat_special_dist`` and
  keeps that feature in the label.  S4's harmless features never enter the
  label at all.  In S3 each abnormal device's mechanism is applied to its own
  rows only, and every diagnostic method uses the designated normal pool as
  its sole reference.

Methods compared: GADS (cross-fitted XGBoost SHAP + Wasserstein), joint L2
interaction GLM (``GLM-Int-Joint-L2``), and additive-spline interaction GAM
(``GAM-Int``), matching the competitor definitions in BKCC revision Pack 2.

Statistics: per scenario and metric across 20 seeds, mean, sample SD and 95%
BCa bootstrap CI (10,000 resamples).  GADS-versus-competitor differences are
paired by seed and summarized with the same BCa procedure.  A pooled
``POOLED_S1_S4M`` scope additionally averages MRR/HR@1/HR@5 equally over S1,
S2, S3 and S4-main within each ``(seed, method)`` cell before the BCa summary,
giving an auditable blind-prior cross-scenario average and its paired
GADS-minus-competitor differences.  Metrics whose manifest is empty
(S4-null root-cause metrics; FAR in S1-S3 because those bundles have no
harmless features) are reported as ``applicable=False`` with ``n=0`` and NaN
summaries, never as zero.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib import metadata
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, cast
import zlib

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.gads import rank_gads_features
from shap_diff_analysis.generate_data import N_DEVICES, SPECIAL_FEATURES, generate_dataset_bundle
from shap_diff_analysis.metrics import evaluate_rankings, evaluate_rankings_per_equipment
from shap_diff_analysis.modeling import fit_cross_fitted_model
from shap_diff_analysis.scenario import get_scenario_spec
from shap_diff_analysis.statistics import metric_applicable

SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4-main", "S4-null")
POOLED_SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4-main")
POOLED_METRICS: tuple[str, ...] = ("mrr", "hr_at_1", "hr_at_5")
POOLED_LABEL = "POOLED_S1_S4M"
USEFUL_FEATURES: tuple[str, ...] = tuple(f"feat_useful_{index}" for index in range(1, 11))
USEFUL_WEIGHTS: tuple[float, ...] = (0.90, 1.00, 1.10, 0.85, 1.05, 0.95, 1.15, 0.80, 1.00, 0.90)
USEFUL_TANH_SCALE = 1.20
PAIRWISE_WEIGHT = 0.35
PAIRWISE_SCALE = 0.80
INTERCEPT = -2.50
SPECIAL_SIGMA: dict[str, float] = {
    "feat_special_dist": 1.0,
    "feat_special_imp": 2.0,
    "feat_special_mix": 1.0,
}
SPECIAL_RESPONSE: dict[str, tuple[float, float]] = {
    "feat_special_dist": (1.00, 0.60),
    "feat_special_imp": (1.60, 1.00),
    "feat_special_mix": (1.40, 0.90),
}
COUPLED_LOAD_FEATURE = "feat_useful_1"
FAULT_COUPLING_WEIGHT = 2.00
LABEL_GENERATOR = "bdcc_internal_review_nonlinear_rsm_v2"
LABEL_RNG_TAG = 20261109
PILOT_SEEDS: tuple[int, ...] = (0, 1)

GADS_METHOD = "GADS"
GLM_METHOD = "GLM-Int-Joint-L2"
GAM_METHOD = "GAM-Int"
METHODS: tuple[str, ...] = (GADS_METHOD, GLM_METHOD, GAM_METHOD)
METRIC_COLUMNS: tuple[str, ...] = ("mrr", "hr_at_1", "hr_at_5", "far_at_1", "far_at_5")

RIDGE_C = 1.0
GAM_C = 1e6
SPLINE_N_KNOTS = 4
SPLINE_DEGREE = 3
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Nonlinear label generation
# ---------------------------------------------------------------------------


def _special_normal_term(values: np.ndarray, *, weight: float, scale: float, sigma: float) -> np.ndarray:
    """Return ``weight * tanh(scale * x / sigma)`` for one special feature."""
    if sigma <= 0.0:
        raise ValueError(f"special-feature sigma must be positive, got {sigma!r}")
    return weight * np.tanh(scale * np.asarray(values, dtype=float) / sigma)


def _special_mechanism_delta(
    values: np.ndarray,
    load: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Return the regime-coupled bilinear response-surface fault term.

    The fault adds ``FAULT_COUPLING_WEIGHT * x_load * (x_s / sigma)``.  Both
    ``x_load`` and ``x_s`` are centered and independent in the generator, so
    the population linear slope of the term in ``x_s`` is exactly zero.  The
    term still changes ``P(Y|EQP)`` because it makes the slope of the special
    feature depend on the load regime.
    """
    if sigma <= 0.0:
        raise ValueError(f"special-feature sigma must be positive, got {sigma!r}")
    values_array = np.asarray(values, dtype=float)
    load_array = np.asarray(load, dtype=float)
    if values_array.shape != load_array.shape:
        raise ValueError("mechanism delta requires matching special and load arrays")
    return FAULT_COUPLING_WEIGHT * load_array * (values_array / sigma)


def _base_logit(data: pd.DataFrame) -> np.ndarray:
    """Compute the normal conditional logit for every row."""
    eta = np.full(len(data), INTERCEPT, dtype=float)
    for weight, feature in zip(USEFUL_WEIGHTS, USEFUL_FEATURES, strict=True):
        eta += weight * np.tanh(USEFUL_TANH_SCALE * data[feature].to_numpy(dtype=float))
    eta += (
        PAIRWISE_WEIGHT
        * np.tanh(PAIRWISE_SCALE * data[USEFUL_FEATURES[0]].to_numpy(dtype=float))
        * np.tanh(PAIRWISE_SCALE * data[USEFUL_FEATURES[1]].to_numpy(dtype=float))
    )
    for feature, (weight, scale) in SPECIAL_RESPONSE.items():
        eta += _special_normal_term(
            data[feature].to_numpy(dtype=float),
            weight=weight,
            scale=scale,
            sigma=SPECIAL_SIGMA[feature],
        )
    return eta


def _validate_bundle_manifest(bundle: DatasetBundle) -> None:
    """Fail loudly if the borrowed bundle disagrees with the scenario contract."""
    spec = get_scenario_spec(bundle.scenario)
    if tuple(bundle.abnormal_devices) != tuple(spec.abnormal_devices):
        raise ValueError(
            f"{bundle.scenario}: abnormal_devices {bundle.abnormal_devices} != contract {spec.abnormal_devices}"
        )
    if dict(bundle.root_causes) != dict(spec.root_causes):
        raise ValueError(f"{bundle.scenario}: root_causes differ from the scenario contract")
    if tuple(bundle.harmless_features) != tuple(spec.harmless_features):
        raise ValueError(f"{bundle.scenario}: harmless_features differ from the scenario contract")
    if set(bundle.abnormal_devices) & set(bundle.normal_devices):
        raise ValueError(f"{bundle.scenario}: abnormal and normal pools overlap")
    expected_pool = set(bundle.abnormal_devices) | set(bundle.normal_devices)
    present_pool = set(bundle.data[bundle.group_column].astype(str).unique())
    if present_pool != expected_pool:
        raise ValueError(f"{bundle.scenario}: device pools do not partition the observed EQP values")
    if len(bundle.process_features) != 100:
        raise ValueError(f"{bundle.scenario}: expected 100 process features, got {len(bundle.process_features)}")
    for feature in (*USEFUL_FEATURES, *SPECIAL_FEATURES):
        if feature not in bundle.process_features:
            raise ValueError(f"{bundle.scenario}: missing required label feature {feature!r}")


def nonlinear_probabilities(
    bundle: DatasetBundle,
    *,
    apply_mechanism: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[str, ...]]]:
    """Return ``(eta, probability, mechanism_overrides)`` for a bundle.

    ``apply_mechanism=False`` is used only by tests and diagnostics to verify
    that a mechanism fault changes the conditional probability while leaving
    the raw feature marginal untouched.
    """
    _validate_bundle_manifest(bundle)
    spec = get_scenario_spec(bundle.scenario)
    data = bundle.data
    eta = _base_logit(data)
    groups = data[bundle.group_column].astype(str).to_numpy()
    overrides: dict[str, tuple[str, ...]] = {}
    for device, features in spec.mechanism_shift_features.items():
        mask = groups == str(device)
        if not mask.any():
            raise ValueError(f"{bundle.scenario}: mechanism device {device!r} has no rows")
        if device not in bundle.abnormal_devices:
            raise ValueError(f"{bundle.scenario}: mechanism device {device!r} is not declared abnormal")
        if set(features) - set(bundle.root_causes.get(str(device), ())):
            raise ValueError(f"{bundle.scenario}: mechanism features for {device!r} are not declared root causes")
        for feature in features:
            if feature not in SPECIAL_RESPONSE:
                raise ValueError(f"{bundle.scenario}: unsupported mechanism feature {feature!r}")
            if apply_mechanism:
                values = data[feature].to_numpy(dtype=float)[mask]
                loads = data[COUPLED_LOAD_FEATURE].to_numpy(dtype=float)[mask]
                eta[mask] += _special_mechanism_delta(values, loads, sigma=SPECIAL_SIGMA[feature])
        overrides[str(device)] = tuple(features)
    return eta, expit(eta), overrides


def generate_nonlinear_bundle(
    *,
    scenario: str = "S1",
    n_samples: int = 5_000,
    seed: int = 42,
    noise_level: float = 0.2,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
) -> DatasetBundle:
    """Borrow a Dataset A feature bundle and regenerate only the KPI label."""
    base = generate_dataset_bundle(
        scenario=scenario,
        n_samples=n_samples,
        seed=seed,
        noise_level=noise_level,
        n_devices=n_devices,
        distribution_shift=distribution_shift,
        mechanism_scale=mechanism_scale,
        harmless_drift=harmless_drift,
    )
    _validate_bundle_manifest(base)
    eta, probability, overrides = nonlinear_probabilities(base, apply_mechanism=True)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), LABEL_RNG_TAG]))
    target = rng.binomial(1, probability).astype(np.int8)
    data = base.data.copy(deep=True)
    data[base.target_column] = target
    metadata = dict(base.metadata)
    metadata["dataset"] = "Dataset A nonlinear-label overlay"
    metadata["dataset_kind"] = "pure_synthetic_nonlinear_label_overlay"
    metadata["source_dataset"] = "Dataset A"
    metadata["label_generator"] = LABEL_GENERATOR
    metadata["nonlinear_label_params"] = {
        "form": "regime_coupled_bilinear_response_surface",
        "intercept": INTERCEPT,
        "useful_features": list(USEFUL_FEATURES),
        "useful_weights": list(USEFUL_WEIGHTS),
        "useful_tanh_scale": USEFUL_TANH_SCALE,
        "pairwise_weight": PAIRWISE_WEIGHT,
        "pairwise_scale": PAIRWISE_SCALE,
        "special_sigma": dict(SPECIAL_SIGMA),
        "special_response": {key: list(value) for key, value in SPECIAL_RESPONSE.items()},
        "coupled_load_feature": COUPLED_LOAD_FEATURE,
        "fault_coupling_weight": FAULT_COUPLING_WEIGHT,
        "mechanism_overrides": {key: list(value) for key, value in overrides.items()},
        "label_rng_tag": LABEL_RNG_TAG,
        "noise_model": "Bernoulli(probability); no additional Gaussian label noise",
    }
    metadata["nonlinear_label_diagnostics"] = {
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "logit_mean": float(eta.mean()),
        "logit_std": float(eta.std()),
        "positive_rate": float(target.mean()),
    }
    return DatasetBundle(
        data=data,
        scenario=base.scenario,
        seed=base.seed,
        group_column=base.group_column,
        target_column=base.target_column,
        process_features=base.process_features,
        abnormal_devices=base.abnormal_devices,
        normal_devices=base.normal_devices,
        root_causes=base.root_causes,
        harmless_features=base.harmless_features,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Competitor methods (same definitions as Pack 2)
# ---------------------------------------------------------------------------


def _ranking_rows(scores: dict[str, float], *, equipment: str, method: str) -> list[dict[str, Any]]:
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    return [
        {
            "equipment": equipment,
            "feature": feature,
            "score": float(scores[feature]),
            "rank": rank,
            "view": "process",
            "method": method,
        }
        for rank, feature in enumerate(ordered, start=1)
    ]


def _one_vs_normal_pool_mask(group_values: np.ndarray, device: str, normal_devices: Sequence[str]) -> np.ndarray:
    mask = (group_values == str(device)) | np.isin(group_values, [str(entry) for entry in normal_devices])
    if not mask.any():
        raise ValueError(f"one-vs-normal-pool sample for {device!r} is empty")
    return np.asarray(mask, dtype=bool)


def _rank_joint_l2_glm(
    X_process: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank features by |beta_{x_j:g}| in a joint L2 interaction GLM."""
    y_values = np.asarray(y.to_numpy(dtype=int), dtype=int)
    group_values = groups.astype(str).to_numpy()
    feature_names = list(X_process.columns)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for device in abnormal_devices:
        selected = _one_vs_normal_pool_mask(group_values, device, normal_devices)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.asarray(X_process.loc[selected].to_numpy(dtype=float), dtype=float))
        indicator = (group_values[selected] == str(device)).astype(float)[:, None]
        design = np.column_stack([scaled, indicator, scaled * indicator])
        model = LogisticRegression(C=RIDGE_C, solver="lbfgs", max_iter=1000, tol=1e-4)
        model.fit(design, y_values[selected])
        coefficients = np.asarray(model.coef_, dtype=float).ravel()
        n_features = len(feature_names)
        if coefficients.shape[0] != 2 * n_features + 1:
            raise RuntimeError(f"unexpected joint GLM coefficient count: {coefficients.shape[0]}")
        interaction = np.abs(coefficients[n_features + 1 :])
        scores = {feature: float(interaction[position]) for position, feature in enumerate(feature_names)}
        diagnostics[str(device)] = {
            "n_rows": int(selected.sum()),
            "max_abs_beta_interaction": float(np.max(interaction)),
            "median_abs_beta_interaction": float(np.median(interaction)),
            "n_nonzero_interaction": int(np.sum(interaction > 1e-8)),
        }
        rows.extend(_ranking_rows(scores, equipment=str(device), method=GLM_METHOD))
    return pd.DataFrame(rows), diagnostics


def _binomial_deviance(probabilities: np.ndarray, y: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-2.0 * np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def _rank_gam_interaction(
    X_process: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank features by the additive-spline interaction deviance drop."""
    y_values = np.asarray(y.to_numpy(dtype=int), dtype=float)
    group_values = groups.astype(str).to_numpy()
    feature_names = list(X_process.columns)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for device in abnormal_devices:
        selected = _one_vs_normal_pool_mask(group_values, device, normal_devices)
        selected_values = np.asarray(X_process.loc[selected].to_numpy(dtype=float), dtype=float)
        indicator = (group_values[selected] == str(device)).astype(float)[:, None]
        target = y_values[selected]
        scores: dict[str, float] = {}
        deltas: list[float] = []
        for feature_position, feature in enumerate(feature_names):
            values = selected_values[:, feature_position][:, None]
            if np.std(values) <= 0:
                scores[feature] = 0.0
                deltas.append(0.0)
                continue
            transformer = SplineTransformer(
                n_knots=SPLINE_N_KNOTS,
                degree=SPLINE_DEGREE,
                include_bias=False,
                knots="uniform",
                extrapolation="constant",
            )
            basis = transformer.fit_transform(values)
            additive = np.column_stack([basis, indicator])
            interaction = np.column_stack([basis, indicator, basis * indicator])
            null_model = LogisticRegression(C=GAM_C, max_iter=1000, solver="lbfgs")
            alt_model = LogisticRegression(C=GAM_C, max_iter=1000, solver="lbfgs")
            null_model.fit(additive, target)
            alt_model.fit(interaction, target)
            null_deviance = _binomial_deviance(null_model.predict_proba(additive)[:, 1], target)
            alt_deviance = _binomial_deviance(alt_model.predict_proba(interaction)[:, 1], target)
            drop = float(max(null_deviance - alt_deviance, 0.0))
            scores[feature] = drop
            deltas.append(drop)
        delta_array = np.asarray(deltas, dtype=float)
        diagnostics[str(device)] = {
            "n_rows": int(selected.sum()),
            "median_deviance_drop": float(np.median(delta_array)),
            "max_deviance_drop": float(np.max(delta_array)),
            "n_drop_gt_3.84": int(np.sum(delta_array > 3.84)),
        }
        rows.extend(_ranking_rows(scores, equipment=str(device), method=GAM_METHOD))
    return pd.DataFrame(rows), diagnostics


def _rank_gads(
    bundle: DatasetBundle,
    *,
    model_seed: int,
    cv_seed: int | None,
    cv_splits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = [*bundle.process_features, bundle.group_column]
    X = bundle.data.loc[:, feature_columns]
    y = cast(pd.Series, pd.to_numeric(bundle.data[bundle.target_column], errors="raise")).astype(int)
    groups = cast(pd.Series, bundle.data[bundle.group_column]).astype(str)
    attribution = fit_cross_fitted_model(
        X,
        y,
        model_type="xgboost",
        model_seed=model_seed,
        cv_seed=cv_seed,
        cv_splits=cv_splits,
    )
    rankings = rank_gads_features(
        attribution.shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
    )
    diagnostics = {
        "oof_auc": attribution.mean_auc,
        "fold_auc": [float(value) for value in attribution.fold_metrics["auc"].to_numpy()],
        "n_features_in_model": len(feature_columns),
    }
    return rankings, diagnostics


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def _linearity_diagnostic(bundle: DatasetBundle, eta: np.ndarray) -> dict[str, float]:
    """Measure how far the true logit is from its best linear approximation."""
    features = [*USEFUL_FEATURES, *SPECIAL_FEATURES]
    design = np.asarray(bundle.data.loc[:, features].to_numpy(dtype=float), dtype=float)
    model = LinearRegression().fit(design, eta)
    residual = eta - model.predict(design)
    return {
        "linear_r2_of_true_logit": float(model.score(design, eta)),
        "linear_mean_abs_residual": float(np.mean(np.abs(residual))),
        "linear_max_abs_residual": float(np.max(np.abs(residual))),
    }


def _run_single(scenario: str, seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(settings["output"])
    run_dir = output_root / "runs" / scenario / f"seed_{seed}"
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.is_file() and not settings["overwrite"]:
        return {"scenario": scenario, "seed": seed, "status": "cached", "run_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle = generate_nonlinear_bundle(
        scenario=scenario,
        n_samples=int(settings["n_samples"]),
        seed=int(seed),
        noise_level=float(settings["noise_level"]),
        n_devices=int(settings["n_devices"]),
        distribution_shift=float(settings["distribution_shift"]),
        mechanism_scale=float(settings["mechanism_scale"]),
        harmless_drift=float(settings["harmless_drift"]),
    )
    eta, probability, overrides = nonlinear_probabilities(bundle, apply_mechanism=True)
    y = cast(pd.Series, pd.to_numeric(bundle.data[bundle.target_column], errors="raise")).astype(int)
    groups = cast(pd.Series, bundle.data[bundle.group_column]).astype(str)
    X_process = bundle.data.loc[:, list(bundle.process_features)]

    timings: dict[str, float] = {}
    method_diagnostics: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []

    start = time.perf_counter()
    gads_rows, gads_diagnostics = _rank_gads(
        bundle,
        model_seed=int(settings["model_seed"]),
        cv_seed=int(settings["cv_seed"]),
        cv_splits=int(settings["cv_splits"]),
    )
    timings[GADS_METHOD] = time.perf_counter() - start
    method_diagnostics[GADS_METHOD] = gads_diagnostics
    frames.append(gads_rows)

    start = time.perf_counter()
    glm_rows, glm_diagnostics = _rank_joint_l2_glm(X_process, y, groups, bundle.abnormal_devices, bundle.normal_devices)
    timings[GLM_METHOD] = time.perf_counter() - start
    method_diagnostics[GLM_METHOD] = glm_diagnostics
    frames.append(glm_rows)

    start = time.perf_counter()
    gam_rows, gam_diagnostics = _rank_gam_interaction(
        X_process, y, groups, bundle.abnormal_devices, bundle.normal_devices
    )
    timings[GAM_METHOD] = time.perf_counter() - start
    method_diagnostics[GAM_METHOD] = gam_diagnostics
    frames.append(gam_rows)

    rankings = pd.concat(frames, ignore_index=True)
    rankings.insert(0, "dataset", str(bundle.metadata["dataset"]))
    rankings.insert(1, "scenario", bundle.scenario)
    rankings.insert(2, "seed", int(seed))
    rankings["is_root_cause"] = [
        feature in set(bundle.root_causes.get(str(device), ()))
        for device, feature in zip(rankings["equipment"], rankings["feature"], strict=True)
    ]
    rankings["is_harmless"] = rankings["feature"].isin(bundle.harmless_features)

    metrics = evaluate_rankings(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=(1, 5),
    )
    metrics.insert(0, "dataset", str(bundle.metadata["dataset"]))
    metrics.insert(1, "scenario", bundle.scenario)
    metrics.insert(2, "seed", int(seed))
    metrics["kpi_auc"] = float(gads_diagnostics["oof_auc"])
    metrics["status"] = "valid"
    equipment_metrics = evaluate_rankings_per_equipment(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=(1, 5),
    )
    equipment_metrics.insert(0, "dataset", str(bundle.metadata["dataset"]))
    equipment_metrics.insert(1, "scenario", bundle.scenario)
    equipment_metrics.insert(2, "seed", int(seed))
    equipment_metrics["kpi_auc"] = float(gads_diagnostics["oof_auc"])
    equipment_metrics["status"] = "valid"

    diagnostics = {
        "scenario": bundle.scenario,
        "seed": int(seed),
        "n_samples": len(bundle.data),
        "n_process_features": len(bundle.process_features),
        "device_pools": {
            "abnormal": list(bundle.abnormal_devices),
            "normal": list(bundle.normal_devices),
        },
        "root_causes": {key: list(value) for key, value in bundle.root_causes.items()},
        "harmless_features": list(bundle.harmless_features),
        "mechanism_overrides": {key: list(value) for key, value in overrides.items()},
        "class_balance": {
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
        },
        "true_probability": {
            "mean": float(probability.mean()),
            "std": float(probability.std()),
            "min": float(probability.min()),
            "max": float(probability.max()),
        },
        "linearity": _linearity_diagnostic(bundle, eta),
        "kpi_oof_auc": float(gads_diagnostics["oof_auc"]),
        "fold_auc": [float(value) for value in gads_diagnostics["fold_auc"]],
        "method_timings_seconds": timings,
        "method_diagnostics": method_diagnostics,
        "elapsed_seconds": None,
    }
    diagnostics["elapsed_seconds"] = time.perf_counter() - started

    metrics.to_csv(metrics_path, index=False)
    equipment_metrics.to_csv(run_dir / "equipment_metrics.csv", index=False)
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle.manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )
    if settings["save_rankings"]:
        rankings.to_parquet(run_dir / "rankings.parquet", index=False)
    return {"scenario": scenario, "seed": seed, "status": "ok", "run_dir": str(run_dir)}


def _worker(payload: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    scenario, seed, settings = payload
    return _run_single(scenario, seed, settings)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _bootstrap_seed(*parts: object) -> int:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(key) & 0xFFFFFFFF)


def _bca_bias_correction(boot_means: np.ndarray, sample_mean: float) -> float:
    """Return the BCa bias-correction quantile using mid-rank tie handling.

    The proportion is ``(n_below + 0.5 * n_equal) / n_bootstrap``.  Strict
    ``<`` would systematically shift intervals for discrete metrics such as
    hit rate and false-alarm rate.  A zero or one proportion makes the bias
    correction mathematically undefined, so it is raised rather than clipped.
    """
    below = int(np.count_nonzero(boot_means < sample_mean))
    equal = int(np.count_nonzero(boot_means == sample_mean))
    proportion = (below + 0.5 * equal) / int(boot_means.size)
    if not 0.0 < proportion < 1.0:
        raise RuntimeError(
            "BCa bias correction is undefined at a bootstrap boundary "
            f"(mid-rank proportion={proportion!r}, n_resamples={boot_means.size})"
        )
    return float(norm.ppf(proportion))


def bca_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bias-corrected and accelerated bootstrap CI for a sample mean.

    Empty input returns ``n=0`` and NaN summaries.  A one-element sample
    returns its mean with undefined interval bounds (NaN).  A zero-variance
    sample returns a degenerate interval at the mean; this is an explicit
    mathematical result, not a hidden fallback.  The bias correction uses the
    mid-rank tie convention from the standard BCa implementation.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "z0": np.nan,
            "acceleration": np.nan,
        }
    if not np.isfinite(array).all():
        raise ValueError("values must be finite; filter undefined metrics before summarization")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else np.nan
    if array.size == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": std,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "z0": np.nan,
            "acceleration": np.nan,
        }
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if np.all(array == array[0]):
        return {
            "n": int(array.size),
            "mean": mean,
            "std": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "z0": np.nan,
            "acceleration": np.nan,
        }

    rng = np.random.default_rng(int(seed))
    boot = rng.choice(array, size=(int(n_resamples), array.size), replace=True).mean(axis=1)
    z0 = _bca_bias_correction(boot, mean)
    jackknife = np.array([np.mean(np.delete(array, index)) for index in range(array.size)])
    jack_mean = float(jackknife.mean())
    deviations = jack_mean - jackknife
    numerator = float(np.sum(deviations**3))
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    if denominator <= 0.0:
        raise RuntimeError(
            "BCa acceleration denominator is non-positive for a non-constant sample; "
            "refusing to substitute a fallback acceleration"
        )
    acceleration = numerator / denominator
    alpha = (1.0 - confidence) / 2.0
    z_low = float(norm.ppf(alpha))
    z_high = float(norm.ppf(1.0 - alpha))

    def adjusted(z_value: float) -> float:
        denominator_value = 1.0 - acceleration * (z0 + z_value)
        if denominator_value == 0.0:
            raise RuntimeError("BCa acceleration produced a singular adjustment")
        return float(norm.cdf(z0 + (z0 + z_value) / denominator_value))

    low = float(np.quantile(boot, adjusted(z_low)))
    high = float(np.quantile(boot, adjusted(z_high)))
    if not np.isfinite(low) or not np.isfinite(high):
        raise RuntimeError("BCa interval is not finite")
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci_low": low,
        "ci_high": high,
        "z0": z0,
        "acceleration": acceleration,
    }


def _validate_metric_cells(
    metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] | None,
    require_complete: bool,
) -> None:
    """Reject duplicate, missing or non-finite cells in complete mode.

    Even in non-complete mode an applicable metric containing NaN is an error:
    silent ``dropna`` would hide a failed or missing run.
    """
    if require_complete:
        if expected_seeds is None:
            raise ValueError("expected_seeds is required when require_complete=True")
        expected_seed_set = {int(value) for value in expected_seeds}
        present_scenarios = set(metrics["scenario"])
        missing_scenarios = set(SCENARIOS) - present_scenarios
        extra_scenarios = present_scenarios - set(SCENARIOS)
        if missing_scenarios or extra_scenarios:
            raise ValueError(
                "scenario set is incomplete or unexpected: "
                f"missing={sorted(missing_scenarios)}, extra={sorted(extra_scenarios)}"
            )
        scenarios = SCENARIOS
        methods = METHODS
    else:
        expected_seed_set = None
        scenarios = tuple(scenario for scenario in SCENARIOS if scenario in set(metrics["scenario"]))
        methods = tuple(method for method in METHODS if method in set(metrics["method"]))

    for scenario, method in product(scenarios, methods):
        cell = metrics.loc[(metrics["scenario"] == scenario) & (metrics["method"] == method)]
        if require_complete and cell.empty:
            raise ValueError(f"missing method={method!r} for scenario={scenario!r}")
        if cell.empty:
            continue
        seeds = cell["seed"].to_numpy(dtype=int)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed rows for scenario={scenario!r}, method={method!r}")
        if expected_seed_set is not None and set(seeds) != expected_seed_set:
            missing = sorted(expected_seed_set - set(seeds))
            extra = sorted(set(seeds) - expected_seed_set)
            raise ValueError(
                f"seed set mismatch for scenario={scenario!r}, method={method!r}: missing={missing}, extra={extra}"
            )
        for metric in METRIC_COLUMNS:
            applicable, _ = metric_applicable(scenario, metric)
            if not applicable:
                continue
            values = cell[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"applicable metric {metric!r} has missing or non-finite values for "
                    f"scenario={scenario!r}, method={method!r}"
                )


def _summary_rows(
    metrics: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
    expected_seeds: Sequence[int] | None = None,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Summarize every scenario/method/metric cell with strict validation."""
    _validate_metric_cells(metrics, expected_seeds=expected_seeds, require_complete=require_complete)
    scenarios = (
        SCENARIOS
        if require_complete
        else tuple(scenario for scenario in SCENARIOS if scenario in set(metrics["scenario"]))
    )
    methods = METHODS if require_complete else tuple(method for method in METHODS if method in set(metrics["method"]))
    rows: list[dict[str, Any]] = []
    for scenario, method, metric in product(scenarios, methods, METRIC_COLUMNS):
        cell = metrics.loc[(metrics["scenario"] == scenario) & (metrics["method"] == method)]
        applicable, note = metric_applicable(scenario, metric)
        values = cell[metric].to_numpy(dtype=float) if applicable else np.array([], dtype=float)
        summary = bca_mean_ci(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=_bootstrap_seed(seed, scenario, method, metric),
        )
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "metric": metric,
                **summary,
                "applicable": bool(applicable),
                "note": note,
                "ci_method": "BCa",
                "n_resamples": int(n_resamples),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


def _paired_rows(
    metrics: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
    expected_seeds: Sequence[int] | None = None,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Summarize same-seed GADS-minus-competitor differences with strict pairing."""
    _validate_metric_cells(metrics, expected_seeds=expected_seeds, require_complete=require_complete)
    scenarios = (
        SCENARIOS
        if require_complete
        else tuple(scenario for scenario in SCENARIOS if scenario in set(metrics["scenario"]))
    )
    competitors = tuple(method for method in METHODS if method != GADS_METHOD)
    rows: list[dict[str, Any]] = []
    for scenario, metric, competitor in product(scenarios, METRIC_COLUMNS, competitors):
        applicable, note = metric_applicable(scenario, metric)
        if applicable:
            reference = metrics.loc[
                (metrics["scenario"] == scenario) & (metrics["method"] == GADS_METHOD), ["seed", metric]
            ]
            other = metrics.loc[(metrics["scenario"] == scenario) & (metrics["method"] == competitor), ["seed", metric]]
            paired = reference.merge(other, on="seed", suffixes=("_gads", "_other"), validate="one_to_one")
            if len(paired) != len(reference) or len(paired) != len(other):
                raise ValueError(
                    f"seed pairing is incomplete for scenario={scenario!r}, metric={metric!r}, "
                    f"competitor={competitor!r}"
                )
            values = (paired[f"{metric}_gads"] - paired[f"{metric}_other"]).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"paired difference has non-finite values for scenario={scenario!r}, "
                    f"metric={metric!r}, competitor={competitor!r}"
                )
        else:
            values = np.array([], dtype=float)
        summary = bca_mean_ci(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=_bootstrap_seed(seed, scenario, metric, competitor, "paired"),
        )
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "reference": GADS_METHOD,
                "competitor": competitor,
                "n_pairs": summary["n"],
                "mean_difference": summary["mean"],
                "sd_difference": summary["std"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "applicable": bool(applicable),
                "note": note,
                "ci_method": "BCa",
                "n_resamples": int(n_resamples),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled S1-S4-main summary (equal weight across the four root-cause scenarios)
# ---------------------------------------------------------------------------


def _pooled_per_seed(metrics: pd.DataFrame, *, expected_seeds: Sequence[int]) -> pd.DataFrame:
    """Return the equal-weight S1/S2/S3/S4-main mean per ``(seed, method)``.

    Each ``(seed, method)`` cell must contain exactly one row for every pooled
    scenario with finite values.  Partial pooling would silently reweight the
    average, so it raises instead.
    """
    subset = metrics.loc[metrics["scenario"].isin(POOLED_SCENARIOS)]
    expected_seed_values = [int(value) for value in expected_seeds]
    expected_keys = {(seed, method) for seed in expected_seed_values for method in METHODS}
    observed_keys = set(map(tuple, subset.loc[:, ["seed", "method"]].itertuples(index=False, name=None)))
    if observed_keys != expected_keys:
        raise ValueError(
            "pooled (seed, method) grid is incomplete or unexpected: "
            f"missing={sorted(expected_keys - observed_keys)}, extra={sorted(observed_keys - expected_keys)}"
        )
    for (seed, method), cell in subset.groupby(["seed", "method"], sort=True):
        scenarios = set(cell["scenario"])
        if scenarios != set(POOLED_SCENARIOS) or len(cell) != len(POOLED_SCENARIOS):
            raise ValueError(
                f"pooled cell seed={seed!r}, method={method!r} must contain one row per pooled scenario; "
                f"got scenarios={sorted(scenarios)}, rows={len(cell)}"
            )
        values = cell.loc[:, list(POOLED_METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"pooled cell seed={seed!r}, method={method!r} has non-finite metric values")
    pooled = (
        subset.groupby(["seed", "method"], as_index=False)[list(POOLED_METRICS)]
        .mean()
        .sort_values(["seed", "method"])
        .reset_index(drop=True)
    )
    return pooled


def _pooled_summary_rows(
    pooled: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Summarize the per-seed pooled mean across the 20 scenario seeds."""
    expected_seed_values = [int(value) for value in expected_seeds]
    rows: list[dict[str, Any]] = []
    for method, metric in product(METHODS, POOLED_METRICS):
        cell = pooled.loc[pooled["method"] == method].sort_values("seed")
        if cell["seed"].astype(int).tolist() != expected_seed_values:
            raise ValueError(f"pooled seed set mismatch for method={method!r}, metric={metric!r}")
        summary = bca_mean_ci(
            cell[metric].to_numpy(dtype=float),
            confidence=confidence,
            n_resamples=n_resamples,
            seed=_bootstrap_seed(seed, POOLED_LABEL, method, metric),
        )
        rows.append(
            {
                "scenario_group": POOLED_LABEL,
                "scenario_set": ";".join(POOLED_SCENARIOS),
                "method": method,
                "metric": metric,
                **summary,
                "applicable": True,
                "note": "",
                "ci_method": "BCa",
                "n_resamples": int(n_resamples),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


def _pooled_paired_rows(
    pooled: pd.DataFrame,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Summarize same-seed pooled GADS-minus-competitor differences."""
    expected_seed_values = [int(value) for value in expected_seeds]
    competitors = tuple(method for method in METHODS if method != GADS_METHOD)
    rows: list[dict[str, Any]] = []
    for metric, competitor in product(POOLED_METRICS, competitors):
        reference = pooled.loc[pooled["method"] == GADS_METHOD, ["seed", metric]]
        other = pooled.loc[pooled["method"] == competitor, ["seed", metric]]
        paired = reference.merge(other, on="seed", suffixes=("_gads", "_other"), validate="one_to_one")
        if paired["seed"].astype(int).tolist() != expected_seed_values:
            raise ValueError(f"pooled seed pairing is incomplete for metric={metric!r}, competitor={competitor!r}")
        values = (paired[f"{metric}_gads"] - paired[f"{metric}_other"]).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"pooled paired difference is non-finite for metric={metric!r}, competitor={competitor!r}")
        summary = bca_mean_ci(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=_bootstrap_seed(seed, POOLED_LABEL, metric, competitor, "paired"),
        )
        rows.append(
            {
                "scenario_group": POOLED_LABEL,
                "scenario_set": ";".join(POOLED_SCENARIOS),
                "metric": metric,
                "reference": GADS_METHOD,
                "competitor": competitor,
                "n_pairs": summary["n"],
                "mean_difference": summary["mean"],
                "sd_difference": summary["std"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "applicable": True,
                "note": "",
                "ci_method": "BCa",
                "n_resamples": int(n_resamples),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation and CLI
# ---------------------------------------------------------------------------


def _protocol(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "bdcc_internal_review_nonlinear",
        "version": 2,
        "label_generator": LABEL_GENERATOR,
        "dataset": {
            "source": "Dataset A feature/EQP bundle, label replaced",
            "n_samples": int(settings["n_samples"]),
            "n_devices": int(settings["n_devices"]),
            "n_process_features": 100,
            "scenarios": list(settings["scenarios"]),
            "seeds": list(settings["seeds"]),
            "source_generation": {
                "noise_level": float(settings["noise_level"]),
                "distribution_shift": float(settings["distribution_shift"]),
                "mechanism_scale": float(settings["mechanism_scale"]),
                "harmless_drift": float(settings["harmless_drift"]),
            },
        },
        "nonlinear_label": {
            "form": "regime_coupled_bilinear_response_surface",
            "intercept": INTERCEPT,
            "useful_features": list(USEFUL_FEATURES),
            "useful_weights": list(USEFUL_WEIGHTS),
            "useful_tanh_scale": USEFUL_TANH_SCALE,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "pairwise_scale": PAIRWISE_SCALE,
            "special_sigma": dict(SPECIAL_SIGMA),
            "special_response": {key: list(value) for key, value in SPECIAL_RESPONSE.items()},
            "coupled_load_feature": COUPLED_LOAD_FEATURE,
            "fault_coupling_weight": FAULT_COUPLING_WEIGHT,
            "label_rng_tag": LABEL_RNG_TAG,
            "noise_model": "Bernoulli(probability); no additional Gaussian label noise",
            "claim": "Statistical mechanism-shift benchmark; not a claim of a physical failure mechanism.",
        },
        "model": {
            "type": "xgboost",
            "model_seed": int(settings["model_seed"]),
            "cv_seed": int(settings["cv_seed"]),
            "cv_splits": int(settings["cv_splits"]),
            "eqp_mode": "contextual_process",
        },
        "methods": {
            GADS_METHOD: {"attribution": "cross-fitted TreeSHAP", "distance": "wasserstein", "bins": 50},
            GLM_METHOD: {"penalty": "l2", "C": RIDGE_C, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4},
            GAM_METHOD: {
                "basis": "cubic B-spline",
                "n_knots": SPLINE_N_KNOTS,
                "degree": SPLINE_DEGREE,
                "C": GAM_C,
                "ranking": "deviance drop of s(x):I(g)",
            },
        },
        "statistics": {
            "metrics": list(METRIC_COLUMNS),
            "pooled_scenarios": list(POOLED_SCENARIOS),
            "pooled_metrics": list(POOLED_METRICS),
            "pooled_aggregation": "equal-weight mean over S1, S2, S3 and S4-main within each (seed, method)",
            "confidence": float(settings["confidence"]),
            "bootstrap": "BCa",
            "n_resamples": int(settings["bootstrap"]),
            "paired_by": "seed",
            "undefined_policy": "applicable=False, n=0, NaN summaries; never zero",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "packages": {
                name: metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap")
            },
        },
        "provenance": {
            "preregistered": False,
            "development_pilot": (
                "Seeds 0 and 1 of every scenario were inspected while specifying the "
                "regime-coupled bilinear response-surface label form (v2) before the full "
                "20-seed evaluation. These pilot seeds are retained in the primary 20-seed "
                "summary for auditability; seeds 2..19 are reported alongside as the "
                "pilot-excluded sensitivity set."
            ),
            "pilot_seeds": [0, 1],
            "pilot_seeds_reused_in_full_run": True,
            "parameters_fixed": "before the full 20-seed evaluation",
            "adverse_results_policy": "all results retained; no tuning on method rankings",
        },
        "counting": {
            "scenario_seed_datasets": int(settings["scenario_seed_datasets"]),
            "methods": len(METHODS),
            "method_evaluations": int(settings["scenario_seed_datasets"]) * len(METHODS),
        },
        "commands": {
            "full": "uv run python scripts/bdcc_internal_review_nonlinear.py --jobs 16 --overwrite",
        },
    }


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _diagnostics_frame(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "runs").glob("*/seed_*/diagnostics.json")):
        payload = _read_json(path)
        rows.append(
            {
                "scenario": payload["scenario"],
                "seed": payload["seed"],
                "n_samples": payload["n_samples"],
                "positive_rate": payload["class_balance"]["positive_rate"],
                "true_probability_mean": payload["true_probability"]["mean"],
                "true_probability_std": payload["true_probability"]["std"],
                "kpi_oof_auc": payload["kpi_oof_auc"],
                "linear_r2_of_true_logit": payload["linearity"]["linear_r2_of_true_logit"],
                "linear_mean_abs_residual": payload["linearity"]["linear_mean_abs_residual"],
                "gads_seconds": payload["method_timings_seconds"].get(GADS_METHOD),
                "glm_seconds": payload["method_timings_seconds"].get(GLM_METHOD),
                "gam_seconds": payload["method_timings_seconds"].get(GAM_METHOD),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_metrics_completeness(metrics: pd.DataFrame, settings: dict[str, Any]) -> None:
    """Require the full scenario-by-seed-by-method grid before aggregation."""
    expected_seeds = [int(value) for value in settings["seeds"]]
    expected_datasets = set(product(SCENARIOS, expected_seeds))
    actual_datasets = set(map(tuple, metrics.loc[:, ["scenario", "seed"]].itertuples(index=False, name=None)))
    missing = sorted(expected_datasets - actual_datasets)
    extra = sorted(actual_datasets - expected_datasets)
    if missing or extra:
        raise ValueError(f"scenario-seed grid is incomplete or unexpected: missing={missing}, extra={extra}")
    if metrics.duplicated(["scenario", "seed", "method"]).any():
        raise ValueError("duplicate (scenario, seed, method) rows in metrics")
    expected_rows = len(expected_datasets) * len(METHODS)
    if len(metrics) != expected_rows:
        raise ValueError(f"expected {expected_rows} method evaluations, found {len(metrics)}")
    _validate_metric_cells(metrics, expected_seeds=expected_seeds, require_complete=True)


def _write_aggregation_provenance(output_dir: Path, settings: dict[str, Any]) -> Path:
    """Record the training source, corrected aggregation source and correction note."""
    manifest_path = output_dir / "run_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    training_source_path = output_dir / "training_source.json"
    training_source = _read_json(training_source_path) if training_source_path.is_file() else {}
    expected_seeds = [int(value) for value in settings["seeds"]]
    training_hash = training_source.get("script_sha256") or manifest.get("script_sha256")
    aggregation_hash = _sha256_file(Path(__file__).resolve())
    same_script = bool(training_hash) and training_hash == aggregation_hash
    run_note = (
        "All 100 scenario-seed datasets and the published summary/sensitivity/pooled tables "
        "were produced by the single RSM v2 script snapshot recorded in training.script_sha256 "
        "(identical to aggregation.script_sha256), replacing the earlier univariate nonlinear "
        "formulation (archived in provenance/v1_univariate_summary_record.json). Running with "
        "--aggregate-only recomputes all summary, sensitivity and pooled CSVs directly from "
        "runs/*/seed_*/metrics.csv without retraining."
        if same_script
        else (
            "Models were trained with the script recorded in training.script_sha256, and "
            "summary/sensitivity/pooled tables were recomputed with aggregation.script_sha256 "
            "from runs/*/seed_*/metrics.csv without retraining or tuning."
        )
    )
    payload = {
        "training": {
            "script_sha256": training_hash,
            "source": training_source.get("source", "RSM v2 full 100-dataset run"),
            "git_commit": manifest.get("git_commit"),
            "python": manifest.get("python"),
            "platform": manifest.get("platform"),
        },
        "aggregation": {
            "script_sha256": aggregation_hash,
            "same_as_training_script": same_script,
            "written_at_unix": time.time(),
            "bca": "mid-rank tie correction; no probability clipping; boundary/singularity raised",
            "completeness": "100 scenario-seed datasets, 20 seeds per method per scenario, applicable metrics finite",
            "sensitivity": (
                "seeds 1..19 and seeds 2..19 secondary only; full 20-seed summary remains primary. "
                "Seeds 0 and 1 were inspected while specifying the v2 label form."
            ),
            "pooled": "equal-weight mean over S1, S2, S3 and S4-main within each (seed, method), then BCa over seeds",
        },
        "provenance_note": run_note,
        "supersedes_v1": "provenance/v1_univariate_summary_record.json",
        "expectations": {
            "scenario_seed_datasets": len(expected_seeds) * len(SCENARIOS),
            "method_evaluations": len(expected_seeds) * len(SCENARIOS) * len(METHODS),
        },
    }
    path = output_dir / "aggregation_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def aggregate(output_dir: Path, settings: dict[str, Any]) -> dict[str, Path]:
    """Write raw, summarized and paired-difference tables for a complete run."""
    metrics_paths = sorted((output_dir / "runs").glob("*/seed_*/metrics.csv"))
    if not metrics_paths:
        raise RuntimeError("no per-run metrics.csv files found; run the experiment before aggregating")
    metrics = pd.concat([pd.read_csv(path) for path in metrics_paths], ignore_index=True)
    metrics = metrics.sort_values(["scenario", "seed", "method"]).reset_index(drop=True)
    _validate_metrics_completeness(metrics, settings)
    expected_seeds = [int(value) for value in settings["seeds"]]
    summary = _summary_rows(
        metrics,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=expected_seeds,
        require_complete=True,
    )
    paired = _paired_rows(
        metrics,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=expected_seeds,
        require_complete=True,
    )
    secondary_seeds = [value for value in expected_seeds if value != 0]
    if not secondary_seeds:
        raise ValueError("seeds 1..19 sensitivity requires at least one non-zero evaluation seed")
    secondary_metrics = metrics.loc[metrics["seed"].isin(secondary_seeds)]
    sensitivity = _summary_rows(
        secondary_metrics,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=secondary_seeds,
        require_complete=True,
    )
    sensitivity.insert(0, "analysis", "secondary_seeds_1_19_excludes_development_seed_0")
    pilot_excluded_seeds = [value for value in expected_seeds if value not in PILOT_SEEDS]
    if not pilot_excluded_seeds:
        raise ValueError("pilot-excluded sensitivity requires at least one seed outside the pilot set")
    pilot_excluded_metrics = metrics.loc[metrics["seed"].isin(pilot_excluded_seeds)]
    sensitivity_pilot_excluded = _summary_rows(
        pilot_excluded_metrics,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=pilot_excluded_seeds,
        require_complete=True,
    )
    sensitivity_pilot_excluded.insert(0, "analysis", "secondary_seeds_2_19_excludes_pilot_seeds_0_and_1")
    pooled = _pooled_per_seed(metrics, expected_seeds=expected_seeds)
    pooled_summary = _pooled_summary_rows(
        pooled,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=expected_seeds,
    )
    pooled_paired = _pooled_paired_rows(
        pooled,
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=expected_seeds,
    )
    pooled_summary_pilot_excluded = _pooled_summary_rows(
        pooled.loc[pooled["seed"].isin(pilot_excluded_seeds)],
        confidence=float(settings["confidence"]),
        n_resamples=int(settings["bootstrap"]),
        seed=int(settings["stats_seed"]),
        expected_seeds=pilot_excluded_seeds,
    )
    pooled_summary_pilot_excluded.insert(0, "analysis", "secondary_seeds_2_19_excludes_pilot_seeds_0_and_1")
    diagnostics = _diagnostics_frame(output_dir)
    paths = {
        "metrics": output_dir / "raw_metrics.csv",
        "summary": output_dir / "summary_metrics.csv",
        "paired": output_dir / "paired_differences.csv",
        "sensitivity_seeds_1_19": output_dir / "sensitivity_seeds_1_19.csv",
        "sensitivity_seeds_2_19": output_dir / "sensitivity_seeds_2_19.csv",
        "pooled_summary": output_dir / "pooled_summary_metrics.csv",
        "pooled_paired": output_dir / "pooled_paired_differences.csv",
        "pooled_summary_seeds_2_19": output_dir / "pooled_summary_metrics_seeds_2_19.csv",
        "diagnostics": output_dir / "diagnostics.csv",
        "aggregation_provenance": output_dir / "aggregation_provenance.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    sensitivity.to_csv(paths["sensitivity_seeds_1_19"], index=False)
    sensitivity_pilot_excluded.to_csv(paths["sensitivity_seeds_2_19"], index=False)
    pooled_summary.to_csv(paths["pooled_summary"], index=False)
    pooled_paired.to_csv(paths["pooled_paired"], index=False)
    pooled_summary_pilot_excluded.to_csv(paths["pooled_summary_seeds_2_19"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    _write_aggregation_provenance(output_dir, settings)
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results/bdcc_internal_review/nonlinear"))
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--n-devices", type=int, default=N_DEVICES)
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--distribution-shift", type=float, default=3.0)
    parser.add_argument("--mechanism-scale", type=float, default=1.0)
    parser.add_argument("--harmless-drift", type=float, default=3.5)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--stats-seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save-rankings", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug limit on the number of runs")
    return parser


def _settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output": str(args.output),
        "n_samples": int(args.n_samples),
        "n_devices": int(args.n_devices),
        "noise_level": float(args.noise_level),
        "distribution_shift": float(args.distribution_shift),
        "mechanism_scale": float(args.mechanism_scale),
        "harmless_drift": float(args.harmless_drift),
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "model_seed": int(args.model_seed),
        "cv_seed": int(args.cv_seed),
        "cv_splits": int(args.cv_splits),
        "bootstrap": int(args.bootstrap),
        "confidence": float(args.confidence),
        "stats_seed": int(args.stats_seed),
        "overwrite": bool(args.overwrite),
        "save_rankings": not bool(args.no_save_rankings),
        "scenario_seed_datasets": len(args.scenarios) * len(args.seeds),
    }


def _validate_settings(settings: dict[str, Any], *, jobs: int) -> None:
    if settings["n_samples"] < settings["n_devices"]:
        raise ValueError("n_samples must be at least n_devices")
    if len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("seeds must be unique")
    if settings["cv_splits"] < 2:
        raise ValueError("cv_splits must be at least 2")
    if not 0.0 < settings["confidence"] < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if settings["bootstrap"] < 100:
        raise ValueError("bootstrap must be at least 100")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs > 1 and "spawn" not in mp.get_all_start_methods():
        raise RuntimeError("jobs>1 requires the 'spawn' multiprocessing start method")


def main(argv: Sequence[str] | None = None) -> int:
    """Run or aggregate the nonlinear-label validation."""
    args = build_parser().parse_args(argv)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings_from_args(args)
    _validate_settings(settings, jobs=int(args.jobs))
    protocol = _protocol(settings)
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    if not args.aggregate_only:
        payloads = [(scenario, int(seed), settings) for scenario in settings["scenarios"] for seed in settings["seeds"]]
        if args.max_runs is not None:
            payloads = payloads[: int(args.max_runs)]
        failures: list[dict[str, Any]] = []
        started = time.perf_counter()
        if int(args.jobs) == 1:
            for scenario, seed, payload_settings in payloads:
                try:
                    result = _run_single(scenario, seed, payload_settings)
                    print(f"[run] {result['status']} {scenario} seed={seed}", flush=True)
                except Exception as error:  # noqa: PERF203 - per-run failures must not abort the batch
                    failures.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[run] FAILED {scenario} seed={seed}: {error}", flush=True)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(args.jobs), mp_context=context) as executor:
                futures = {executor.submit(_worker, payload): (payload[0], payload[1]) for payload in payloads}
                for future in as_completed(futures):
                    scenario, seed = futures[future]
                    try:
                        result = future.result()
                        print(f"[run] {result['status']} {scenario} seed={seed}", flush=True)
                    except Exception as error:
                        failures.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(),
                            }
                        )
                        print(f"[run] FAILED {scenario} seed={seed}: {error}", flush=True)
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        manifest = {
            "created_at_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started,
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "settings": settings,
            "requested_runs": len(payloads),
            "failures": failures,
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    paths = aggregate(output_dir, settings)
    print("[summary] wrote " + ", ".join(f"{name}={path}" for name, path in paths.items()), flush=True)
    failures_path = output_dir / "failures.json"
    if failures_path.is_file() and _read_json(failures_path):
        print("[error] some runs failed; see failures.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

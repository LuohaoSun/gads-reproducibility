#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOptionalOperand=false
"""BDCC revision Pack 2: reviewer-requested competitors plus a clean cost benchmark.

This script produces two self-contained result packs under one output directory:

1. a 20-seed comparison of Reviewer 3's requested competitors
   (``GLM-Int-Marginal``, ``GLM-Int-Joint`` with L2 and L1, ``GAM-Int``,
   ``TreeSHAP-Int-Abs``, ``TreeSHAP-Int-W1`` and ``Expl-Domain-SHAP``) against
   the exact paper-protocol ``GADS`` reference on Dataset A (n=5000) and
   Dataset B (``include_missing_indicators=False``) across all five scenarios;
2. a sequential, uncontended wall-clock and memory benchmark of one diagnosis
   for ``GADS`` (``n_jobs=1`` and ``n_jobs=8``, with a stage breakdown) versus
   ``TreeSHAP-Int``, ``Expl-Domain-SHAP``, ``M2OE-Group``, ``XPE`` and
   ``GLM-Int`` on Dataset A and Dataset B.

Protocol notes
--------------
* Every method ranks exactly the bundle's ``process_features`` (the paper's
  ``contextual_process`` protocol: EQP stays in the KPI model but never in the
  process ranking view).
* Every method uses the one-versus-normal-pool sample for a device ``g``:
  ``(EQP == g) | EQP in normal_pool``.  In the multi-fault scenario ``S3`` the
  two other abnormal devices are therefore excluded from both the reference
  distribution and the GLM/GAM fitting sample; single-fault scenarios are
  unaffected because ``EQP != g`` already equals the normal pool there.
* ``GADS``, ``TreeSHAP-Int`` and ``Expl-Domain-SHAP`` share one cross-fitted
  KPI model per run: 5-fold ``StratifiedKFold(shuffle=True, random_state=42)``
  with the paper's XGBoost settings (``max_depth=6, min_child_weight=1`` on
  Dataset A; ``max_depth=3, min_child_weight=5`` on Dataset B).  The fold loop
  below mirrors ``modeling.fit_cross_fitted_model`` line for line (same
  splitter, preprocessor, estimator seeds, class weights and SHAP
  aggregation); the script verifies OOF SHAP equality against
  ``fit_cross_fitted_model`` before any experiment runs.
* ``TreeSHAP-Int`` uses XGBoost's native ``pred_interactions=True``, which is
  the exact code path ``shap.TreeExplainer(...).shap_interaction_values(...)``
  takes for XGBoost tree-path-dependent explanations (verified in the
  self-check).  Only the process-feature x EQP block is extracted, so the
  reported values are identical while post-processing memory stays small.
* Metrics follow ``metrics.evaluate_rankings`` exactly: MRR is the reciprocal
  rank of the best-ranked annotated root cause per device, HR@K is a
  root-cause hit in the top K, FAR@K is a harmless-feature hit in the top K;
  S4-null root-cause metrics and S1-S3 false-alarm metrics stay NaN.

Runtime environment
-------------------
On headless Linux / WSL2 hosts where ``libcuda.so.1`` is a stub or is
incompatible with the CUDA-enabled XGBoost wheel, ``xgboost.DMatrix`` can
segfault while probing the GPU driver.  Preload a CPU-only ``libcuda.so.1``
stub on ``LD_LIBRARY_PATH`` first; the stub only reports "no CUDA device", so
all computation stays on the CPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from scipy.stats import wasserstein_distance
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import xgboost

from shap_diff_analysis.baselines import rank_domain_shap, rank_m2oe_group, rank_xpe
from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.gads import rank_gads_features
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.metrics import evaluate_rankings, evaluate_rankings_per_equipment
from shap_diff_analysis.modeling import (
    CrossFittedAttributions,
    _aggregate_shap,
    _build_encoded_to_original_mapping,
    _class_weight,
    _extract_binary_shap,
    _make_estimator,
    _make_preprocessor,
    fit_cross_fitted_model,
)
from shap_diff_analysis.secom import generate_dataset_b

# Private modeling helpers are imported on purpose: the competitor pack must
# reuse the exact fold preprocessor/estimator/SHAP plumbing of the paper
# pipeline instead of re-implementing (and silently diverging from) it.

SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4-main", "S4-null")
DATASETS: tuple[str, ...] = ("dataset_a", "dataset_b")
MODEL_SEED = 42
CV_SPLITS = 5
RIDGE = 1e-6
SPLINE_N_KNOTS = 4
SPLINE_DEGREE = 3
M2OE_MAX_GROUP_ROWS = 128
M2OE_COMPARISON_MAX_GROUP_ROWS = 64
M2OE_CAP_LABEL = "M2OE-Group (cap={cap})"
# The original runtime table timed M2OE-Group at cap=128; on Dataset B those
# legacy rows take ~2070 s.  Rows below this threshold cannot be attributed to
# that configuration and must not be relabelled silently.
LEGACY_M2OE_CAP128_MIN_SECONDS = 1500.0
M2OE_RUNTIME_KEYS: tuple[str, ...] = ("dataset", "scenario", "seed", "n_jobs")
TOP_KS: tuple[int, ...] = (1, 3, 5)
METRIC_COLUMNS: tuple[str, ...] = ("mrr", "hr_at_1", "hr_at_3", "hr_at_5", "far_at_1", "far_at_5")

GADS_METHOD = "GADS"
GLM_MARGINAL_METHOD = "GLM-Int-Marginal"
GLM_JOINT_METHOD = "GLM-Int-Joint"
GLM_JOINT_L1_METHOD = "GLM-Int-Joint-L1-C1.0"
GLM_JOINT_L1_HALF_METHOD = "GLM-Int-Joint-L1-C0.5"
GAM_METHOD = "GAM-Int"
TREESHAP_ABS_METHOD = "TreeSHAP-Int-Abs"
TREESHAP_W1_METHOD = "TreeSHAP-Int-W1"
EXPL_DOMAIN_METHOD = "Expl-Domain-SHAP"
BENCHMARK_METHODS: tuple[str, ...] = (
    "GADS",
    "TreeSHAP-Int",
    "Expl-Domain-SHAP",
    "M2OE-Group",
    "XPE",
    "GLM-Int",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Benchmark-construction parameters for one paper dataset."""

    name: str
    model_params: dict[str, Any]
    noise_level: float = 0.2
    n_devices: int = 10
    distribution_shift: float = 3.0
    mechanism_scale: float = 1.0
    harmless_drift: float = 3.5
    group_seed: int = 2024
    physical_shift: float = 3.0
    mechanism_weight: float = 12.0
    harmless_shift: float = 4.0
    include_missing_indicators: bool = True


DATASET_SPECS: dict[str, DatasetSpec] = {
    "dataset_a": DatasetSpec(
        name="dataset_a",
        model_params={"max_depth": 6, "min_child_weight": 1},
    ),
    "dataset_b": DatasetSpec(
        name="dataset_b",
        model_params={"max_depth": 3, "min_child_weight": 5},
        include_missing_indicators=False,
    ),
}


# ---------------------------------------------------------------------------
# Small infrastructure helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas values and non-finite floats for JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON document with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _log(message: str) -> None:
    """Emit a flushed progress line to stderr."""
    print(message, file=sys.stderr, flush=True)


def _sample_rss(process: psutil.Process, stop_event: threading.Event, samples: list[int], interval: float) -> None:
    """Sample process RSS until ``stop_event`` is set."""
    while not stop_event.is_set():
        samples.append(int(process.memory_info().rss))
        stop_event.wait(interval)


class MemoryProbe:
    """Measure peak RSS delta and (optionally) Python allocation peaks."""

    def __init__(self, *, interval: float = 0.02, use_tracemalloc: bool = False) -> None:
        """Configure the probe with a sampling interval and tracemalloc toggle."""
        if interval <= 0:
            raise ValueError("sampling interval must be positive")
        self.interval = float(interval)
        self.use_tracemalloc = bool(use_tracemalloc)
        self._process = psutil.Process(os.getpid())
        self._samples: list[int] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0

    def start(self) -> None:
        """Record the baseline RSS and start sampling."""
        self._baseline = int(self._process.memory_info().rss)
        self._samples = [self._baseline]
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=_sample_rss,
            args=(self._process, self._stop_event, self._samples, self.interval),
            daemon=True,
        )
        self._thread.start()
        if self.use_tracemalloc:
            import tracemalloc

            tracemalloc.start()

    def stop(self) -> dict[str, float]:
        """Stop sampling and return baseline/peak/delta memory figures in MiB."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        peak = max(self._samples) if self._samples else self._baseline
        traced_peak = 0.0
        if self.use_tracemalloc:
            import tracemalloc

            _current, traced_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return {
            "rss_baseline_mb": self._baseline / 2**20,
            "rss_peak_mb": peak / 2**20,
            "rss_delta_mb": (peak - self._baseline) / 2**20,
            "tracemalloc_peak_mb": float(traced_peak) / 2**20,
        }


# ---------------------------------------------------------------------------
# Dataset construction and fold-level model fitting
# ---------------------------------------------------------------------------


def bundle_dataset_name(bundle: DatasetBundle) -> str:
    """Return the canonical dataset name stored in the bundle metadata."""
    value = bundle.metadata.get("dataset", bundle.metadata.get("dataset_name"))
    if value is None:
        raise ValueError("bundle metadata does not record the dataset name")
    aliases = {
        "A": "dataset_a",
        "Dataset A": "dataset_a",
        "dataset_a": "dataset_a",
        "B": "dataset_b",
        "Dataset B": "dataset_b",
        "dataset_b": "dataset_b",
    }
    canonical = aliases.get(str(value))
    if canonical is None:
        raise ValueError(f"unsupported bundle dataset {value!r}")
    return canonical


def build_bundle(dataset: str, scenario: str, seed: int) -> DatasetBundle:
    """Generate the exact paper bundle for one (dataset, scenario, seed)."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {DATASETS}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario {scenario!r}; expected one of {SCENARIOS}")
    spec = DATASET_SPECS[dataset]
    if dataset == "dataset_a":
        return generate_dataset_bundle(
            scenario=scenario,
            n_samples=5000,
            seed=int(seed),
            noise_level=spec.noise_level,
            n_devices=spec.n_devices,
            distribution_shift=spec.distribution_shift,
            mechanism_scale=spec.mechanism_scale,
            harmless_drift=spec.harmless_drift,
        )
    bundle = generate_dataset_b(
        scenario=cast(Any, scenario),
        seed=int(seed),
        group_seed=spec.group_seed,
        raw_dir=None,
        physical_shift_magnitude=spec.physical_shift,
        mechanism_local_weight=spec.mechanism_weight,
        harmless_shift_magnitude=spec.harmless_shift,
        include_missing_indicators=spec.include_missing_indicators,
    )
    metadata = dict(bundle.metadata)
    metadata["dataset"] = "dataset_b"
    return DatasetBundle(
        data=bundle.data,
        scenario=bundle.scenario,
        seed=bundle.seed,
        group_column=bundle.group_column,
        target_column=bundle.target_column,
        process_features=bundle.process_features,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        metadata=metadata,
    )


def bundle_matrices(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the KPI design matrix, binary target and device groups."""
    feature_columns = [*bundle.process_features, bundle.group_column]
    X = bundle.data.loc[:, feature_columns].copy()
    y = cast(pd.Series, pd.to_numeric(bundle.data[bundle.target_column], errors="raise")).astype(int)
    groups = cast(pd.Series, bundle.data[bundle.group_column]).astype(str)
    if not X.index.equals(y.index) or not X.index.equals(groups.index):
        raise ValueError("bundle columns must share one index")
    return X, y, groups


@dataclass
class FoldFit:
    """One fitted KPI fold together with its held-out transformed matrix."""

    fold: int
    valid_index: np.ndarray
    preprocessor: Any
    estimator: Any
    x_valid: np.ndarray
    auc: float
    fit_seconds: float
    shap_seconds: float


@dataclass
class CrossFit:
    """Cross-fitted KPI models, OOF attributions and stage timings."""

    folds: list[FoldFit]
    shap_values: pd.DataFrame
    oof_predictions: np.ndarray
    fold_aucs: list[float]
    encoded_feature_names: tuple[str, ...]
    final_model: Pipeline | None
    kpi_training_seconds: float
    oof_shap_seconds: float
    final_fit_seconds: float

    @property
    def mean_auc(self) -> float:
        """Return the mean out-of-fold AUC."""
        return float(np.mean(self.fold_aucs))


def fit_cross_fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_params: dict[str, Any] | None,
    model_seed: int = MODEL_SEED,
    cv_splits: int = CV_SPLITS,
    include_final_model: bool = True,
) -> CrossFit:
    """Mirror ``fit_cross_fitted_model`` while keeping fold models and timings.

    The fold splitter, preprocessor, estimator seeds, class weighting and SHAP
    aggregation are exactly those of ``modeling.fit_cross_fitted_model``.  The
    self-check compares this function's OOF SHAP matrix with that function's
    output before any experiment runs.
    """
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_seed)
    shap_oof = np.full((len(X), X.shape[1]), np.nan, dtype=float)
    predictions = np.full(len(X), np.nan, dtype=float)
    folds: list[FoldFit] = []
    encoded_names: tuple[str, ...] | None = None
    training_seconds = 0.0
    shap_seconds = 0.0
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        preprocessor = _make_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)
        names = tuple(preprocessor.get_feature_names_out())
        if encoded_names is None:
            encoded_names = names
        elif names != encoded_names:
            raise ValueError("fold preprocessors produced inconsistent feature names")
        estimator = _make_estimator(
            "xgboost",
            random_state=model_seed + fold,
            scale_pos_weight=_class_weight(y_train),
            model_params=model_params,
        )
        start = time.perf_counter()
        estimator.fit(X_train_t, y_train)
        fit_seconds = time.perf_counter() - start
        training_seconds += fit_seconds
        predictions[valid_idx] = estimator.predict_proba(X_valid_t)[:, 1]
        start = time.perf_counter()
        shap_encoded = _extract_binary_shap(estimator, np.asarray(X_valid_t))
        aggregated = _aggregate_shap(shap_encoded, preprocessor, list(X.columns))
        shap_seconds_fold = time.perf_counter() - start
        shap_seconds += shap_seconds_fold
        shap_oof[valid_idx] = aggregated.to_numpy()
        folds.append(
            FoldFit(
                fold=fold,
                valid_index=np.asarray(valid_idx),
                preprocessor=preprocessor,
                estimator=estimator,
                x_valid=np.asarray(X_valid_t),
                auc=float(roc_auc_score(y.iloc[valid_idx], predictions[valid_idx])),
                fit_seconds=fit_seconds,
                shap_seconds=shap_seconds_fold,
            )
        )
    if np.isnan(predictions).any() or np.isnan(shap_oof).any():
        raise RuntimeError("cross-fitting did not produce complete OOF predictions/attributions")
    final_model: Pipeline | None = None
    final_fit_seconds = 0.0
    if include_final_model:
        start = time.perf_counter()
        final_preprocessor = _make_preprocessor(X)
        X_all_t = final_preprocessor.fit_transform(X)
        final_estimator = _make_estimator(
            "xgboost",
            random_state=model_seed,
            scale_pos_weight=_class_weight(y),
            model_params=model_params,
        )
        final_estimator.fit(X_all_t, y)
        final_fit_seconds = time.perf_counter() - start
        final_model = Pipeline([("preprocessor", final_preprocessor), ("estimator", final_estimator)])
    return CrossFit(
        folds=folds,
        shap_values=pd.DataFrame(shap_oof, index=X.index, columns=X.columns),
        oof_predictions=predictions,
        fold_aucs=[fold_fit.auc for fold_fit in folds],
        encoded_feature_names=tuple(encoded_names or ()),
        final_model=final_model,
        kpi_training_seconds=training_seconds,
        oof_shap_seconds=shap_seconds,
        final_fit_seconds=final_fit_seconds,
    )


def cross_fit_attribution(
    crossfit: CrossFit,
    X: pd.DataFrame,
    *,
    model_params: dict[str, Any] | None,
    model_seed: int = MODEL_SEED,
    cv_splits: int = CV_SPLITS,
) -> CrossFittedAttributions:
    """Wrap a :class:`CrossFit` in the pipeline's public attribution container."""
    del model_params
    if crossfit.final_model is None:
        raise ValueError("an attribution container requires the final fitted model")
    fold_metrics = pd.DataFrame(
        [
            {
                "fold": fold_fit.fold,
                "n_train": len(X) - len(fold_fit.valid_index),
                "n_valid": len(fold_fit.valid_index),
                "auc": fold_fit.auc,
            }
            for fold_fit in crossfit.folds
        ]
    )
    metadata = {
        "model_type": "xgboost",
        "model_seed": model_seed,
        "cv_seed": model_seed,
        "cv_splits": cv_splits,
        "mean_auc": crossfit.mean_auc,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "encoded_feature_names": list(crossfit.encoded_feature_names),
    }
    return CrossFittedAttributions(
        shap_values=crossfit.shap_values,
        oof_predictions=pd.Series(crossfit.oof_predictions, index=X.index, name="oof_prediction"),
        fold_metrics=fold_metrics,
        model=crossfit.final_model,
        feature_names=tuple(X.columns),
        encoded_feature_names=crossfit.encoded_feature_names,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Competitor baselines
# ---------------------------------------------------------------------------


def _ranking_rows(
    scores: dict[str, float], *, equipment: str, method: str, view: str = "process"
) -> list[dict[str, Any]]:
    """Convert one score dictionary into ranked long-form rows."""
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


def fit_logistic_irls(
    design: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = RIDGE,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ridge-penalised logistic regression by Newton/IRLS.

    Returns the coefficient vector and Wald standard errors
    ``sqrt(diag(inv(X' W X + ridge I)))``.  The intercept is not penalised; the
    tiny ridge only stabilises collinear or constant columns.
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    if design.ndim != 2 or design.shape[0] != y.shape[0]:
        raise ValueError("design and y must align")
    n_features = design.shape[1]
    penalty = np.full(n_features, float(ridge))
    penalty[0] = 0.0
    beta = np.zeros(n_features, dtype=float)
    hessian = np.eye(n_features, dtype=float)
    for _ in range(max_iter):
        probabilities = expit(design @ beta)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-12, None)
        hessian = design.T @ (design * weights[:, None]) + np.diag(penalty)
        gradient = design.T @ (y - probabilities) - penalty * beta
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    inverse = np.linalg.inv(hessian)
    standard_errors = np.sqrt(np.clip(np.diag(inverse), 0.0, None))
    return beta, standard_errors


def one_vs_normal_pool_mask(
    group_values: np.ndarray,
    device: str,
    normal_devices: Sequence[str],
) -> np.ndarray:
    """Return the rows of ``device`` plus the designated normal pool only.

    Multi-fault scenarios (``S3``) contain several abnormal devices; diagnosing
    ``device`` must never treat another abnormal device as a reference, so this
    mask is ``(EQP == g) | EQP in normal_pool``.  In single-fault scenarios the
    mask equals every row, because there is no other abnormal device to drop.
    """
    normal_keys = [str(entry) for entry in normal_devices]
    mask = (group_values == str(device)) | np.isin(group_values, normal_keys)
    if not np.any(mask):
        raise ValueError(f"one-vs-normal-pool sample for {device!r} is empty")
    return np.asarray(mask, dtype=bool)


def standardised_process_matrix(X_process: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return z-scored process values plus the zero-variance column mask."""
    values = np.asarray(X_process.to_numpy(dtype=float), dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    constant = stds <= 0
    safe = np.where(constant, 1.0, stds)
    return (values - means) / safe, constant


def rank_glm_int_marginal(
    X_process: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank features by the |Wald z| of the feature x device interaction.

    For every process feature ``x_j`` and abnormal device ``g`` the model
    ``y ~ 1 + x_j + I(EQP == g) + x_j * I(EQP == g)`` is fitted on device ``g``'s
    rows plus the designated normal pool, with a 1e-6 ridge; other abnormal
    devices of a multi-fault scenario (``S3``) are excluded so the reference
    group is the normal pool alone (Section 3.1 / 4.1.2 protocol).  Constant
    features are skipped with score 0.
    """
    y_values = np.asarray(y.to_numpy(dtype=int), dtype=float)
    group_values = groups.astype(str).to_numpy()
    feature_names = list(X_process.columns)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for device in abnormal_devices:
        selected = one_vs_normal_pool_mask(group_values, device, normal_devices)
        z_values, constant = standardised_process_matrix(X_process.loc[selected])
        indicator = (group_values[selected] == str(device)).astype(float)[:, None]
        target = y_values[selected]
        scores: dict[str, float] = {}
        z_stats: list[float] = []
        for feature_position, feature in enumerate(feature_names):
            if constant[feature_position]:
                scores[feature] = 0.0
                z_stats.append(0.0)
                continue
            column = z_values[:, feature_position][:, None]
            design = np.column_stack([np.ones(len(target)), column, indicator, column * indicator])
            beta, standard_errors = fit_logistic_irls(design, target)
            interaction_beta = float(beta[3])
            standard_error = float(standard_errors[3])
            if standard_error <= 0 or not np.isfinite(standard_error):
                score = 0.0
            else:
                score = abs(interaction_beta / standard_error)
            scores[feature] = float(score) if np.isfinite(score) else 0.0
            z_stats.append(scores[feature])
        z_array = np.asarray(z_stats, dtype=float)
        diagnostics[str(device)] = {
            "n_rows": int(selected.sum()),
            "n_features": len(z_array),
            "median_abs_z": float(np.median(z_array)),
            "max_abs_z": float(np.max(z_array)),
            "n_abs_z_gt_1.96": int(np.sum(z_array > 1.96)),
            "n_abs_z_gt_3": int(np.sum(z_array > 3.0)),
        }
        rows.extend(_ranking_rows(scores, equipment=str(device), method=GLM_MARGINAL_METHOD))
    return rows, diagnostics


def rank_glm_int_joint(
    X_process: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
    *,
    penalty: Literal["l1", "l2"],
    c_value: float,
    solver: str,
    max_iter: int,
    tol: float,
    method_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank features by |beta_{x_j:g}| in a joint penalised interaction GLM.

    Every device-specific model is fitted on ``(EQP == g) | EQP in normal_pool``
    only, so other abnormal devices of ``S3`` never enter the reference group;
    the feature scaler is fitted on that same subset.
    """
    if penalty not in {"l1", "l2"}:
        raise ValueError(f"unsupported penalty {penalty!r}")
    y_values = np.asarray(y.to_numpy(dtype=int), dtype=int)
    group_values = groups.astype(str).to_numpy()
    feature_names = list(X_process.columns)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for device in abnormal_devices:
        selected = one_vs_normal_pool_mask(group_values, device, normal_devices)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.asarray(X_process.loc[selected].to_numpy(dtype=float), dtype=float))
        indicator = (group_values[selected] == str(device)).astype(float)[:, None]
        design = np.column_stack([scaled, indicator, scaled * indicator])
        model = LogisticRegression(
            C=float(c_value),
            l1_ratio=0.0 if penalty == "l2" else 1.0,
            solver=solver,
            max_iter=int(max_iter),
            tol=float(tol),
            fit_intercept=True,
        )
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
        rows.extend(_ranking_rows(scores, equipment=str(device), method=method_name))
    return rows, diagnostics


def binomial_deviance(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Return the binomial deviance ``-2 log-likelihood`` (saturated term dropped)."""
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-2.0 * np.sum(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def rank_gam_int(
    X_process: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank features by the additive-spline interaction deviance drop.

    For each feature the script compares ``y ~ s(x_j) + I(g)`` with
    ``y ~ s(x_j) + I(g) + s(x_j):I(g)`` using a cubic B-spline basis
    (4 knots, degree 3) and reports ``Delta D_j = D_null - D_interaction``.
    Both models are fitted on device ``g``'s rows plus the designated normal
    pool, so other abnormal devices of ``S3`` are excluded.
    """
    y_values = np.asarray(y.to_numpy(dtype=int), dtype=float)
    group_values = groups.astype(str).to_numpy()
    feature_names = list(X_process.columns)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for device in abnormal_devices:
        selected = one_vs_normal_pool_mask(group_values, device, normal_devices)
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
            null_model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
            alt_model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
            null_model.fit(additive, target)
            alt_model.fit(interaction, target)
            null_deviance = binomial_deviance(null_model.predict_proba(additive)[:, 1], target)
            alt_deviance = binomial_deviance(alt_model.predict_proba(interaction)[:, 1], target)
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
    return rows, diagnostics


def eqp_category_index(preprocessor: Any, group_column: str) -> dict[str, int]:
    """Map every EQP category to its one-hot column index in the design matrix."""
    names = list(preprocessor.get_feature_names_out())
    encoder = preprocessor.named_transformers_["categorical"]
    if not isinstance(encoder, OneHotEncoder):
        raise TypeError("expected a OneHotEncoder for the group column")
    if len(encoder.categories_) != 1:
        raise ValueError("exactly one categorical column (EQP) is supported")
    columns: dict[str, int] = {}
    encoded_names = list(encoder.get_feature_names_out([group_column]))
    for category, encoded_name in zip(encoder.categories_[0], encoded_names, strict=True):
        columns[str(category)] = names.index(encoded_name)
    return columns


def interaction_block(
    fold_fit: FoldFit,
    process_features: Sequence[str],
    group_column: str,
    interaction_threads: int,
    row_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return the held-out ``(rows, process, EQP-category)`` interaction block.

    ``row_positions`` restricts the expensive interaction computation to a
    subset of the fold's validation rows (positions into ``fold_fit.x_valid``).
    """
    encoded_names = list(fold_fit.preprocessor.get_feature_names_out())
    mapping = _build_encoded_to_original_mapping(fold_fit.preprocessor)
    process_positions = []
    for feature in process_features:
        positions = [index for index, name in enumerate(encoded_names) if mapping[name] == feature]
        if len(positions) != 1:
            raise ValueError(f"expected exactly one encoded column for process feature {feature!r}")
        process_positions.append(positions[0])
    category_index = eqp_category_index(fold_fit.preprocessor, group_column)
    eqp_positions = [category_index[category] for category in sorted(category_index)]
    booster = fold_fit.estimator.get_booster()
    booster.set_param({"nthread": str(int(interaction_threads))})
    matrix = fold_fit.x_valid if row_positions is None else fold_fit.x_valid[row_positions]
    dmatrix = xgboost.DMatrix(np.ascontiguousarray(matrix))
    phi = np.asarray(booster.predict(dmatrix, pred_interactions=True, validate_features=False), dtype=float)
    if phi.ndim != 3 or phi.shape[1] != phi.shape[2] or phi.shape[1] != fold_fit.x_valid.shape[1] + 1:
        raise RuntimeError(f"unexpected pred_interactions shape: {phi.shape}")
    sub = phi[:, :-1, eqp_positions]
    block = np.ascontiguousarray(sub[:, process_positions, :])
    del phi, sub, dmatrix
    return block


def sample_stratified_rows(strata: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Sample ``fraction`` of the rows proportionally within each stratum.

    At least one row per stratum is always kept so that no device/label cell
    disappears from the reference distribution. Returns sorted positions.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("row fraction must be in (0, 1]")
    if fraction == 1.0:
        return np.arange(len(strata))
    _, inverse = np.unique(strata, return_inverse=True)
    chosen: list[np.ndarray] = []
    for group in range(int(inverse.max()) + 1):
        positions = np.flatnonzero(inverse == group)
        n_keep = max(1, round(fraction * len(positions)))
        if n_keep >= len(positions):
            chosen.append(positions)
        else:
            chosen.append(np.sort(rng.choice(positions, size=n_keep, replace=False)))
    return np.sort(np.concatenate(chosen))


def interaction_row_selection(
    valid_groups: np.ndarray,
    valid_labels: np.ndarray,
    *,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
    normal_row_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep every abnormal row plus a stratified sample of normal-pool rows."""
    abnormal_keys = [str(device) for device in abnormal_devices]
    normal_keys = [str(device) for device in normal_devices]
    is_abnormal = np.isin(valid_groups, abnormal_keys)
    is_normal = np.isin(valid_groups, normal_keys)
    if not np.all(is_abnormal | is_normal):
        raise ValueError("fold validation rows contain devices outside the abnormal/normal pools")
    abnormal_positions = np.flatnonzero(is_abnormal)
    normal_positions = np.flatnonzero(is_normal)
    normal_strata = np.char.add(
        np.char.add(valid_groups[normal_positions], "|"),
        valid_labels[normal_positions].astype(str),
    )
    sampled_normal = normal_positions[sample_stratified_rows(normal_strata, normal_row_fraction, rng)]
    return np.sort(np.concatenate([abnormal_positions, sampled_normal]))


def rank_treeshap_interaction(
    crossfit: CrossFit,
    X: pd.DataFrame,
    groups: pd.Series,
    y: pd.Series,
    *,
    process_features: Sequence[str],
    group_column: str,
    abnormal_devices: Sequence[str],
    normal_devices: Sequence[str],
    interaction_threads: int,
    normal_row_fraction: float = 1.0,
    sampling_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Score process features by OOF TreeSHAP interactions with each EQP dummy.

    Every abnormal row is always scored.  ``normal_row_fraction`` optionally
    subsamples the normal-pool rows (device x KPI-label stratified, seeded by
    ``sampling_seed`` and the fold index) before the expensive interaction
    computation; the default 1.0 keeps the complete OOF sample.
    """
    all_devices = {str(device) for device in abnormal_devices} | {str(device) for device in normal_devices}
    n_rows = len(X)
    n_process = len(process_features)
    eqp_sum = np.full((n_rows, n_process), np.nan, dtype=float)
    scored_mask = np.zeros(n_rows, dtype=bool)
    abs_sum = {str(device): np.zeros(n_process, dtype=float) for device in abnormal_devices}
    abs_count = dict.fromkeys((str(device) for device in abnormal_devices), 0)
    group_values = groups.astype(str).to_numpy()
    label_values = np.asarray(y.to_numpy(dtype=int), dtype=int)
    normal_mask = np.isin(group_values, [str(device) for device in normal_devices])
    interaction_seconds = 0.0
    n_scored_rows = 0
    for fold_fit in crossfit.folds:
        category_index = eqp_category_index(fold_fit.preprocessor, group_column)
        if set(category_index) != all_devices:
            raise ValueError("fold EQP categories differ from the bundle's device pools")
        ordered_categories = sorted(category_index)
        valid_groups = group_values[fold_fit.valid_index]
        valid_labels = label_values[fold_fit.valid_index]
        rng = np.random.default_rng([int(sampling_seed), int(fold_fit.fold), 2024])
        row_positions = interaction_row_selection(
            valid_groups,
            valid_labels,
            abnormal_devices=abnormal_devices,
            normal_devices=normal_devices,
            normal_row_fraction=float(normal_row_fraction),
            rng=rng,
        )
        start = time.perf_counter()
        block = interaction_block(
            fold_fit,
            process_features,
            group_column,
            interaction_threads,
            row_positions=row_positions,
        )
        interaction_seconds += time.perf_counter() - start
        sampled_index = fold_fit.valid_index[row_positions]
        eqp_sum[sampled_index] = block.sum(axis=2)
        scored_mask[sampled_index] = True
        n_scored_rows += len(sampled_index)
        sampled_groups = valid_groups[row_positions]
        sampled_normal = normal_mask[sampled_index]
        for device in abnormal_devices:
            device_key = str(device)
            column = ordered_categories.index(device_key)
            selected = (sampled_groups == device_key) | sampled_normal
            if not np.any(selected):
                raise ValueError(f"no device/normal rows for {device_key!r} in fold {fold_fit.fold}")
            abs_sum[device_key] += np.abs(block[selected, :, column]).sum(axis=0)
            abs_count[device_key] += int(np.sum(selected))
        del block
    if np.isnan(eqp_sum[scored_mask]).any():
        raise RuntimeError("interaction aggregation left unassigned sampled rows")
    process_values = list(process_features)
    abs_rows: list[dict[str, Any]] = []
    w1_rows: list[dict[str, Any]] = []
    normal_values = eqp_sum[normal_mask & scored_mask]
    for device in abnormal_devices:
        device_key = str(device)
        if abs_count[device_key] == 0:
            raise RuntimeError(f"no rows accumulated for device {device_key!r}")
        abs_scores = {
            feature: float(abs_sum[device_key][position] / abs_count[device_key])
            for position, feature in enumerate(process_values)
        }
        abs_rows.extend(_ranking_rows(abs_scores, equipment=device_key, method=TREESHAP_ABS_METHOD))
        device_values = eqp_sum[(group_values == device_key) & scored_mask]
        w1_scores = {
            feature: float(wasserstein_distance(device_values[:, position], normal_values[:, position]))
            for position, feature in enumerate(process_values)
        }
        w1_rows.extend(_ranking_rows(w1_scores, equipment=device_key, method=TREESHAP_W1_METHOD))
    diagnostics = {
        "n_eqp_categories": len(all_devices),
        "interaction_compute_seconds": interaction_seconds,
        "n_rows": int(n_rows),
        "n_scored_rows": int(n_scored_rows),
        "normal_row_fraction": float(normal_row_fraction),
    }
    return abs_rows, w1_rows, diagnostics


# ---------------------------------------------------------------------------
# One comparison run (all methods for one dataset/scenario/seed)
# ---------------------------------------------------------------------------


def compute_comparison_rankings(
    bundle: DatasetBundle,
    *,
    interaction_threads: int,
    interaction_row_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute every Pack-2 competitor ranking plus the GADS reference."""
    X, y, groups = bundle_matrices(bundle)
    dataset = bundle_dataset_name(bundle)
    model_params = DATASET_SPECS[dataset].model_params
    timings: dict[str, float] = {}
    start = time.perf_counter()
    crossfit = fit_cross_fit(X, y, model_params=model_params, include_final_model=False)
    timings["kpi_cross_fit"] = time.perf_counter() - start
    shap_values = crossfit.shap_values
    process_frame = X.loc[:, list(bundle.process_features)]

    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}

    start = time.perf_counter()
    gads = rank_gads_features(
        shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
    )
    timings["GADS"] = time.perf_counter() - start
    frames.append(gads)

    start = time.perf_counter()
    marginal_rows, marginal_diagnostics = rank_glm_int_marginal(
        process_frame, y, groups, bundle.abnormal_devices, bundle.normal_devices
    )
    timings[GLM_MARGINAL_METHOD] = time.perf_counter() - start
    diagnostics[GLM_MARGINAL_METHOD] = marginal_diagnostics
    frames.append(pd.DataFrame(marginal_rows))

    start = time.perf_counter()
    joint_rows, joint_diagnostics = rank_glm_int_joint(
        process_frame,
        y,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        penalty="l2",
        c_value=1.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        method_name=GLM_JOINT_METHOD,
    )
    timings[GLM_JOINT_METHOD] = time.perf_counter() - start
    diagnostics[GLM_JOINT_METHOD] = joint_diagnostics
    frames.append(pd.DataFrame(joint_rows))

    start = time.perf_counter()
    joint_l1_rows, joint_l1_diagnostics = rank_glm_int_joint(
        process_frame,
        y,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        penalty="l1",
        c_value=1.0,
        solver="saga",
        max_iter=500,
        tol=1e-3,
        method_name=GLM_JOINT_L1_METHOD,
    )
    timings[GLM_JOINT_L1_METHOD] = time.perf_counter() - start
    diagnostics[GLM_JOINT_L1_METHOD] = joint_l1_diagnostics
    frames.append(pd.DataFrame(joint_l1_rows))

    start = time.perf_counter()
    gam_rows, gam_diagnostics = rank_gam_int(process_frame, y, groups, bundle.abnormal_devices, bundle.normal_devices)
    timings[GAM_METHOD] = time.perf_counter() - start
    diagnostics[GAM_METHOD] = gam_diagnostics
    frames.append(pd.DataFrame(gam_rows))

    start = time.perf_counter()
    abs_rows, w1_rows, interaction_diagnostics = rank_treeshap_interaction(
        crossfit,
        X,
        groups,
        y,
        process_features=bundle.process_features,
        group_column=bundle.group_column,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        interaction_threads=interaction_threads,
        normal_row_fraction=interaction_row_fraction,
        sampling_seed=int(bundle.seed),
    )
    timings["TreeSHAP-Int"] = time.perf_counter() - start
    diagnostics["TreeSHAP-Int"] = interaction_diagnostics
    frames.append(pd.DataFrame(abs_rows))
    frames.append(pd.DataFrame(w1_rows))

    start = time.perf_counter()
    domain = rank_domain_shap(
        shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        include_eqp_in_classifier=False,
        model_type="xgboost",
        model_seed=MODEL_SEED,
        cv_seed=None,
        cv_splits=CV_SPLITS,
        model_params=model_params,
    )
    domain["method"] = EXPL_DOMAIN_METHOD
    timings[EXPL_DOMAIN_METHOD] = time.perf_counter() - start
    frames.append(domain)

    rankings = pd.concat(frames, ignore_index=True)
    rankings.insert(0, "dataset", dataset)
    rankings.insert(1, "scenario", bundle.scenario)
    rankings.insert(2, "seed", bundle.seed)
    rankings["is_root_cause"] = [
        feature in set(bundle.root_causes.get(str(device), ()))
        for device, feature in zip(rankings["equipment"], rankings["feature"], strict=True)
    ]
    rankings["is_harmless"] = rankings["feature"].isin(bundle.harmless_features)
    metrics = evaluate_rankings(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=TOP_KS,
    )
    equipment_metrics = evaluate_rankings_per_equipment(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=TOP_KS,
    )
    context = {"dataset": dataset, "scenario": bundle.scenario, "seed": bundle.seed, "kpi_auc": crossfit.mean_auc}
    for column, value in context.items():
        metrics.insert(0, column, value)
        equipment_metrics.insert(0, column, value)
    metadata = {
        **context,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
        "n_eqp_categories": int(len(bundle.normal_devices) + len(bundle.abnormal_devices)),
        "fold_aucs": crossfit.fold_aucs,
        "timings_seconds": timings,
        "kpi_training_seconds": crossfit.kpi_training_seconds,
        "oof_shap_seconds": crossfit.oof_shap_seconds,
        "diagnostics": diagnostics,
        "metrics": metrics.to_dict(orient="records"),
        "equipment_metrics": equipment_metrics.to_dict(orient="records"),
        "model_params": model_params,
        "interaction_threads": int(interaction_threads),
        "interaction_row_fraction": float(interaction_row_fraction),
    }
    return rankings, metadata


def run_comparison_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: run one (dataset, scenario, seed) comparison."""
    dataset = str(task["dataset"])
    scenario = str(task["scenario"])
    seed = int(task["seed"])
    runs_dir = Path(task["runs_dir"])
    run_id = f"{dataset}_{scenario}_seed{seed}"
    parquet_path = runs_dir / f"{run_id}.parquet"
    meta_path = runs_dir / f"{run_id}.json"
    if task["resume"] and parquet_path.is_file() and meta_path.is_file():
        return {"run_id": run_id, "status": "skipped"}
    start = time.perf_counter()
    try:
        bundle = build_bundle(dataset, scenario, seed)
        rankings, metadata = compute_comparison_rankings(
            bundle,
            interaction_threads=int(task["interaction_threads"]),
            interaction_row_fraction=float(task["interaction_row_fraction"]),
        )
        runs_dir.mkdir(parents=True, exist_ok=True)
        rankings.to_parquet(parquet_path, index=False)
        metadata["total_seconds"] = time.perf_counter() - start
        _write_json(meta_path, metadata)
    except Exception as error:
        return {
            "run_id": run_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    return {"run_id": run_id, "status": "completed", "seconds": metadata["total_seconds"]}


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def self_check_dataset(dataset: str, *, interaction_threads: int) -> dict[str, Any]:
    """Verify fold-loop equivalence with the paper pipeline and the SHAP path."""
    bundle = build_bundle(dataset, "S1", 0)
    X, y, _groups = bundle_matrices(bundle)
    params = DATASET_SPECS[dataset].model_params
    reference = fit_cross_fitted_model(X, y, model_seed=MODEL_SEED, cv_splits=CV_SPLITS, model_params=params)
    candidate = fit_cross_fit(X, y, model_params=params, include_final_model=True)
    shap_diff = float(np.max(np.abs(reference.shap_values.to_numpy() - candidate.shap_values.to_numpy())))
    auc_diff = float(abs(reference.mean_auc - candidate.mean_auc))
    if shap_diff > 1e-8 or auc_diff > 1e-12:
        raise RuntimeError(
            f"{dataset}: fold loop diverges from fit_cross_fitted_model "
            f"(max_shap_diff={shap_diff!r}, auc_diff={auc_diff!r})"
        )
    fold_fit = candidate.folds[0]
    process_features = list(bundle.process_features)
    rows = min(20, fold_fit.x_valid.shape[0])
    block = interaction_block(
        fold_fit,
        process_features,
        bundle.group_column,
        interaction_threads,
        row_positions=np.arange(rows),
    )
    explainer = shap.TreeExplainer(fold_fit.estimator)
    reference_interactions = np.asarray(explainer.shap_interaction_values(fold_fit.x_valid[:rows]), dtype=float)
    category_index = eqp_category_index(fold_fit.preprocessor, bundle.group_column)
    encoded_names = list(fold_fit.preprocessor.get_feature_names_out())
    mapping = _build_encoded_to_original_mapping(fold_fit.preprocessor)
    process_positions = [
        next(index for index, name in enumerate(encoded_names) if mapping[name] == feature)
        for feature in process_features
    ]
    eqp_positions = [category_index[category] for category in sorted(category_index)]
    expected = reference_interactions[:, :, eqp_positions][:, process_positions, :]
    interaction_diff = float(np.max(np.abs(expected - block[:rows])))
    if interaction_diff > 1e-8:
        raise RuntimeError(f"{dataset}: interaction block diverges from shap.TreeExplainer ({interaction_diff!r})")
    return {
        "dataset": dataset,
        "max_abs_oof_shap_diff": shap_diff,
        "abs_mean_auc_diff": auc_diff,
        "max_abs_interaction_diff": interaction_diff,
        "n_rows": len(X),
        "n_process_features": len(process_features),
        "mean_auc": candidate.mean_auc,
    }


# ---------------------------------------------------------------------------
# Runtime / memory benchmark
# ---------------------------------------------------------------------------


def benchmark_gads(bundle: DatasetBundle, *, n_jobs: int, memory_probe: MemoryProbe) -> dict[str, Any]:
    """Run one GADS diagnosis with stage-level timings and memory sampling."""
    X, y, groups = bundle_matrices(bundle)
    params = {**DATASET_SPECS[bundle_dataset_name(bundle)].model_params, "n_jobs": int(n_jobs)}
    memory_probe.start()
    total_start = time.perf_counter()
    crossfit = fit_cross_fit(X, y, model_params=params, include_final_model=True)
    ranking_start = time.perf_counter()
    rank_gads_features(
        crossfit.shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
    )
    ranking_seconds = time.perf_counter() - ranking_start
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": "GADS",
        "n_jobs": int(n_jobs),
        "stages": {
            "kpi_5fold_training": crossfit.kpi_training_seconds,
            "oof_treeshap": crossfit.oof_shap_seconds,
            "final_model_fit": crossfit.final_fit_seconds,
            "wasserstein_ranking": ranking_seconds,
            "total": total_seconds,
        },
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
    }


def benchmark_treeshap_interaction(
    bundle: DatasetBundle,
    *,
    interaction_row_fraction: float,
    memory_probe: MemoryProbe,
) -> dict[str, Any]:
    """Run one TreeSHAP-Int diagnosis with stage timings."""
    X, y, groups = bundle_matrices(bundle)
    params = DATASET_SPECS[bundle_dataset_name(bundle)].model_params
    memory_probe.start()
    total_start = time.perf_counter()
    crossfit = fit_cross_fit(X, y, model_params=params, include_final_model=False)
    interaction_start = time.perf_counter()
    abs_rows, w1_rows, diagnostics = rank_treeshap_interaction(
        crossfit,
        X,
        groups,
        y,
        process_features=bundle.process_features,
        group_column=bundle.group_column,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        interaction_threads=1,
        normal_row_fraction=interaction_row_fraction,
        sampling_seed=int(bundle.seed),
    )
    interaction_seconds = time.perf_counter() - interaction_start
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": "TreeSHAP-Int",
        "n_jobs": 1,
        "stages": {
            "kpi_5fold_training": crossfit.kpi_training_seconds,
            "oof_treeshap": crossfit.oof_shap_seconds,
            "interaction_and_ranking": interaction_seconds,
            "total": total_seconds,
        },
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
        "n_abs_rows": len(abs_rows),
        "n_w1_rows": len(w1_rows),
        "diagnostics": diagnostics,
    }


def benchmark_expl_domain(bundle: DatasetBundle, *, memory_probe: MemoryProbe) -> dict[str, Any]:
    """Run one Expl-Domain-SHAP diagnosis with stage timings."""
    X, y, groups = bundle_matrices(bundle)
    params = DATASET_SPECS[bundle_dataset_name(bundle)].model_params
    memory_probe.start()
    total_start = time.perf_counter()
    crossfit = fit_cross_fit(X, y, model_params=params, include_final_model=False)
    classifier_start = time.perf_counter()
    rank_domain_shap(
        crossfit.shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        include_eqp_in_classifier=False,
        model_type="xgboost",
        model_seed=MODEL_SEED,
        cv_seed=None,
        cv_splits=CV_SPLITS,
        model_params=params,
    )
    classifier_seconds = time.perf_counter() - classifier_start
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": "Expl-Domain-SHAP",
        "n_jobs": 1,
        "stages": {
            "kpi_5fold_training": crossfit.kpi_training_seconds,
            "oof_treeshap": crossfit.oof_shap_seconds,
            "domain_classifier_and_ranking": classifier_seconds,
            "total": total_seconds,
        },
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
    }


def benchmark_m2oe(
    bundle: DatasetBundle,
    *,
    memory_probe: MemoryProbe,
    max_group_rows: int = M2OE_COMPARISON_MAX_GROUP_ROWS,
) -> dict[str, Any]:
    """Run one M2OE-Group diagnosis (no KPI model required)."""
    X, _y, groups = bundle_matrices(bundle)
    memory_probe.start()
    total_start = time.perf_counter()
    rank_m2oe_group(
        X,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        seed=42,
        epochs=30,
        n_neighbors=30,
        batch_size=16,
        max_group_rows=int(max_group_rows),
    )
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": M2OE_CAP_LABEL.format(cap=int(max_group_rows)),
        "m2oe_max_group_rows": int(max_group_rows),
        "n_jobs": 1,
        "stages": {"m2oe_fit_and_ranking": total_seconds, "total": total_seconds},
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
    }


def benchmark_xpe(bundle: DatasetBundle, *, memory_probe: MemoryProbe) -> dict[str, Any]:
    """Run one XPE diagnosis including its monitored KPI model."""
    X, y, groups = bundle_matrices(bundle)
    params = DATASET_SPECS[bundle_dataset_name(bundle)].model_params
    memory_probe.start()
    total_start = time.perf_counter()
    crossfit = fit_cross_fit(X, y, model_params=params, include_final_model=True)
    attribution = cross_fit_attribution(crossfit, X, model_params=params)
    xpe_start = time.perf_counter()
    rank_xpe(
        X,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        y=y,
        attribution=attribution,
        process_features=bundle.process_features,
        view="process",
        permutations=8,
        seed=42,
        max_target_rows=None,
        max_source_rows=None,
    )
    xpe_seconds = time.perf_counter() - xpe_start
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": "XPE",
        "n_jobs": 1,
        "stages": {
            "kpi_5fold_training": crossfit.kpi_training_seconds,
            "oof_treeshap": crossfit.oof_shap_seconds,
            "final_model_fit": crossfit.final_fit_seconds,
            "xpe_transport_and_ranking": xpe_seconds,
            "total": total_seconds,
        },
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
    }


def benchmark_glm_joint(bundle: DatasetBundle, *, memory_probe: MemoryProbe) -> dict[str, Any]:
    """Run one joint interaction GLM diagnosis."""
    X, y, groups = bundle_matrices(bundle)
    memory_probe.start()
    total_start = time.perf_counter()
    rank_glm_int_joint(
        X.loc[:, list(bundle.process_features)],
        y,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        penalty="l2",
        c_value=1.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        method_name=GLM_JOINT_METHOD,
    )
    total_seconds = time.perf_counter() - total_start
    memory = memory_probe.stop()
    return {
        "method": "GLM-Int",
        "n_jobs": 1,
        "stages": {"glm_fit_and_ranking": total_seconds, "total": total_seconds},
        "memory": memory,
        "n_rows": len(X),
        "n_process_features": len(bundle.process_features),
    }


BENCHMARK_DISPATCH: dict[str, Any] = {
    "GADS": benchmark_gads,
    "TreeSHAP-Int": benchmark_treeshap_interaction,
    "Expl-Domain-SHAP": benchmark_expl_domain,
    "M2OE-Group": benchmark_m2oe,
    "XPE": benchmark_xpe,
    "GLM-Int": benchmark_glm_joint,
}


def run_runtime_benchmark_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one benchmark method configuration for one (dataset, seed, scenario).

    ``M2OE-Group`` returns its own explicit ``M2OE-Group (cap=N)`` label and the
    matching ``m2oe_max_group_rows`` value; every other method reports ``None``.
    The cap is read from the task so a caller can benchmark accuracy-style
    (``64``) and legacy (``128``) configurations without relabelling later.
    """
    dataset = str(task["dataset"])
    scenario = str(task["scenario"])
    seed = int(task["seed"])
    method = str(task["method"])
    if method not in BENCHMARK_DISPATCH:
        raise ValueError(f"unknown benchmark method {method!r}")
    bundle = build_bundle(dataset, scenario, seed)
    probe = MemoryProbe(interval=0.02, use_tracemalloc=bool(task["use_tracemalloc"]))
    load_before = float(os.getloadavg()[0])
    if method == "GADS":
        result = benchmark_gads(bundle, n_jobs=int(task["gads_jobs"]), memory_probe=probe)
    elif method == "TreeSHAP-Int":
        result = benchmark_treeshap_interaction(
            bundle,
            interaction_row_fraction=float(task["interaction_row_fraction"]),
            memory_probe=probe,
        )
    elif method == "M2OE-Group":
        result = benchmark_m2oe(
            bundle,
            memory_probe=probe,
            max_group_rows=int(task.get("m2oe_max_group_rows", M2OE_COMPARISON_MAX_GROUP_ROWS)),
        )
    else:
        result = BENCHMARK_DISPATCH[method](bundle, memory_probe=probe)
    result["m2oe_max_group_rows"] = result.get("m2oe_max_group_rows") if method == "M2OE-Group" else None
    if method == "M2OE-Group" and result["m2oe_max_group_rows"] is None:
        raise RuntimeError("the M2OE-Group benchmark did not report its max_group_rows cap")
    result.update(
        {
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "load_1m_before": load_before,
            "load_1m_after": float(os.getloadavg()[0]),
        }
    )
    _log(f"[benchmark] {dataset} seed={seed} {method} n_jobs={result['n_jobs']} total={result['stages']['total']:.2f}s")
    return result


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def load_comparison_outputs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-run metrics, equipment metrics and metadata records."""
    metric_records: list[dict[str, Any]] = []
    equipment_records: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata_records.append(
            {
                key: payload.get(key)
                for key in (
                    "dataset",
                    "scenario",
                    "seed",
                    "n_rows",
                    "n_process_features",
                    "kpi_auc",
                    "timings_seconds",
                    "kpi_training_seconds",
                    "oof_shap_seconds",
                    "diagnostics",
                    "model_params",
                    "interaction_threads",
                    "total_seconds",
                )
            }
        )
        metric_records.extend(payload.get("metrics", []))
        equipment_records.extend(payload.get("equipment_metrics", []))
    return pd.DataFrame(metric_records), pd.DataFrame(equipment_records), pd.DataFrame(metadata_records)


def summarize_comparison(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level metrics into mean/SD per dataset, scenario and method."""
    if per_seed.empty:
        raise ValueError("no comparison metrics were loaded")
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, method), frame in per_seed.groupby(["dataset", "scenario", "method"], dropna=False):
        record: dict[str, Any] = {
            "dataset": dataset,
            "scenario": scenario,
            "method": method,
            "n_seeds": int(frame["seed"].nunique()),
            "kpi_auc_mean": float(frame["kpi_auc"].mean()),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else float("nan")
            record[f"{metric}_n_defined"] = int(values.notna().sum())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["dataset", "scenario", "method"], kind="stable").reset_index(drop=True)


def summarize_pooled(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Pool root-cause metrics over S1-S4-main and false alarms over S4 scenarios."""
    root_scenarios = ("S1", "S2", "S3", "S4-main")
    far_scenarios = ("S4-main", "S4-null")
    rows: list[dict[str, Any]] = []
    for (dataset, method), frame in per_seed.groupby(["dataset", "method"], sort=False):
        root_frame = frame.loc[frame["scenario"].isin(root_scenarios)]
        far_frame = frame.loc[frame["scenario"].isin(far_scenarios)]
        record: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "n_runs_root": len(root_frame),
            "n_runs_far": len(far_frame),
        }
        for metric in ("mrr", "hr_at_1", "hr_at_3", "hr_at_5"):
            values = pd.to_numeric(root_frame[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
        for metric in ("far_at_1", "far_at_5"):
            values = pd.to_numeric(far_frame[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["dataset", "method"], kind="stable").reset_index(drop=True)


def observed_interaction_row_fraction(metadata: pd.DataFrame, fallback: float) -> tuple[float, int]:
    """Recover the TreeSHAP interaction subsample actually used by the saved runs.

    ``--summary-only`` rebuilds must not silently describe the runs with the CLI
    default when the stored run metadata records a different fraction.
    """
    if metadata.empty or "diagnostics" not in metadata.columns:
        return float(fallback), 0
    observed: list[float] = []
    for diagnostics in metadata["diagnostics"]:
        entry = (diagnostics or {}).get("TreeSHAP-Int")
        if isinstance(entry, dict) and "normal_row_fraction" in entry:
            observed.append(float(entry["normal_row_fraction"]))
    if not observed:
        return float(fallback), 0
    unique = sorted({round(value, 8) for value in observed})
    if len(unique) != 1:
        raise ValueError(f"stored runs disagree on the TreeSHAP interaction row fraction: {unique}")
    return unique[0], len(observed)


def summarize_diagnostics(metadata: pd.DataFrame) -> pd.DataFrame:
    """Summarize interaction-test diagnostics (S1 blind-spot evidence)."""
    rows: list[dict[str, Any]] = []
    for _, record in metadata.iterrows():
        diagnostics = record.get("diagnostics") or {}
        for method, per_device in diagnostics.items():
            if not isinstance(per_device, dict):
                continue
            for device, values in per_device.items():
                if not isinstance(values, dict):
                    continue
                rows.append(
                    {
                        "dataset": record["dataset"],
                        "scenario": record["scenario"],
                        "seed": int(record["seed"]),
                        "method": method,
                        "equipment": device,
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def _interaction_row_note(fraction: float) -> str:
    """Describe the TreeSHAP interaction row protocol used for the run."""
    if fraction >= 1.0:
        return (
            "Every held-out OOF row is scored (no row subsampling); the interaction "
            "computation is the exact full-sample TreeSHAP interaction matrix."
        )
    return (
        f"All abnormal-device rows are scored together with a seeded, device x KPI-label "
        f"stratified {fraction:.0%} sample of the normal-pool rows per fold; this keeps the "
        "O(rows x trees) interaction computation tractable on Dataset A and is applied "
        "identically to every seed and scenario."
    )


def _compact_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the compact, paper-facing columns of a summary table."""
    columns = [
        column
        for column in (
            "scenario",
            "method",
            "n_seeds",
            "mrr_mean",
            "hr_at_1_mean",
            "hr_at_3_mean",
            "hr_at_5_mean",
            "far_at_1_mean",
            "far_at_5_mean",
        )
        if column in frame.columns
    ]
    return frame.loc[:, columns]


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without external extensions."""
    if frame.empty:
        return "_(empty)_\n"
    return frame.to_markdown(index=False, floatfmt=".4f") + "\n"


def write_markdown_summary(
    output_dir: Path,
    *,
    comparison_summary: pd.DataFrame,
    pooled: pd.DataFrame,
    diagnostics_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame | None,
    manifest: dict[str, Any],
    notes: Sequence[str] = (),
) -> Path:
    """Render ``summary.md`` with the headline tables and protocol notes."""
    lines: list[str] = []
    lines.append("# BDCC revision Pack 2: requested competitors and runtime cost\n")
    lines.append("## What this pack contains\n")
    lines.append(
        "1. A 20-seed comparison of Reviewer 3's requested competitors against GADS on Dataset A "
        "(n=5000) and Dataset B (`include_missing_indicators=False`) across S1, S2, S3, S4-main "
        "and S4-null.\n"
    )
    lines.append(
        "2. A sequential, uncontended one-diagnosis wall-clock and memory benchmark on Dataset A "
        "(S2, n=5000) and Dataset B (S2), including the stage breakdown of GADS at `n_jobs=1` "
        "and `n_jobs=8`.\n"
    )
    lines.append("\n## Protocol\n")
    lines.append(
        "- GADS reference: exact paper protocol (`model_seed=42`, 5-fold OOF XGBoost, "
        "`max_depth=6, min_child_weight=1` on A; `max_depth=3, min_child_weight=5` on B).\n"
        "- `GLM-Int-Marginal`: per-feature logistic regression `y ~ 1 + x_j + I(EQP==g) + "
        "x_j*I(EQP==g)` with a 1e-6 ridge; features are ranked by the |Wald z| of the interaction "
        "term. Zero-variance features score 0.\n"
        "- `GLM-Int-Joint`: joint L2 (`C=1.0`, lbfgs) and joint L1 (`C=1.0` and `C=0.5`, saga) "
        "logistic regression on `[X_process, I(EQP==g), X_process*I(EQP==g)]`; ranking uses "
        "`|beta_{x_j:g}|`.\n"
        "- `GAM-Int`: cubic B-spline basis (4 knots, degree 3) with a device interaction; ranking "
        "uses the deviance drop `Delta D_j` between the additive and interaction models.\n"
        "- `GLM-Int-Marginal`, `GLM-Int-Joint` and `GAM-Int` follow the same one-versus-normal-pool "
        "protocol as every other method: for device `g` the fitted sample is "
        "`(EQP == g) | EQP in normal_pool`, so in the multi-fault scenario S3 the other two abnormal "
        "devices never enter the reference group `I(EQP != g)`. Single-fault scenarios (S1, S2, "
        "S4-main, S4-null) contain no other abnormal device, so their fitted sample is unchanged.\n"
        "- `TreeSHAP-Int-Abs` / `TreeSHAP-Int-W1`: OOF TreeSHAP interaction values from the same "
        "five KPI fold models. Abs scores the mean `|Phi_{i,j,EQP_g}|` over device-g and normal "
        "rows; W1 compares the per-row `sum_g' Phi_{i,j,EQP_g'}` distributions of device g and the "
        "normal pool. " + _interaction_row_note(float(manifest["args"]["interaction_row_fraction"])) + "\n"
        "- `Expl-Domain-SHAP`: one-vs-rest XGBoost equipment classifier trained on the KPI OOF "
        "SHAP matrix, ranked by mean |SHAP| (the pipeline's `rank_domain_shap` hyperparameters).\n"
        "- Metrics follow `metrics.evaluate_rankings`: MRR is the reciprocal rank of the best "
        "annotated root cause; HR@K is a root-cause hit in the top K; FAR@K is a harmless-feature "
        "hit in the top K. S4-null root-cause metrics and S1-S3 false-alarm metrics are undefined "
        "and stay NaN rather than being reported as zero.\n"
    )
    lines.append("\n## Dataset A: seed-level summary (20 seeds per cell)\n")
    lines.append(_markdown_table(_compact_table(comparison_summary.loc[comparison_summary["dataset"] == "dataset_a"])))
    lines.append("\n## Dataset B: seed-level summary (20 seeds per cell)\n")
    lines.append(_markdown_table(_compact_table(comparison_summary.loc[comparison_summary["dataset"] == "dataset_b"])))
    lines.append("\n## Pooled head-to-head\n")
    lines.append(_markdown_table(pooled))
    if not diagnostics_summary.empty:
        lines.append("\n## Interaction-test diagnostics (S1 blind-spot evidence)\n")
        s1 = diagnostics_summary.loc[diagnostics_summary["scenario"] == "S1"]
        keep = [
            column
            for column in (
                "dataset",
                "method",
                "median_abs_z",
                "max_abs_z",
                "n_abs_z_gt_1.96",
                "n_abs_z_gt_3",
                "max_abs_beta_interaction",
                "median_abs_beta_interaction",
                "n_nonzero_interaction",
                "median_deviance_drop",
                "max_deviance_drop",
                "n_drop_gt_3.84",
            )
            if column in s1.columns
        ]
        if not s1.empty:
            grouped = s1.loc[:, keep].groupby(["dataset", "method"], dropna=False).mean(numeric_only=True)
            lines.append(_markdown_table(grouped.reset_index()))
    if runtime_summary is not None and not runtime_summary.empty:
        lines.append("\n## Runtime and peak memory (sequential, one idle process)\n")
        lines.append(_markdown_table(runtime_summary.drop(columns=["m2oe_max_group_rows"], errors="ignore")))
        lines.append(
            "\nTimings are wall-clock seconds for one full diagnosis (all abnormal devices of S2). "
            "`total` includes the shared KPI pipeline for methods that need it; stage rows make the "
            "shared cost explicit. Memory is sampled at 20 ms during the method call: "
            "`peak_rss_delta_mb_mean` is the growth of resident memory over the process baseline and "
            "`peak_rss_mb_mean` is the absolute resident peak; the delta can be near zero when the "
            "allocator reuses pages freed by earlier methods, so the absolute peak is the safer "
            "cross-method number. The separate tracemalloc pass is reported for seed 0 only because "
            "tracemalloc distorts wall-clock timing. `TreeSHAP-Int` uses the same interaction-row "
            f"fraction as the comparison (`{manifest['args']['interaction_row_fraction']}`), so its "
            "interaction stage scales with that fraction while KPI training and OOF TreeSHAP use all "
            "rows. `M2OE-Group` rows are labelled with their `max_group_rows` cap: the accuracy table "
            f"uses `cap={manifest['args']['m2oe_max_group_rows_accuracy']}`, and the cap sweep reports "
            "the same cell at every measured cap so the runtime cost can be read at the table's setting.\n"
        )
    lines.append("\n## Reproducibility notes\n")
    environment = manifest["environment"]
    args = manifest["args"]
    lines.append(
        f"- Created at `{manifest['created_at_utc']}`; git commit `{manifest.get('git_commit')}`.\n"
        f"- Python `{environment['python']}`, numpy `{environment['numpy']}`, pandas "
        f"`{environment['pandas']}`, scikit-learn `{environment['sklearn']}`, xgboost "
        f"`{environment['xgboost']}`, shap `{environment['shap']}`.\n"
        f"- Host `{environment['platform']}`; comparison workers `{args['workers']}` with "
        f"`OMP_NUM_THREADS=1` and XGBoost `n_jobs=1`; TreeSHAP interaction prediction used "
        f"`nthread={args['interaction_threads']}` per worker so at most "
        f"{args['workers'] * args['interaction_threads']} threads are active.\n"
        "- On headless Linux / WSL2 hosts, preload a CPU-only `libcuda.so.1` stub on "
        "`LD_LIBRARY_PATH` so that XGBoost's CUDA probe reports no device; the stub does not change "
        "any computation.\n"
        "- Platform reproducibility note: runs executed on Linux x86_64 vs macOS arm64 may differ "
        "slightly in XGBoost histogram tie-breaking. Dataset A reproduces the published GADS MRR and "
        "HR@K in all 100 runs, whereas Dataset B (590 correlated sensors, many near-ties) can differ "
        "on a minority of MRR values because near-tied SHAP scores change order. Every head-to-head "
        "number in this file is computed from the same platform-consistent models, so the GADS column "
        "is the correct in-run reference for the competitors.\n"
        "- Each per-run ranking table is stored under `runs/<dataset>_<scenario>_seed<seed>.parquet` "
        "and its metadata (KPI AUC, stage timings, diagnostics, metrics) under the matching `.json`.\n"
    )
    if notes:
        lines.append("\n## Execution notes\n")
        lines.extend(f"- {note}\n" for note in notes)
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bdcc_revision/pack2_competitors_runtime"),
        help="output directory for CSVs, JSON, summary.md and per-run artifacts",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--workers", type=int, default=6, help="comparison worker processes")
    parser.add_argument(
        "--interaction-threads",
        type=int,
        default=3,
        help="XGBoost prediction threads used only for the TreeSHAP interaction call",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("comparison", "runtime", "m2oe-caps", "glm-extra"),
        default=("comparison", "runtime"),
        help="which stage(s) to execute",
    )
    parser.add_argument(
        "--treeshap-int-normal-subsample",
        "--interaction-row-fraction",
        dest="interaction_row_fraction",
        type=float,
        default=1.0,
        help=(
            "fraction of normal-pool OOF rows used for TreeSHAP interaction scoring "
            "(all abnormal rows are always kept; 1.0 keeps the complete sample; the saved "
            "20-seed runs used 0.25)"
        ),
    )
    parser.add_argument("--runtime-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--runtime-scenario", default="S2")
    parser.add_argument("--runtime-methods", nargs="+", choices=BENCHMARK_METHODS, default=list(BENCHMARK_METHODS))
    parser.add_argument(
        "--m2oe-caps",
        nargs="+",
        type=int,
        default=[M2OE_COMPARISON_MAX_GROUP_ROWS, M2OE_MAX_GROUP_ROWS],
        help="group-row caps measured by the m2oe-caps stage",
    )
    parser.add_argument("--m2oe-repeats", type=int, default=5, help="repeats per cap in the m2oe-caps stage")
    parser.add_argument("--m2oe-dataset", choices=DATASETS, default="dataset_b")
    parser.add_argument("--m2oe-scenario", default="S2")
    parser.add_argument("--m2oe-seed", type=int, default=0)
    parser.add_argument(
        "--no-memory-pass",
        action="store_true",
        help="skip the separate tracemalloc seed-0 memory pass",
    )
    parser.add_argument("--no-self-check", action="store_true", help="skip the fold-loop/SHAP equivalence checks")
    parser.add_argument("--no-resume", action="store_true", help="recompute runs even when cached outputs exist")
    parser.add_argument("--summary-only", action="store_true", help="rebuild tables from existing per-run outputs")
    parser.add_argument(
        "--note",
        dest="notes",
        action="append",
        default=[],
        help="free-text execution note appended verbatim to summary.md (repeatable)",
    )
    return parser


def require_supported_sklearn() -> None:
    """Fail loudly unless the scikit-learn penalty API used here is available."""
    import sklearn

    parts = sklearn.__version__.split(".")
    if (int(parts[0]), int(parts[1])) < (1, 8):
        raise RuntimeError(
            f"this pack uses the scikit-learn >= 1.8 l1_ratio penalty API; installed version is {sklearn.__version__}"
        )


def environment_record() -> dict[str, Any]:
    """Record the software/host environment in the run manifest."""
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "cpu_count": os.cpu_count(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }


def git_commit() -> str | None:
    """Return the current git commit hash if available."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_comparison_stage(args: argparse.Namespace, output_dir: Path) -> None:
    """Run the 20-seed comparison across the multiprocessing pool."""
    import multiprocessing

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "dataset": dataset,
            "scenario": scenario,
            "seed": int(seed),
            "runs_dir": str(runs_dir),
            "interaction_threads": int(args.interaction_threads),
            "interaction_row_fraction": float(args.interaction_row_fraction),
            "resume": not args.no_resume,
        }
        for dataset in args.datasets
        for scenario in args.scenarios
        for seed in args.seeds
    ]
    _log(f"[comparison] {len(tasks)} tasks with {args.workers} workers")
    context = multiprocessing.get_context("spawn")
    failures: list[dict[str, Any]] = []
    completed = 0
    start = time.perf_counter()
    with context.Pool(processes=int(args.workers)) as pool:
        for result in pool.imap_unordered(run_comparison_task, tasks):
            if result["status"] == "failed":
                failures.append(result)
                _log(f"[comparison] FAILED {result['run_id']}: {result['error']}")
            elif result["status"] == "completed":
                completed += 1
                _log(f"[comparison] {result['run_id']} done in {result['seconds']:.1f}s")
    if failures:
        raise RuntimeError(f"comparison tasks failed: {[failure['run_id'] for failure in failures]}")
    _log(f"[comparison] {completed} runs completed in {time.perf_counter() - start:.1f}s")


def merge_glm_l1_half_run(
    dataset: str,
    scenario: str,
    seed: int,
    *,
    runs_dir: Path,
) -> dict[str, Any]:
    """Add the L1 ``C=0.5`` joint-interaction ranking to one stored run."""
    run_id = f"{dataset}_{scenario}_seed{seed}"
    parquet_path = runs_dir / f"{run_id}.parquet"
    meta_path = runs_dir / f"{run_id}.json"
    if not parquet_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing stored run artifacts for {run_id}")
    stored = pd.read_parquet(parquet_path)
    if (stored["method"] == GLM_JOINT_L1_HALF_METHOD).any():
        return {"run_id": run_id, "status": "cached"}
    bundle = build_bundle(dataset, scenario, seed)
    X, y, groups = bundle_matrices(bundle)
    process_frame = X.loc[:, list(bundle.process_features)]
    start = time.perf_counter()
    rows, diagnostics = rank_glm_int_joint(
        process_frame,
        y,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        penalty="l1",
        c_value=0.5,
        solver="saga",
        max_iter=500,
        tol=1e-3,
        method_name=GLM_JOINT_L1_HALF_METHOD,
    )
    seconds = time.perf_counter() - start
    frame = pd.DataFrame(rows)
    frame["is_root_cause"] = [
        feature in set(bundle.root_causes.get(str(device), ()))
        for device, feature in zip(frame["equipment"], frame["feature"], strict=True)
    ]
    frame["is_harmless"] = frame["feature"].isin(bundle.harmless_features)
    frame.insert(0, "dataset", dataset)
    frame.insert(1, "scenario", scenario)
    frame.insert(2, "seed", int(seed))
    merged = pd.concat([stored, frame.reindex(columns=stored.columns)], ignore_index=True)
    temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
    merged.to_parquet(temporary_parquet, index=False)
    temporary_parquet.replace(parquet_path)
    metrics = evaluate_rankings(
        frame,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=TOP_KS,
    )
    equipment_metrics = evaluate_rankings_per_equipment(
        frame,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=TOP_KS,
    )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for column, value in (
        ("dataset", dataset),
        ("scenario", scenario),
        ("seed", int(seed)),
        ("kpi_auc", float(metadata["kpi_auc"])),
    ):
        metrics.insert(0, column, value)
        equipment_metrics.insert(0, column, value)
    metadata["timings_seconds"][GLM_JOINT_L1_HALF_METHOD] = seconds
    metadata["diagnostics"][GLM_JOINT_L1_HALF_METHOD] = diagnostics
    metadata["metrics"].extend(metrics.to_dict(orient="records"))
    metadata["equipment_metrics"].extend(equipment_metrics.to_dict(orient="records"))
    temporary_meta = meta_path.with_suffix(".json.tmp")
    _write_json(temporary_meta, metadata)
    temporary_meta.replace(meta_path)
    return {"run_id": run_id, "status": "updated", "seconds": seconds}


def run_glm_l1_half_stage(args: argparse.Namespace, output_dir: Path) -> None:
    """Add the missing L1 ``C=0.5`` ranking to every stored comparison run."""
    runs_dir = output_dir / "runs"
    tasks = [
        (dataset, scenario, int(seed))
        for dataset in args.datasets
        for scenario in args.scenarios
        for seed in args.seeds
        if (runs_dir / f"{dataset}_{scenario}_seed{int(seed)}.parquet").is_file()
    ]
    _log(f"[glm-extra] {len(tasks)} stored runs to extend")
    updated = 0
    start = time.perf_counter()
    for dataset, scenario, seed in tasks:
        result = merge_glm_l1_half_run(dataset, scenario, seed, runs_dir=runs_dir)
        if result["status"] == "updated":
            updated += 1
            _log(f"[glm-extra] {result['run_id']} done in {result['seconds']:.1f}s")
    _log(f"[glm-extra] {updated} runs updated in {time.perf_counter() - start:.1f}s")


def run_runtime_stage(args: argparse.Namespace, output_dir: Path) -> None:
    """Run the sequential runtime/memory benchmark on the idle machine."""
    stage_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for seed in args.runtime_seeds:
            for method in args.runtime_methods:
                for n_jobs in (1, 8) if method == "GADS" else (1,):
                    result = run_runtime_benchmark_task(
                        {
                            "dataset": dataset,
                            "scenario": args.runtime_scenario,
                            "seed": int(seed),
                            "method": method,
                            "gads_jobs": n_jobs,
                            "interaction_row_fraction": float(args.interaction_row_fraction),
                            # The runtime table historically measured cap=128; keep
                            # that configuration and let the result label it explicitly.
                            "m2oe_max_group_rows": M2OE_MAX_GROUP_ROWS,
                            "use_tracemalloc": False,
                        }
                    )
                    measured_method = str(result["method"])
                    for stage, seconds in result["stages"].items():
                        stage_rows.append(
                            {
                                "dataset": dataset,
                                "scenario": args.runtime_scenario,
                                "seed": int(seed),
                                "method": measured_method,
                                "m2oe_max_group_rows": result["m2oe_max_group_rows"],
                                "n_jobs": int(result["n_jobs"]),
                                "stage": stage,
                                "seconds": float(seconds),
                                "n_rows": result["n_rows"],
                                "n_process_features": result["n_process_features"],
                            }
                        )
                    memory_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": args.runtime_scenario,
                            "seed": int(seed),
                            "method": measured_method,
                            "m2oe_max_group_rows": result["m2oe_max_group_rows"],
                            "n_jobs": int(result["n_jobs"]),
                            "load_1m_before": result["load_1m_before"],
                            "load_1m_after": result["load_1m_after"],
                            **result["memory"],
                        }
                    )
    pd.DataFrame(stage_rows).to_csv(output_dir / "runtime_per_seed.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(output_dir / "runtime_memory_per_seed.csv", index=False)
    if args.no_memory_pass:
        return
    traced_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for method in args.runtime_methods:
            for n_jobs in (1, 8) if method == "GADS" else (1,):
                result = run_runtime_benchmark_task(
                    {
                        "dataset": dataset,
                        "scenario": args.runtime_scenario,
                        "seed": 0,
                        "method": method,
                        "gads_jobs": n_jobs,
                        "interaction_row_fraction": float(args.interaction_row_fraction),
                        "m2oe_max_group_rows": M2OE_MAX_GROUP_ROWS,
                        "use_tracemalloc": True,
                    }
                )
                traced_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": args.runtime_scenario,
                        "seed": 0,
                        "method": str(result["method"]),
                        "m2oe_max_group_rows": result["m2oe_max_group_rows"],
                        "n_jobs": int(result["n_jobs"]),
                        "load_1m_before": result["load_1m_before"],
                        "load_1m_after": result["load_1m_after"],
                        **result["memory"],
                    }
                )
    pd.DataFrame(traced_rows).to_csv(output_dir / "runtime_memory_tracemalloc.csv", index=False)


def _m2oe_cap_from_label(method: str) -> int | None:
    """Return the cap encoded in an explicit ``M2OE-Group (cap=N)`` label."""
    prefix = "M2OE-Group (cap="
    if method.startswith(prefix) and method.endswith(")"):
        value = method[len(prefix) : -1]
        if value.isdigit():
            return int(value)
    return None


def _empty_cap_column(frame: pd.DataFrame) -> pd.Series:
    """Return an all-missing nullable integer column for ``len(frame)`` rows."""
    return pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)


def normalize_m2oe_runtime_stage(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[Any, ...], set[int]]]:
    """Add ``m2oe_max_group_rows`` and label only verifiable legacy M2OE rows.

    Older runtime tables stored the M2OE cap only in the method label, or left it
    generic.  A generic row is attributed to ``cap=128`` only when the stored
    evidence identifies that configuration:

    * Dataset A: every device group has 500 rows, so the original benchmark used
      ``cap=128`` for all generic rows;
    * Dataset B: only a legacy total-stage wall time above
      ``LEGACY_M2OE_CAP128_MIN_SECONDS`` came from the original ``cap=128`` run.

    Anything else raises instead of being silently relabelled.  The returned map
    records every cap measured for each runtime cell because the cap sweep can
    time the same (dataset, scenario, seed, n_jobs) cell at several caps.
    """
    frame = frame.copy()
    frame["m2oe_max_group_rows"] = _empty_cap_column(frame)
    methods = frame["method"].astype(str)
    labeled_caps = methods.map(_m2oe_cap_from_label)
    labeled = labeled_caps.notna()
    frame.loc[labeled, "m2oe_max_group_rows"] = pd.array(labeled_caps[labeled].astype(int), dtype="Int64")

    generic = methods == "M2OE-Group"
    if generic.any():
        dataset_a = generic & (frame["dataset"] == "dataset_a")
        frame.loc[dataset_a, "method"] = M2OE_CAP_LABEL.format(cap=M2OE_MAX_GROUP_ROWS)
        frame.loc[dataset_a, "m2oe_max_group_rows"] = M2OE_MAX_GROUP_ROWS

        dataset_b = generic & (frame["dataset"] == "dataset_b")
        if dataset_b.any():
            totals = frame.loc[dataset_b & (frame["stage"] == "total"), [*M2OE_RUNTIME_KEYS, "seconds"]]
            if totals.empty:
                raise ValueError("legacy generic M2OE-Group rows on Dataset B have no total-stage timing")
            slow_keys = {
                tuple(record)
                for record in totals.loc[
                    totals["seconds"] > LEGACY_M2OE_CAP128_MIN_SECONDS, list(M2OE_RUNTIME_KEYS)
                ].itertuples(index=False, name=None)
            }
            for index in frame.index[dataset_b]:
                key = tuple(frame.loc[index, list(M2OE_RUNTIME_KEYS)].tolist())
                slow = totals.loc[
                    (totals["dataset"] == key[0])
                    & (totals["scenario"] == key[1])
                    & (totals["seed"] == key[2])
                    & (totals["n_jobs"] == key[3]),
                    "seconds",
                ]
                if slow.empty:
                    raise ValueError(f"legacy generic M2OE-Group row {key} has no total-stage timing")
                if key not in slow_keys:
                    raise ValueError(
                        f"legacy generic M2OE-Group row {key} took {float(slow.iloc[0]):.1f}s, below the "
                        f"{LEGACY_M2OE_CAP128_MIN_SECONDS:.0f}s cap=128 signature; refusing to relabel it"
                    )
                frame.loc[index, "method"] = M2OE_CAP_LABEL.format(cap=M2OE_MAX_GROUP_ROWS)
                frame.loc[index, "m2oe_max_group_rows"] = M2OE_MAX_GROUP_ROWS

        other = generic & ~dataset_a & ~dataset_b
        if other.any():
            unknown = sorted(frame.loc[other, "dataset"].unique())
            raise ValueError(f"generic M2OE-Group rows on unknown datasets: {unknown}")

    key_to_cap: dict[tuple[Any, ...], set[int]] = {}
    for index in frame.index[frame["method"].astype(str).str.startswith("M2OE-Group")]:
        if pd.isna(frame.loc[index, "m2oe_max_group_rows"]):
            raise ValueError(f"M2OE-Group row {index} has no recorded max_group_rows cap")
        key = tuple(frame.loc[index, list(M2OE_RUNTIME_KEYS)].tolist())
        cap = int(frame.loc[index, "m2oe_max_group_rows"])
        key_to_cap.setdefault(key, set()).add(cap)
    return frame, key_to_cap


def apply_m2oe_cap_map(frame: pd.DataFrame, key_to_cap: dict[tuple[Any, ...], set[int]]) -> pd.DataFrame:
    """Apply the stage-level M2OE cap map to a memory or tracemalloc table."""
    frame = frame.copy()
    frame["m2oe_max_group_rows"] = _empty_cap_column(frame)
    methods = frame["method"].astype(str)
    labeled_caps = methods.map(_m2oe_cap_from_label)
    labeled = labeled_caps.notna()
    frame.loc[labeled, "m2oe_max_group_rows"] = pd.array(labeled_caps[labeled].astype(int), dtype="Int64")
    for index in frame.index[labeled]:
        key = tuple(frame.loc[index, list(M2OE_RUNTIME_KEYS)].tolist())
        cap = int(frame.loc[index, "m2oe_max_group_rows"])
        if key in key_to_cap and cap not in key_to_cap[key]:
            raise ValueError(f"M2OE-Group row {key} is labelled cap={cap} but timed at {sorted(key_to_cap[key])}")

    generic = methods == "M2OE-Group"
    for index in frame.index[generic]:
        key = tuple(frame.loc[index, list(M2OE_RUNTIME_KEYS)].tolist())
        if key not in key_to_cap:
            raise ValueError(f"generic M2OE-Group row {key} has no matching runtime-stage cap")
        if len(key_to_cap[key]) != 1:
            raise ValueError(f"generic M2OE-Group row {key} is ambiguous: the stage timed {sorted(key_to_cap[key])}")
        cap = next(iter(key_to_cap[key]))
        frame.loc[index, "method"] = M2OE_CAP_LABEL.format(cap=cap)
        frame.loc[index, "m2oe_max_group_rows"] = cap
    return frame


def run_m2oe_cap_stage(args: argparse.Namespace, output_dir: Path) -> None:
    """Measure M2OE-Group at each configured group cap on one dataset/scenario/seed.

    The accuracy table (``tab:TabModern`` / ``tab:TabInteractionModern``) uses
    ``max_group_rows=64`` while the original runtime table used 128.  This stage
    runs the requested caps under the same uncontended single-process
    conditions, labels every row with its explicit cap, preserves already
    labelled ``cap=128`` rows, and replaces only the caps that are re-measured.
    """
    stage_path = output_dir / "runtime_per_seed.csv"
    memory_path = output_dir / "runtime_memory_per_seed.csv"
    if not stage_path.is_file() or not memory_path.is_file():
        raise FileNotFoundError(
            f"the m2oe-caps stage needs existing runtime tables in {output_dir}; run the runtime stage first"
        )
    stage_frame = pd.read_csv(stage_path)
    memory_frame = pd.read_csv(memory_path)
    stage_frame, cap_by_key = normalize_m2oe_runtime_stage(stage_frame)
    memory_frame = apply_m2oe_cap_map(memory_frame, cap_by_key)
    rerun_labels = {M2OE_CAP_LABEL.format(cap=int(cap)) for cap in args.m2oe_caps}
    cell_mask = (
        (stage_frame["dataset"] == args.m2oe_dataset)
        & (stage_frame["scenario"] == args.m2oe_scenario)
        & (stage_frame["method"].isin(rerun_labels))
    )
    stage_frame = stage_frame.loc[~cell_mask].reset_index(drop=True)
    memory_mask = (
        (memory_frame["dataset"] == args.m2oe_dataset)
        & (memory_frame["scenario"] == args.m2oe_scenario)
        & (memory_frame["method"].isin(rerun_labels))
    )
    memory_frame = memory_frame.loc[~memory_mask].reset_index(drop=True)
    stage_rows = stage_frame.to_dict(orient="records")
    memory_rows = memory_frame.to_dict(orient="records")
    for cap in args.m2oe_caps:
        for repeat in range(args.m2oe_repeats):
            result = run_runtime_benchmark_task(
                {
                    "dataset": args.m2oe_dataset,
                    "scenario": args.m2oe_scenario,
                    "seed": int(args.m2oe_seed),
                    "method": "M2OE-Group",
                    "gads_jobs": 1,
                    "interaction_row_fraction": float(args.interaction_row_fraction),
                    "m2oe_max_group_rows": int(cap),
                    "use_tracemalloc": False,
                }
            )
            method = str(result["method"])
            for stage, seconds in result["stages"].items():
                stage_rows.append(
                    {
                        "dataset": args.m2oe_dataset,
                        "scenario": args.m2oe_scenario,
                        "seed": int(args.m2oe_seed),
                        "method": method,
                        "m2oe_max_group_rows": result["m2oe_max_group_rows"],
                        "n_jobs": int(result["n_jobs"]),
                        "stage": stage,
                        "seconds": float(seconds),
                        "n_rows": result["n_rows"],
                        "n_process_features": result["n_process_features"],
                    }
                )
            memory_rows.append(
                {
                    "dataset": args.m2oe_dataset,
                    "scenario": args.m2oe_scenario,
                    "seed": int(args.m2oe_seed),
                    "method": method,
                    "m2oe_max_group_rows": result["m2oe_max_group_rows"],
                    "n_jobs": int(result["n_jobs"]),
                    "load_1m_before": result["load_1m_before"],
                    "load_1m_after": result["load_1m_after"],
                    **result["memory"],
                }
            )
            _log(
                f"[m2oe-caps] {args.m2oe_dataset} {args.m2oe_scenario} seed={args.m2oe_seed} "
                f"cap={cap} repeat={repeat + 1}/{args.m2oe_repeats} "
                f"total={result['stages']['total']:.2f}s"
            )
    pd.DataFrame(stage_rows).to_csv(stage_path, index=False)
    pd.DataFrame(memory_rows).to_csv(memory_path, index=False)


def build_runtime_summary(output_dir: Path) -> pd.DataFrame | None:
    """Build the runtime summary table from the per-seed CSVs."""
    runtime_path = output_dir / "runtime_per_seed.csv"
    if not runtime_path.is_file():
        return None
    runtime_stage = pd.read_csv(runtime_path)
    if "m2oe_max_group_rows" not in runtime_stage.columns:
        raise ValueError(
            "runtime_per_seed.csv lacks the m2oe_max_group_rows column; rerun the runtime stage "
            "or normalize the legacy table before rebuilding the summary"
        )
    rows: list[dict[str, Any]] = []
    for (dataset, method, n_jobs, stage), frame in runtime_stage.groupby(
        ["dataset", "method", "n_jobs", "stage"], dropna=False
    ):
        caps = frame["m2oe_max_group_rows"].dropna().unique()
        if len(caps) > 1:
            raise ValueError(
                f"runtime rows for {dataset}/{method}/n_jobs={n_jobs}/{stage} disagree on "
                f"m2oe_max_group_rows: {sorted(int(cap) for cap in caps)}"
            )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "m2oe_max_group_rows": int(caps[0]) if len(caps) == 1 else pd.NA,
                "n_jobs": int(n_jobs),
                "stage": stage,
                "seconds_mean": float(frame["seconds"].mean()),
                "seconds_sd": float(frame["seconds"].std(ddof=1)),
                "n_runs": len(frame),
            }
        )
    summary = pd.DataFrame(rows)
    memory_path = output_dir / "runtime_memory_per_seed.csv"
    if memory_path.is_file():
        memory_frame = pd.read_csv(memory_path)
        if not memory_frame.empty:
            memory_means = (
                memory_frame.groupby(["dataset", "method", "n_jobs"], dropna=False)[["rss_delta_mb", "rss_peak_mb"]]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "rss_delta_mb": "peak_rss_delta_mb_mean",
                        "rss_peak_mb": "peak_rss_mb_mean",
                    }
                )
            )
            summary = summary.merge(memory_means, on=["dataset", "method", "n_jobs"], how="left")
    traced_path = output_dir / "runtime_memory_tracemalloc.csv"
    if traced_path.is_file():
        traced_frame = pd.read_csv(traced_path)
        if not traced_frame.empty:
            traced_means = (
                traced_frame.groupby(["dataset", "method", "n_jobs"], dropna=False)["tracemalloc_peak_mb"]
                .mean()
                .reset_index()
                .rename(columns={"tracemalloc_peak_mb": "tracemalloc_peak_mb_seed0"})
            )
            summary = summary.merge(traced_means, on=["dataset", "method", "n_jobs"], how="left")
    summary.to_csv(output_dir / "runtime_summary.csv", index=False)
    return summary


def load_self_checks(output_dir: Path, self_checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the provided self-check records, or replay a persisted record."""
    if self_checks:
        _write_json(output_dir / "self_check.json", self_checks)
        return self_checks
    path = output_dir / "self_check.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def empty_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return empty comparison tables with the canonical columns."""
    summary_columns = [
        "dataset",
        "scenario",
        "method",
        "n_seeds",
        "kpi_auc_mean",
        *[f"{metric}_{suffix}" for metric in METRIC_COLUMNS for suffix in ("mean", "sd", "n_defined")],
    ]
    pooled_columns = [
        "dataset",
        "method",
        "n_runs_root",
        "n_runs_far",
        *[f"{metric}_{suffix}" for metric in ("mrr", "hr_at_1", "hr_at_3", "hr_at_5") for suffix in ("mean", "sd")],
        *[f"{metric}_{suffix}" for metric in ("far_at_1", "far_at_5") for suffix in ("mean", "sd")],
    ]
    return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=pooled_columns), pd.DataFrame()


def rebuild_tables(output_dir: Path, args: argparse.Namespace, self_checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run artifacts into CSV/JSON tables and ``summary.md``."""
    runs_dir = output_dir / "runs"
    self_checks = load_self_checks(output_dir, self_checks)
    per_seed, equipment, metadata = load_comparison_outputs(runs_dir)
    comparison_available = not per_seed.empty
    interaction_row_fraction, n_observed_fractions = observed_interaction_row_fraction(
        metadata, args.interaction_row_fraction
    )
    if comparison_available:
        comparison_summary = summarize_comparison(per_seed)
        pooled = summarize_pooled(per_seed)
        diagnostics_summary = summarize_diagnostics(metadata)
        per_seed = per_seed.sort_values(["dataset", "scenario", "seed", "method"], kind="stable").reset_index(drop=True)
        per_seed.to_csv(output_dir / "comparison_per_seed.csv", index=False)
        if not equipment.empty:
            equipment.sort_values(["dataset", "scenario", "seed", "method", "equipment"], kind="stable").reset_index(
                drop=True
            ).to_csv(output_dir / "comparison_equipment_per_seed.csv", index=False)
        comparison_summary.to_csv(output_dir / "comparison_summary.csv", index=False)
        pooled.to_csv(output_dir / "comparison_pooled.csv", index=False)
        if not diagnostics_summary.empty:
            diagnostics_summary.to_csv(output_dir / "interaction_diagnostics_per_seed.csv", index=False)
    else:
        comparison_summary, pooled, diagnostics_summary = empty_comparison_tables()
        if build_runtime_summary(output_dir) is None:
            raise RuntimeError(f"no per-run comparison outputs found under {runs_dir}")
    runtime_summary = build_runtime_summary(output_dir)
    manifest = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": git_commit(),
        "args": {
            "datasets": list(args.datasets),
            "scenarios": list(args.scenarios),
            "seeds": [int(seed) for seed in args.seeds],
            "workers": int(args.workers),
            "interaction_threads": int(args.interaction_threads),
            "interaction_row_fraction": float(interaction_row_fraction),
            "interaction_row_fraction_source": (
                f"stored run metadata ({n_observed_fractions} runs)"
                if n_observed_fractions
                else "command line (no per-run metadata available)"
            ),
            "runtime_seeds": [int(seed) for seed in args.runtime_seeds],
            "runtime_scenario": args.runtime_scenario,
            "m2oe_max_group_rows_accuracy": int(M2OE_COMPARISON_MAX_GROUP_ROWS),
            "m2oe_caps": [int(cap) for cap in args.m2oe_caps],
            "m2oe_repeats": int(args.m2oe_repeats),
            "m2oe_cell": {
                "dataset": args.m2oe_dataset,
                "scenario": args.m2oe_scenario,
                "seed": int(args.m2oe_seed),
            },
            "notes": list(getattr(args, "notes", ())),
        },
        "environment": environment_record(),
        "comparison_available": comparison_available,
        "n_runs": len(metadata),
        "n_unique_runs": (
            int(metadata[["dataset", "scenario", "seed"]].drop_duplicates().shape[0]) if comparison_available else 0
        ),
        "comparison_summary_rows": len(comparison_summary),
        "self_checks": self_checks,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    write_markdown_summary(
        output_dir,
        comparison_summary=comparison_summary,
        pooled=pooled,
        diagnostics_summary=diagnostics_summary,
        runtime_summary=runtime_summary,
        manifest=manifest,
        notes=list(getattr(args, "notes", ())),
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the requested stages and rebuild the tables."""
    require_supported_sklearn()
    args = build_parser().parse_args(argv)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        rebuild_tables(output_dir, args, self_checks=[])
        return 0
    self_checks: list[dict[str, Any]] = []
    if not args.no_self_check:
        for dataset in args.datasets:
            self_checks.append(self_check_dataset(dataset, interaction_threads=int(args.interaction_threads)))
            _log(f"[self-check] {dataset}: {self_checks[-1]}")
    if "comparison" in args.stages:
        run_comparison_stage(args, output_dir)
    if "glm-extra" in args.stages:
        run_glm_l1_half_stage(args, output_dir)
    if "runtime" in args.stages:
        run_runtime_stage(args, output_dir)
    if "m2oe-caps" in args.stages:
        run_m2oe_cap_stage(args, output_dir)
    rebuild_tables(output_dir, args, self_checks=self_checks)
    _log(f"[done] outputs written under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

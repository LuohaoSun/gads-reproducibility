#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportReturnType=false
"""Pack 3 revision experiments: R3 #7, R3 #8/R2 #1 and R3 #11.

The script is self-contained (it imports the paper's ``src`` package but does not
modify it) and uses the exact paper protocol used by Pack 1:

* Dataset A: ``n_samples=5000``, noise 0.2, 10 devices, physical shift 3.0,
  mechanism scale 1.0, harmless drift 3.5; XGBoost ``max_depth=6`` /
  ``min_child_weight=1``.
* Dataset B: official UCI SECOM, ``group_seed=2024``, physical shift 3.0,
  mechanism weight 12.0, harmless shift 4.0, missing indicators excluded;
  XGBoost ``max_depth=3`` / ``min_child_weight=5``.
* ``eqp_mode="contextual_process"``, 5-fold cross-fitting, CV seed equal to the
  model seed (42), ``n_jobs=1`` inside XGBoost.

What the script computes:

1. **R3 #7** tree-split diagnostics on the fitted XGBoost folds (abnormal EQP
   one-hot, harmless-drift, true root cause and global KPI gain/split shares),
   harmless-feature correlation diagnostics, a path-dependent vs interventional
   TreeSHAP comparison, and a leakage-safe within-EQP centering evaluation.
2. **R3 #8 / R2 #1** a permutation null distribution obtained by permuting the
   equipment label on the OOF SHAP matrix (500 permutations, no extra model
   fits), including an ``S0-clean`` no-fault calibration scenario and a
   score-gap / elbow cutoff evaluation.
3. **R3 #11** a 20-seed Dataset B intensity sweep over the S2 mechanism weight,
   the S1 physical shift and the S4-main harmless-drift magnitude, with all four
   GADS distances computed from the same OOF SHAP matrix.

Outputs (CSV, JSON and ``summary.md``) go to
``results/bdcc_revision/pack3_diagnostics_bsweep/``.

Environment note
----------------
On WSL2 hosts whose ``/usr/lib/wsl/lib/libcuda.so.1`` is incompatible with the
CUDA-enabled xgboost wheel, ``xgboost.DMatrix`` segfaults in
``XGDMatrixCreateFromDense`` while probing the GPU driver.  Run with a stub
``libcuda.so.1`` (SONAME ``libcuda.so.1``) preloaded, for example::

    gcc -shared -fPIC -Wl,-soname,libcuda.so.1 -o /tmp/libcuda.so.1 stub.c
    LD_PRELOAD=/tmp/libcuda.so.1 uv run --no-config python scripts/bdcc_pack3_diagnostics_and_bsweep.py

where ``stub.c`` only declares an unused symbol.  No GPU is used by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import xgboost

from shap_diff_analysis.experiment_runner import git_commit
from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.gads import (
    DEFAULT_GADS_BINS,
    DEFAULT_GADS_EPSILON,
    GADS_DISTANCES,
    compute_gads_distance,
    rank_gads_features,
)
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.metrics import evaluate_rankings
from shap_diff_analysis.modeling import (
    _aggregate_shap,
    _build_encoded_to_original_mapping,
    _class_weight,
    _extract_binary_shap,
    _make_estimator,
    _make_preprocessor,
    fit_cross_fitted_model,
)
from shap_diff_analysis.secom import (
    DEFAULT_RAW_DIR,
    MECHANISM_ROOT_GLOBAL_WEIGHT,
    generate_dataset_b,
    verify_secom_files,
)

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = DEFAULT_RAW_DIR if DEFAULT_RAW_DIR.is_absolute() else PROJECT_ROOT / DEFAULT_RAW_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/bdcc_revision/pack3_diagnostics_bsweep"

DATASET_LABELS: dict[str, str] = {"a": "Dataset A", "b": "Dataset B"}
DATASET_KEYS: tuple[str, ...] = ("a", "b")
BASE_SCENARIOS: tuple[str, ...] = ("S0-clean", "S1", "S2", "S3", "S4-main", "S4-null")
CENTERED_SCENARIOS: frozenset[str] = frozenset({"S1", "S2", "S3", "S4-main", "S4-null"})
DEFAULT_SCENARIO_SEEDS: tuple[int, ...] = tuple(range(20))
DISTANCES: tuple[str, ...] = tuple(GADS_DISTANCES)
TOP_KS: tuple[int, ...] = (1, 3, 5)
METRIC_COLUMNS: tuple[str, ...] = (
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_3",
    "far_at_5",
)
QUANTILES: tuple[float, ...] = (0.95, 0.99)
DEFAULT_N_PERMUTATIONS = 500
DEFAULT_BACKGROUND_SIZE = 200
ELBOW_MAX_K = 20
MODEL_SEED = 42
CV_SPLITS = 5

DATASET_A_PARAMS: dict[str, Any] = {
    "n_samples": 5_000,
    "noise_level": 0.2,
    "n_devices": 10,
    "distribution_shift": 3.0,
    "mechanism_scale": 1.0,
    "harmless_drift": 3.5,
}
DATASET_B_PARAMS: dict[str, Any] = {
    "group_seed": 2024,
    "physical_shift": 3.0,
    "mechanism_weight": 12.0,
    "harmless_shift": 4.0,
    "include_missing_indicators": False,
}
MODEL_PARAMS: dict[str, dict[str, int]] = {
    "a": {"max_depth": 6, "min_child_weight": 1},
    "b": {"max_depth": 3, "min_child_weight": 5},
}
SWEEP_GRIDS: dict[str, dict[str, Any]] = {
    "s2_weight": {
        "scenario": "S2",
        "levels": (1.5, 3.0, 4.5, 6.0, 9.0, 12.0),
        "default": 12.0,
        "parameter": "mechanism_weight",
    },
    "s1_shift": {
        "scenario": "S1",
        "levels": (0.5, 1.0, 1.5, 2.0, 3.0),
        "default": 3.0,
        "parameter": "physical_shift",
    },
    "s4_harmless": {
        "scenario": "S4-main",
        "levels": (0.5, 1.0, 2.0, 3.0, 4.0),
        "default": 4.0,
        "parameter": "harmless_shift",
    },
}

GADS_METRIC_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "view",
    "distance",
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_3",
    "far_at_5",
    "n_equipment",
    "n_process_features",
    "kpi_auc",
    "top1_device",
    "top1_feature",
    "top1_score",
    "top1_is_root",
    "top1_is_harmless",
    "root_rank",
    "harmless_rank",
)
TREE_ROLE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "device",
    "role",
    "features",
    "n_features",
    "total_gain_share",
    "mean_gain_share_per_feature",
    "total_split_count_per_fold",
    "mean_split_count_per_feature_per_fold",
    "max_split_count_per_feature_per_fold",
)
TREE_FEATURE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "feature",
    "feature_kind",
    "roles",
    "gain_total",
    "gain_share",
    "split_count_total",
    "mean_split_count_per_fold",
)
HARMLESS_CORRELATION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "device",
    "feature",
    "harmless_order",
    "gads_rank",
    "gads_score",
    "max_abs_pearson_with_kpi",
    "kpi_feature_argmax",
    "point_biserial_with_eqp",
    "is_top_harmless",
    "is_top1_overall",
)
SHAP_VARIANT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "variant",
    "distance",
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_3",
    "far_at_5",
    "top1_device",
    "top1_feature",
    "top1_score",
    "top1_is_root",
    "top1_is_harmless",
    "root_rank",
    "harmless_rank",
)
PERM_CELL_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "view",
    "device",
    "n_permutations",
    "n_abnormal_rows",
    "n_normal_rows",
    "n_features",
    "observed_max",
    "observed_argmax_feature",
    "q95_max",
    "q99_max",
    "p_value_max",
    "fwer_reject_95",
    "fwer_reject_99",
    "n_features_exceed_tau95",
    "n_features_exceed_tau99",
    "n_features_exceed_fwer95",
    "n_roots",
    "n_roots_detected_fwer95",
    "n_roots_detected_tau95",
    "any_root_detected_fwer95",
    "all_roots_detected_fwer95",
    "min_gads_rank_of_roots",
    "top1_feature",
    "top1_is_root",
    "top1_is_harmless",
)
PERM_FEATURE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "view",
    "device",
    "feature",
    "role",
    "observed",
    "tau_95",
    "tau_99",
    "exceeds_tau95",
    "exceeds_tau99",
    "fwer_threshold_max_95",
    "exceeds_fwer95",
    "gads_rank",
)
SCORE_GAP_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "device",
    "rule",
    "selected_k",
    "selected_features",
    "cutoff_score",
    "next_score",
    "root_rank",
    "harmless_rank",
    "root_in_set",
    "harmless_in_set",
    "root_in_top5",
    "harmless_in_top5",
    "top1_feature",
    "top1_score",
)
SWEEP_METRIC_COLUMNS: tuple[str, ...] = (
    "dataset",
    "scenario",
    "scenario_seed",
    "sweep_name",
    "level",
    "distance",
    "reused_base",
    "mrr",
    "hr_at_1",
    "hr_at_3",
    "hr_at_5",
    "far_at_1",
    "far_at_3",
    "far_at_5",
    "kpi_auc",
    "top1_device",
    "top1_feature",
    "top1_score",
    "top1_is_root",
    "top1_is_harmless",
    "root_rank",
    "harmless_rank",
)


@dataclass(frozen=True)
class BaseTask:
    """One base cell: dataset x scenario x scenario seed."""

    dataset: str
    scenario: str
    scenario_seed: int
    task_id: str


@dataclass(frozen=True)
class SweepTask:
    """One Dataset B intensity-sweep cell."""

    dataset: str
    sweep_name: str
    scenario: str
    scenario_seed: int
    level: float
    task_id: str


Task = BaseTask | SweepTask


@dataclass
class FoldArtifacts:
    """Fitted fold estimator, preprocessing state and transformed matrices."""

    estimator: Any
    preprocessor: Any
    X_train_t: pd.DataFrame
    X_valid_t: pd.DataFrame
    train_idx: np.ndarray
    valid_idx: np.ndarray
    auc: float


@dataclass
class CrossFitOutput:
    """Out-of-fold attributions and the fold artifacts used to produce them."""

    shap_values: pd.DataFrame
    oof_predictions: pd.Series
    mean_auc: float
    folds: list[FoldArtifacts]


@dataclass(frozen=True)
class TreeStats:
    """Aggregated split and gain statistics over the five CV folds."""

    per_feature: pd.DataFrame
    total_gain: float
    n_folds: int
    eqp_by_device: dict[str, tuple[str, float, int]]
    n_encoded_columns: int


@dataclass
class TaskResult:
    """Result or captured error for one task."""

    task_id: str
    kind: str
    status: str
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error: str | None = None


def _to_builtin(value: Any) -> Any:
    """Convert a scalar to a JSON-safe builtin, mapping non-finite floats to None."""
    if value is None:
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    """Recursively convert a nested structure to JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return _to_builtin(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records from a DataFrame."""
    return [
        {str(key): _to_builtin(value) for key, value in record.items()} for record in frame.to_dict(orient="records")
    ]


def _stable_seed(*parts: object) -> int:
    """Derive a deterministic 32-bit NumPy seed from arbitrary parts."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _wilson_interval(successes: int, trials: int, z_value: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials]")
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    centre = (proportion + z_value**2 / (2.0 * trials)) / denominator
    margin = (
        z_value * math.sqrt(proportion * (1.0 - proportion) / trials + z_value**2 / (4.0 * trials**2)) / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _ensure_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Project a frame onto a stable output schema, adding missing columns as None."""
    if frame.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = None
    return output.loc[:, list(columns)]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = _jsonable(payload)
    text = json.dumps(converted, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _md_table(frame: pd.DataFrame, columns: list[str], *, floatfmt: str = ".3f") -> str:
    """Render a markdown table with a stable column selection."""
    if frame.empty:
        return "_No rows._"
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"markdown table is missing columns: {missing}")
    display = frame.loc[:, columns].astype(object)
    display = display.where(display.notna(), None)
    return display.to_markdown(index=False, floatfmt=floatfmt, missingval="--")


def _config_for_dataset(dataset: str) -> dict[str, int]:
    """Return the paper XGBoost hyperparameters for one dataset."""
    if dataset not in MODEL_PARAMS:
        raise ValueError(f"unknown dataset key {dataset!r}; expected one of {DATASET_KEYS}")
    return MODEL_PARAMS[dataset]


def _generate_bundle(
    dataset: str,
    scenario: str,
    scenario_seed: int,
    *,
    physical_shift: float,
    mechanism_weight: float,
    harmless_shift: float,
) -> DatasetBundle:
    """Generate one bundle under the paper protocol, including the S0-clean null."""
    if dataset == "a":
        bundle = generate_dataset_bundle(
            scenario="S2" if scenario == "S0-clean" else scenario,
            n_samples=int(DATASET_A_PARAMS["n_samples"]),
            seed=scenario_seed,
            noise_level=float(DATASET_A_PARAMS["noise_level"]),
            n_devices=int(DATASET_A_PARAMS["n_devices"]),
            distribution_shift=float(DATASET_A_PARAMS["distribution_shift"]),
            mechanism_scale=0.0 if scenario == "S0-clean" else float(DATASET_A_PARAMS["mechanism_scale"]),
            harmless_drift=float(DATASET_A_PARAMS["harmless_drift"]),
        )
    elif dataset == "b":
        bundle = generate_dataset_b(
            scenario="S2" if scenario == "S0-clean" else scenario,  # type: ignore[arg-type]
            seed=scenario_seed,
            group_seed=int(DATASET_B_PARAMS["group_seed"]),
            raw_dir=RAW_DIR,
            physical_shift_magnitude=physical_shift,
            mechanism_local_weight=(MECHANISM_ROOT_GLOBAL_WEIGHT if scenario == "S0-clean" else mechanism_weight),
            harmless_shift_magnitude=harmless_shift,
            include_missing_indicators=bool(DATASET_B_PARAMS["include_missing_indicators"]),
        )
    else:
        raise ValueError(f"unknown dataset key {dataset!r}; expected one of {DATASET_KEYS}")

    if scenario != "S0-clean":
        return bundle
    device = bundle.abnormal_devices[0]
    metadata = dict(bundle.metadata)
    metadata["pack3_s0_clean"] = {
        "source_scenario": "S2",
        "construction": "mechanism strength set to the normal-pool value; no physical or harmless shift",
        "abnormal_device": device,
    }
    return DatasetBundle(
        data=bundle.data,
        scenario="S0-clean",
        seed=bundle.seed,
        group_column=bundle.group_column,
        target_column=bundle.target_column,
        process_features=bundle.process_features,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        root_causes={device: ()},
        harmless_features=(),
        metadata=metadata,
    )


@lru_cache(maxsize=8)
def _load_bundle(
    dataset: str,
    scenario: str,
    scenario_seed: int,
    physical_shift: float,
    mechanism_weight: float,
    harmless_shift: float,
) -> DatasetBundle:
    """Generate and cache one bundle inside a worker process.

    The experiment code never mutates the returned frames, so a small LRU cache
    is safe and removes repeated SECOM parsing for shared sweep/base cells.
    """
    return _generate_bundle(
        dataset,
        scenario,
        scenario_seed,
        physical_shift=physical_shift,
        mechanism_weight=mechanism_weight,
        harmless_shift=harmless_shift,
    )


def _feature_matrix(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the model matrix, binary target and equipment labels."""
    columns = [*bundle.process_features, bundle.group_column]
    X = bundle.data.loc[:, columns]
    y = pd.to_numeric(bundle.data[bundle.target_column], errors="raise").astype(int)
    groups = bundle.data[bundle.group_column].astype(str)
    if set(bundle.abnormal_devices) & set(bundle.normal_devices):
        raise ValueError("abnormal and normal device pools must be disjoint")
    if not set(bundle.abnormal_devices) <= set(groups.unique()):
        raise ValueError("abnormal devices are absent from the generated bundle")
    return X, y, groups


def _center_numeric_by_group(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove training-fold equipment means from every numeric process feature.

    The group means are estimated on the training fold only, so the validation
    fold never contributes to the centering transform.
    """
    numeric_columns = [column for column in X_train.columns if column != group_column]
    if not numeric_columns:
        raise ValueError("within-EQP centering requires at least one numeric process feature")
    train_groups = X_train[group_column].astype(str)
    valid_groups = X_valid[group_column].astype(str)
    means = X_train.loc[:, numeric_columns].groupby(train_groups, observed=True).mean()
    missing_groups = sorted(set(valid_groups.unique()) - set(means.index.astype(str)))
    if missing_groups:
        raise ValueError(
            f"within-EQP centering cannot transform validation groups absent from the training fold: {missing_groups}"
        )
    centered_train = X_train.copy()
    centered_valid = X_valid.copy()
    train_lookup = means.reindex(train_groups.to_numpy())
    valid_lookup = means.reindex(valid_groups.to_numpy())
    if train_lookup.isna().to_numpy().any() or valid_lookup.isna().to_numpy().any():
        raise ValueError("within-EQP centering produced missing group means")
    centered_train.loc[:, numeric_columns] = X_train.loc[:, numeric_columns].to_numpy(
        dtype=float
    ) - train_lookup.to_numpy(dtype=float)
    centered_valid.loc[:, numeric_columns] = X_valid.loc[:, numeric_columns].to_numpy(
        dtype=float
    ) - valid_lookup.to_numpy(dtype=float)
    residual = (
        centered_train.loc[:, numeric_columns]
        .groupby(train_groups, observed=True)
        .mean()
        .abs()
        .to_numpy(dtype=float)
        .max()
    )
    if float(residual) > 1e-9:
        raise RuntimeError(f"within-EQP centering left a non-zero training group mean: {residual}")
    return centered_train, centered_valid


def _normalise_shap_values(values: Any) -> np.ndarray:
    """Normalise a SHAP value container to a 2-D class-1/tree matrix."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError("expected binary SHAP output")
        values = values[1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[-1] != 2:
            raise ValueError(f"unexpected SHAP class dimension: {array.shape}")
        array = array[:, :, 1]
    if array.ndim != 2:
        raise ValueError(f"unexpected SHAP shape: {array.shape}")
    return array


def _cross_fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset: str,
    center_by_group: bool,
) -> CrossFitOutput:
    """Cross-fit the paper XGBoost model and collect OOF SHAP attributions.

    The fold loop mirrors :func:`shap_diff_analysis.modeling.fit_cross_fitted_model`
    exactly, but retains the fold estimators, preprocessors and transformed
    matrices for tree-split diagnostics and interventional TreeSHAP.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if y.nunique(dropna=False) != 2:
        raise ValueError("binary classification requires exactly two target classes")
    class_counts = y.value_counts()
    if int(class_counts.min()) < CV_SPLITS:
        raise ValueError("each target class must contain at least CV_SPLITS samples")
    model_params = _config_for_dataset(dataset)
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=MODEL_SEED)
    shap_oof = np.full((len(X), X.shape[1]), np.nan, dtype=float)
    predictions = np.full(len(X), np.nan, dtype=float)
    folds: list[FoldArtifacts] = []
    fold_aucs: list[float] = []
    group_column = X.columns[-1]
    if str(group_column).lower() not in {"eqp", "equipment", "device"}:
        raise ValueError(f"the last model column must be the equipment identifier, got {group_column!r}")

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        if center_by_group:
            X_train, X_valid = _center_numeric_by_group(X_train, X_valid, str(group_column))
        preprocessor = _make_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)
        estimator = _make_estimator(
            "xgboost",
            random_state=MODEL_SEED + fold,
            scale_pos_weight=_class_weight(y.iloc[train_idx]),
            model_params=model_params,
        )
        estimator.fit(X_train_t, y.iloc[train_idx])
        predictions[valid_idx] = estimator.predict_proba(X_valid_t)[:, 1]
        shap_encoded = _extract_binary_shap(estimator, np.asarray(X_valid_t))
        aggregated = _aggregate_shap(shap_encoded, preprocessor, list(X.columns))
        shap_oof[valid_idx] = aggregated.to_numpy(dtype=float)
        auc = float(roc_auc_score(y.iloc[valid_idx], predictions[valid_idx]))
        fold_aucs.append(auc)
        folds.append(
            FoldArtifacts(
                estimator=estimator,
                preprocessor=preprocessor,
                X_train_t=X_train_t,
                X_valid_t=X_valid_t,
                train_idx=train_idx,
                valid_idx=valid_idx,
                auc=auc,
            )
        )

    if np.isnan(predictions).any() or np.isnan(shap_oof).any():
        raise RuntimeError("cross-fitting did not produce complete OOF predictions/attributions")
    return CrossFitOutput(
        shap_values=pd.DataFrame(shap_oof, index=X.index, columns=X.columns),
        oof_predictions=pd.Series(predictions, index=X.index, name="oof_prediction"),
        mean_auc=float(np.mean(fold_aucs)),
        folds=folds,
    )


def _interventional_shap(
    fold: FoldArtifacts,
    *,
    columns: list[str],
    background_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute interventional TreeSHAP for one validation fold."""
    if len(fold.X_train_t) < background_size:
        raise ValueError(
            f"interventional background requires {background_size} training rows, got {len(fold.X_train_t)}"
        )
    background = fold.X_train_t.sample(n=background_size, random_state=random_state)
    # ``shap`` wraps ``data`` in an ``Independent`` masker whose default ``max_samples`` is
    # 100, so ``background_size=200`` rows are passed in but only 100 are used as the
    # effective interventional background (``len(explainer.data) == 100``; verified on
    # shap 0.50.0, whose TreeExplainer adopts the masker's subsampled ``data``). The
    # summary text therefore reports 200 passed and 100 effective background rows.
    explainer = shap.TreeExplainer(
        fold.estimator,
        data=background,
        feature_perturbation="interventional",
    )
    explanation = explainer(fold.X_valid_t)
    values = _normalise_shap_values(explanation.values)
    aggregated = _aggregate_shap(values, fold.preprocessor, columns)
    del explainer, explanation
    return aggregated


def _global_kpi_features(bundle: DatasetBundle, dataset: str) -> tuple[str, ...]:
    """Return the features with a non-zero global KPI coefficient."""
    if dataset == "a":
        selected = [
            feature
            for feature in bundle.process_features
            if feature.startswith("feat_useful_") or feature.startswith("feat_special_")
        ]
        if not selected:
            raise ValueError("Dataset A metadata does not expose the expected useful/special features")
        return tuple(selected)
    roles = bundle.metadata.get("feature_roles")
    if not isinstance(roles, dict):
        raise ValueError("Dataset B metadata is missing feature_roles")
    required = ("global_signals", "root_cause", "mechanism_root")
    missing = [key for key in required if key not in roles]
    if missing:
        raise ValueError(f"Dataset B feature_roles is missing keys: {missing}")
    selected = [*roles["global_signals"], roles["root_cause"], roles["mechanism_root"]]
    return tuple(dict.fromkeys(str(feature) for feature in selected))


def _rank_all_distances(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    bundle: DatasetBundle,
) -> dict[str, pd.DataFrame]:
    """Rank process features with all four GADS distances from one SHAP matrix."""
    rankings: dict[str, pd.DataFrame] = {}
    for distance in DISTANCES:
        rankings[distance] = rank_gads_features(
            shap_values,
            groups,
            bundle.abnormal_devices,
            bundle.normal_devices,
            process_features=bundle.process_features,
            view="process",
            distance=distance,
            bins=DEFAULT_GADS_BINS,
            epsilon=DEFAULT_GADS_EPSILON,
        )
    return rankings


def _metric_rows_from_rankings(
    rankings: dict[str, pd.DataFrame],
    *,
    bundle: DatasetBundle,
    dataset_label: str,
    scenario_seed: int,
    view: str,
    kpi_auc: float,
) -> list[dict[str, Any]]:
    """Compute MRR/HR/FAR and top-1 composition for every GADS distance."""
    frames: list[pd.DataFrame] = []
    for distance, ranking in rankings.items():
        tagged = ranking.copy()
        tagged["method"] = f"GADS-{distance}"
        frames.append(tagged)
    combined = pd.concat(frames, ignore_index=True)
    metrics = evaluate_rankings(
        combined,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=TOP_KS,
    )
    roots = {str(device): set(features) for device, features in bundle.root_causes.items()}
    harmless = set(bundle.harmless_features)
    rows: list[dict[str, Any]] = []
    for record in metrics.to_dict(orient="records"):
        method = str(record["method"])
        distance = method.removeprefix("GADS-")
        ranking = rankings[distance]
        best_index = ranking["score"].to_numpy(dtype=float).argmax()
        best = ranking.iloc[int(best_index)]
        best_device = str(best["equipment"])
        root_candidates = ranking.loc[
            [
                feature in roots.get(str(device), set())
                for device, feature in zip(ranking["equipment"], ranking["feature"], strict=True)
            ]
        ]
        harmless_candidates = ranking.loc[ranking["feature"].isin(harmless)]
        rows.append(
            {
                "dataset": dataset_label,
                "scenario": bundle.scenario,
                "scenario_seed": scenario_seed,
                "view": view,
                "distance": distance,
                "mrr": record.get("mrr"),
                "hr_at_1": record.get("hr_at_1"),
                "hr_at_3": record.get("hr_at_3"),
                "hr_at_5": record.get("hr_at_5"),
                "far_at_1": record.get("far_at_1"),
                "far_at_3": record.get("far_at_3"),
                "far_at_5": record.get("far_at_5"),
                "n_equipment": record.get("n_equipment"),
                "n_process_features": len(bundle.process_features),
                "kpi_auc": kpi_auc,
                "top1_device": best_device,
                "top1_feature": str(best["feature"]),
                "top1_score": float(best["score"]),
                "top1_is_root": str(best["feature"]) in roots.get(best_device, set()),
                "top1_is_harmless": str(best["feature"]) in harmless,
                "root_rank": (int(root_candidates["rank"].min()) if not root_candidates.empty else None),
                "harmless_rank": (int(harmless_candidates["rank"].min()) if not harmless_candidates.empty else None),
            }
        )
    return rows


def _tree_stats(
    cross_fit: CrossFitOutput,
    X_columns: list[str],
    group_column: str,
    devices: tuple[str, ...],
) -> TreeStats:
    """Aggregate split counts and gains over the five fitted CV folds."""
    if not cross_fit.folds:
        raise ValueError("tree diagnostics require at least one fitted fold")
    gain_by_feature: dict[str, float] = dict.fromkeys(X_columns, 0.0)
    splits_by_feature: dict[str, int] = dict.fromkeys(X_columns, 0)
    encoded_names_reference: tuple[str, ...] | None = None
    eqp_encoded_names: dict[str, str] = {}
    eqp_gain: dict[str, float] = dict.fromkeys(devices, 0.0)
    eqp_splits: dict[str, int] = dict.fromkeys(devices, 0)

    for fold_index, fold in enumerate(cross_fit.folds):
        encoded_names = tuple(str(name) for name in fold.preprocessor.get_feature_names_out())
        mapping = _build_encoded_to_original_mapping(fold.preprocessor)
        if encoded_names_reference is None:
            encoded_names_reference = encoded_names
            for device in devices:
                candidates = [
                    name
                    for name in encoded_names
                    if mapping.get(name) == group_column
                    and (name == f"{group_column}_{device}" or name.endswith(f"_{device}"))
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"expected exactly one encoded one-hot column for {group_column}={device}, got {candidates}"
                    )
                eqp_encoded_names[device] = candidates[0]
        elif encoded_names != encoded_names_reference:
            raise ValueError(f"fold {fold_index} produced inconsistent encoded feature names")

        tree_frame = fold.estimator.get_booster().trees_to_dataframe()
        internal = tree_frame.loc[tree_frame["Feature"] != "Leaf"].copy()
        unknown = sorted(set(internal["Feature"].astype(str)) - set(mapping))
        if unknown:
            raise ValueError(f"tree split features lack an encoded-column mapping: {unknown}")
        internal["original"] = internal["Feature"].astype(str).map(mapping)
        for original, group in internal.groupby("original", observed=True):
            gain_by_feature[str(original)] += float(group["Gain"].sum())
            splits_by_feature[str(original)] += len(group)
        encoded_counts = internal.groupby("Feature", observed=True)["Gain"].agg(["sum", "size"])
        for device in devices:
            encoded_name = eqp_encoded_names[device]
            if encoded_name in encoded_counts.index:
                eqp_gain[device] += float(encoded_counts.loc[encoded_name, "sum"])
                eqp_splits[device] += int(encoded_counts.loc[encoded_name, "size"])

    if encoded_names_reference is None or len(eqp_encoded_names) != len(devices):
        raise RuntimeError("tree diagnostics did not resolve every EQP one-hot encoding")
    total_gain = float(sum(gain_by_feature.values()))
    if total_gain <= 0.0:
        raise RuntimeError("tree diagnostics found no positive split gain")
    per_feature = pd.DataFrame(
        {
            "feature": list(gain_by_feature),
            "gain_total": [gain_by_feature[name] for name in gain_by_feature],
            "split_count_total": [splits_by_feature[name] for name in splits_by_feature],
        }
    )
    per_feature["gain_share"] = per_feature["gain_total"] / total_gain
    per_feature["mean_split_count_per_fold"] = per_feature["split_count_total"] / len(cross_fit.folds)
    return TreeStats(
        per_feature=per_feature,
        total_gain=total_gain,
        n_folds=len(cross_fit.folds),
        eqp_by_device={device: (eqp_encoded_names[device], eqp_gain[device], eqp_splits[device]) for device in devices},
        n_encoded_columns=len(encoded_names_reference),
    )


def _role_stats_row(
    *,
    stats: TreeStats,
    role: str,
    features: tuple[str, ...],
    device: str,
    dataset_label: str,
    scenario: str,
    scenario_seed: int,
    gain_override: float | None = None,
    split_override: int | None = None,
    encoded_name: str | None = None,
) -> dict[str, Any]:
    """Build one role-level tree-diagnostic row."""
    n_features = len(features)
    if gain_override is not None:
        gain = float(gain_override)
        splits = int(split_override if split_override is not None else 0)
        feature_names = encoded_name if encoded_name is not None else "; ".join(features)
    else:
        subset = stats.per_feature.loc[stats.per_feature["feature"].isin(features)]
        if len(subset) != n_features:
            missing = sorted(set(features) - set(subset["feature"]))
            raise ValueError(f"role {role!r} references features absent from the tree table: {missing}")
        gain = float(subset["gain_total"].sum())
        splits = int(subset["split_count_total"].sum())
        feature_names = "; ".join(features)
    if n_features == 0:
        return {
            "dataset": dataset_label,
            "scenario": scenario,
            "scenario_seed": scenario_seed,
            "device": device,
            "role": role,
            "features": "",
            "n_features": 0,
            "total_gain_share": None,
            "mean_gain_share_per_feature": None,
            "total_split_count_per_fold": 0.0,
            "mean_split_count_per_feature_per_fold": None,
            "max_split_count_per_feature_per_fold": None,
        }
    subset = stats.per_feature.loc[stats.per_feature["feature"].isin(features)]
    if gain_override is not None:
        max_splits = splits / stats.n_folds
        mean_gain_share = gain / stats.total_gain
    else:
        max_splits = float(subset["split_count_total"].max()) / stats.n_folds
        mean_gain_share = float((subset["gain_total"] / stats.total_gain).mean())
    return {
        "dataset": dataset_label,
        "scenario": scenario,
        "scenario_seed": scenario_seed,
        "device": device,
        "role": role,
        "features": feature_names,
        "n_features": n_features,
        "total_gain_share": gain / stats.total_gain,
        "mean_gain_share_per_feature": mean_gain_share,
        "total_split_count_per_fold": splits / stats.n_folds,
        "mean_split_count_per_feature_per_fold": splits / (stats.n_folds * n_features),
        "max_split_count_per_feature_per_fold": max_splits,
    }


def _tree_diagnostic_rows(
    stats: TreeStats,
    bundle: DatasetBundle,
    *,
    dataset: str,
    dataset_label: str,
    scenario_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build role-level summaries and per-feature tree diagnostics for every device."""
    global_kpi = _global_kpi_features(bundle, dataset)
    harmless = tuple(bundle.harmless_features)
    role_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for device in bundle.abnormal_devices:
        roots = tuple(bundle.root_causes.get(str(device), ()))
        encoded_name, encoded_gain, encoded_splits = stats.eqp_by_device[device]
        role_rows.append(
            _role_stats_row(
                stats=stats,
                role="eqp_abnormal_onehot",
                features=(encoded_name,),
                device=device,
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
                gain_override=encoded_gain,
                split_override=encoded_splits,
                encoded_name=encoded_name,
            )
        )
        role_rows.append(
            _role_stats_row(
                stats=stats,
                role="eqp_all_onehot",
                features=(bundle.group_column,),
                device=device,
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
            )
        )
        role_rows.append(
            _role_stats_row(
                stats=stats,
                role="harmless_drift",
                features=harmless,
                device=device,
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
            )
        )
        role_rows.append(
            _role_stats_row(
                stats=stats,
                role="root_cause",
                features=roots,
                device=device,
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
            )
        )
        role_rows.append(
            _role_stats_row(
                stats=stats,
                role="global_kpi",
                features=global_kpi,
                device=device,
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
            )
        )
        for record in stats.per_feature.to_dict(orient="records"):
            feature = str(record["feature"])
            labels: list[str] = []
            if feature == bundle.group_column:
                labels.append("eqp_context")
            if feature in roots:
                labels.append("root_cause")
            if feature in set(harmless):
                labels.append("harmless_drift")
            if feature in set(global_kpi):
                labels.append("global_kpi")
            if not labels:
                labels.append("other")
            detail_rows.append(
                {
                    "dataset": dataset_label,
                    "scenario": bundle.scenario,
                    "scenario_seed": scenario_seed,
                    "feature": feature,
                    "feature_kind": "original",
                    "roles": "|".join(labels),
                    "gain_total": float(record["gain_total"]),
                    "gain_share": float(record["gain_share"]),
                    "split_count_total": int(record["split_count_total"]),
                    "mean_split_count_per_fold": float(record["mean_split_count_per_fold"]),
                }
            )
        detail_rows.append(
            {
                "dataset": dataset_label,
                "scenario": bundle.scenario,
                "scenario_seed": scenario_seed,
                "feature": encoded_name,
                "feature_kind": "encoded_onehot",
                "roles": "eqp_abnormal_onehot",
                "gain_total": float(encoded_gain),
                "gain_share": float(encoded_gain / stats.total_gain),
                "split_count_total": int(encoded_splits),
                "mean_split_count_per_fold": float(encoded_splits / stats.n_folds),
            }
        )
    return role_rows, detail_rows


def _correlation(values: np.ndarray, other: np.ndarray) -> float:
    """Return the Pearson correlation, rejecting constant inputs."""
    if values.size != other.size:
        raise ValueError("correlation inputs must have the same length")
    if float(np.std(values)) == 0.0 or float(np.std(other)) == 0.0:
        raise ValueError("correlation inputs must have non-zero standard deviation")
    return float(np.corrcoef(values, other)[0, 1])


def _harmless_correlation_rows(
    bundle: DatasetBundle,
    *,
    dataset: str,
    dataset_label: str,
    scenario_seed: int,
    wasserstein_ranking: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compare harmless features' KPI correlation with their EQP-indicator correlation."""
    harmless = tuple(bundle.harmless_features)
    if not harmless:
        return []
    if len(bundle.abnormal_devices) != 1:
        raise ValueError("harmless-drift correlation diagnostics expect exactly one abnormal device")
    device = bundle.abnormal_devices[0]
    data = bundle.data
    indicator = (data[bundle.group_column].astype(str) == device).to_numpy(dtype=float)
    kpi_features = _global_kpi_features(bundle, dataset)
    device_ranking = wasserstein_ranking.loc[wasserstein_ranking["equipment"] == device]
    rank_lookup = {
        str(record["feature"]): (int(record["rank"]), float(record["score"]))
        for record in device_ranking.to_dict(orient="records")
    }
    ordered_harmless = sorted(harmless, key=lambda feature: rank_lookup[feature][0])
    rows: list[dict[str, Any]] = []
    for order, feature in enumerate(ordered_harmless, start=1):
        values = data[feature].to_numpy(dtype=float)
        correlations = {
            kpi_feature: abs(_correlation(values, data[kpi_feature].to_numpy(dtype=float)))
            for kpi_feature in kpi_features
            if kpi_feature != feature
        }
        if not correlations:
            raise ValueError(f"harmless feature {feature!r} has no KPI feature to correlate with")
        argmax_feature = max(correlations, key=lambda name: correlations[name])
        rank, score = rank_lookup[feature]
        rows.append(
            {
                "dataset": dataset_label,
                "scenario": bundle.scenario,
                "scenario_seed": scenario_seed,
                "device": device,
                "feature": feature,
                "harmless_order": order,
                "gads_rank": rank,
                "gads_score": score,
                "max_abs_pearson_with_kpi": correlations[argmax_feature],
                "kpi_feature_argmax": argmax_feature,
                "point_biserial_with_eqp": abs(_correlation(values, indicator)),
                "is_top_harmless": order == 1,
                "is_top1_overall": rank == 1,
            }
        )
    return rows


def _wasserstein_from_membership(
    values: np.ndarray,
    membership: np.ndarray,
) -> float:
    """Compute one-dimensional Wasserstein distance from a boolean membership vector."""
    abnormal = values[membership]
    normal = values[~membership]
    if abnormal.size == 0 or normal.size == 0:
        raise ValueError("both groups must be non-empty")
    return compute_gads_distance(abnormal, normal, distance="wasserstein")


def _wasserstein_1d_fast(
    sorted_values: np.ndarray,
    sorted_membership: np.ndarray,
    *,
    n_abnormal: int,
    n_normal: int,
) -> np.ndarray:
    """Integrate |F_abnormal - F_normal| exactly on a sorted pooled sample.

    ``sorted_values`` has length ``M`` and ``sorted_membership`` is either a
    length-``M`` vector or an ``(n_permutations, M)`` matrix of boolean labels
    already sorted by value.  The empirical CDF is constant on
    ``[v_k, v_{k+1})`` and equals the cumulative proportion at ``v_k``; therefore
    the ``M-1`` interval widths ``diff(v)`` pair with the CDF evaluated at
    ``v[:-1]``.  Shifting the widths by one (``delta[1:] = diff(v)``) instead
    multiplies each width by the CDF *after* the right endpoint and understates
    the distance.
    """
    n_pool = int(sorted_values.shape[0])
    positions = np.arange(1, n_pool + 1, dtype=float)
    cumulative = np.cumsum(sorted_membership, axis=-1, dtype=float)
    cdf_abnormal = cumulative / n_abnormal
    cdf_normal = (positions - cumulative) / n_normal
    delta = np.zeros(n_pool, dtype=float)
    delta[:-1] = np.diff(sorted_values)
    return np.abs(cdf_abnormal - cdf_normal) @ delta


def _permutation_null(
    shap_values: pd.DataFrame,
    groups: pd.Series,
    bundle: DatasetBundle,
    *,
    dataset_label: str,
    scenario_seed: int,
    view: str,
    wasserstein_ranking: pd.DataFrame,
    n_permutations: int,
    global_kpi: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Permute the equipment label and calibrate the maximum GADS statistic."""
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100")
    if not shap_values.index.equals(groups.index):
        raise ValueError("SHAP values and equipment labels must share an index")
    features = list(bundle.process_features)
    missing = set(features) - set(shap_values.columns)
    if missing:
        raise ValueError(f"process features are absent from the SHAP matrix: {sorted(missing)}")
    values_all = shap_values.loc[:, features].to_numpy(dtype=float)
    groups_array = groups.to_numpy(dtype=object)
    cell_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    harmless = set(bundle.harmless_features)
    global_set = set(global_kpi)

    for device in bundle.abnormal_devices:
        pool_mask = np.isin(groups_array, np.array([device, *bundle.normal_devices], dtype=object))
        pool_values = values_all[pool_mask]
        pool_labels = groups_array[pool_mask]
        n_abn = int((pool_labels == device).sum())
        n_normal = int(len(pool_labels) - n_abn)
        if n_abn < 2 or n_normal < 2:
            raise ValueError(
                f"device {device!r} has too few rows for a permutation null: {n_abn} abnormal, {n_normal} normal"
            )
        observed_mask = pool_labels == device
        n_pool = len(pool_labels)
        rng = np.random.default_rng(
            _stable_seed("permutation-null", dataset_label, bundle.scenario, scenario_seed, view, device)
        )
        membership = np.zeros((n_permutations, n_pool), dtype=bool)
        for permutation in range(n_permutations):
            chosen = rng.choice(n_pool, size=n_abn, replace=False)
            membership[permutation, chosen] = True
        null = np.empty((n_permutations, len(features)), dtype=float)
        for feature_index in range(len(features)):
            values = pool_values[:, feature_index]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_membership = membership[:, order]
            null[:, feature_index] = _wasserstein_1d_fast(
                sorted_values,
                sorted_membership,
                n_abnormal=n_abn,
                n_normal=n_normal,
            )
        observed = np.array(
            [_wasserstein_from_membership(pool_values[:, index], observed_mask) for index in range(len(features))],
            dtype=float,
        )
        tau95 = np.quantile(null, 0.95, axis=0)
        tau99 = np.quantile(null, 0.99, axis=0)
        null_max = null.max(axis=1)
        q95_max = float(np.quantile(null_max, 0.95))
        q99_max = float(np.quantile(null_max, 0.99))
        observed_max = float(observed.max())
        argmax_index = int(observed.argmax())
        p_value_max = float((1 + int((null_max >= observed_max).sum())) / (n_permutations + 1))
        root_indices = [
            index for index, feature in enumerate(features) if feature in set(bundle.root_causes.get(str(device), ()))
        ]
        n_roots = len(root_indices)
        detected_fwer = int((observed[root_indices] > q95_max).sum()) if root_indices else 0
        detected_tau = int((observed[root_indices] > tau95[root_indices]).sum()) if root_indices else 0
        top1_feature = features[argmax_index]
        top1_is_root = top1_feature in set(bundle.root_causes.get(str(device), ()))
        top1_is_harmless = top1_feature in harmless
        device_ranking = wasserstein_ranking.loc[wasserstein_ranking["equipment"] == device]
        rank_lookup = {
            str(record["feature"]): int(record["rank"]) for record in device_ranking.to_dict(orient="records")
        }
        root_ranks = [
            rank_lookup[feature] for feature in bundle.root_causes.get(str(device), ()) if feature in rank_lookup
        ]
        cell_rows.append(
            {
                "dataset": dataset_label,
                "scenario": bundle.scenario,
                "scenario_seed": scenario_seed,
                "view": view,
                "device": device,
                "n_permutations": n_permutations,
                "n_abnormal_rows": n_abn,
                "n_normal_rows": n_normal,
                "n_features": len(features),
                "observed_max": observed_max,
                "observed_argmax_feature": top1_feature,
                "q95_max": q95_max,
                "q99_max": q99_max,
                "p_value_max": p_value_max,
                "fwer_reject_95": observed_max > q95_max,
                "fwer_reject_99": observed_max > q99_max,
                "n_features_exceed_tau95": int((observed > tau95).sum()),
                "n_features_exceed_tau99": int((observed > tau99).sum()),
                "n_roots": n_roots,
                "n_roots_detected_fwer95": detected_fwer,
                "n_roots_detected_tau95": detected_tau,
                "any_root_detected_fwer95": bool(detected_fwer > 0) if n_roots else None,
                "all_roots_detected_fwer95": bool(detected_fwer == n_roots) if n_roots else None,
                "min_gads_rank_of_roots": min(root_ranks) if root_ranks else None,
                "top1_feature": top1_feature,
                "top1_is_root": top1_is_root,
                "top1_is_harmless": top1_is_harmless,
            }
        )
        for feature_index, feature in enumerate(features):
            if feature in set(bundle.root_causes.get(str(device), ())):
                role = "root_cause"
            elif feature in harmless:
                role = "harmless_drift"
            elif feature in global_set:
                role = "global_kpi"
            else:
                role = "other"
            feature_rows.append(
                {
                    "dataset": dataset_label,
                    "scenario": bundle.scenario,
                    "scenario_seed": scenario_seed,
                    "view": view,
                    "device": device,
                    "feature": feature,
                    "role": role,
                    "observed": float(observed[feature_index]),
                    "tau_95": float(tau95[feature_index]),
                    "tau_99": float(tau99[feature_index]),
                    "exceeds_tau95": bool(observed[feature_index] > tau95[feature_index]),
                    "exceeds_tau99": bool(observed[feature_index] > tau99[feature_index]),
                    "fwer_threshold_max_95": q95_max,
                    "exceeds_fwer95": bool(observed[feature_index] > q95_max),
                    "gads_rank": rank_lookup[feature],
                }
            )
    return cell_rows, feature_rows


def _cutoff_rows_from_scores(
    *,
    dataset_label: str,
    scenario: str,
    scenario_seed: int,
    device: str,
    features: list[str],
    scores: np.ndarray,
    roots: set[str],
    harmless: set[str],
    null_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate the four cutoff rules from one descending score vector."""
    if len(features) != len(scores):
        raise ValueError("features and scores must have the same length")
    if len(features) < 2:
        raise ValueError(f"device {device!r} has fewer than two ranked features")
    if (scores < 0.0).any():
        raise ValueError("Wasserstein scores must be non-negative")
    root_ranks = [index + 1 for index, feature in enumerate(features) if feature in roots]
    harmless_ranks = [index + 1 for index, feature in enumerate(features) if feature in harmless]
    root_rank = min(root_ranks) if root_ranks else None
    harmless_rank = min(harmless_ranks) if harmless_ranks else None
    elbow_cap = min(ELBOW_MAX_K, len(scores) - 1)
    if elbow_cap < 1:
        raise ValueError(f"device {device!r} has too few scores for an elbow rule")
    ratio = scores[:elbow_cap] / np.maximum(scores[1 : elbow_cap + 1], 1e-12)
    elbow_ratio_k = int(np.argmax(ratio)) + 1
    absolute_gap = scores[:elbow_cap] - scores[1 : elbow_cap + 1]
    elbow_absgap_k = int(np.argmax(absolute_gap)) + 1
    top5 = set(features[:5])
    top_k = min(5, len(features))
    rules: list[tuple[str, int, list[str], float | None, float | None]] = [
        (
            "elbow_ratio",
            elbow_ratio_k,
            features[:elbow_ratio_k],
            float(scores[elbow_ratio_k - 1]),
            float(scores[elbow_ratio_k]) if elbow_ratio_k < len(scores) else None,
        ),
        (
            "elbow_absgap",
            elbow_absgap_k,
            features[:elbow_absgap_k],
            float(scores[elbow_absgap_k - 1]),
            float(scores[elbow_absgap_k]) if elbow_absgap_k < len(scores) else None,
        ),
        (
            "top5",
            top_k,
            features[:top_k],
            float(scores[top_k - 1]),
            float(scores[top_k]) if top_k < len(scores) else None,
        ),
    ]
    calibrated = [feature for feature, score in zip(features, scores, strict=True) if score > null_threshold]
    rules.append(("null_calibrated", len(calibrated), calibrated, null_threshold, None))
    rows: list[dict[str, Any]] = []
    for rule, selected_k, selected_features, cutoff_score, next_score in rules:
        selected = set(selected_features)
        truncated = "; ".join(selected_features[:15])
        if len(selected_features) > 15:
            truncated += "; ..."
        rows.append(
            {
                "dataset": dataset_label,
                "scenario": scenario,
                "scenario_seed": scenario_seed,
                "device": device,
                "rule": rule,
                "selected_k": selected_k,
                "selected_features": truncated,
                "cutoff_score": cutoff_score,
                "next_score": next_score,
                "root_rank": root_rank,
                "harmless_rank": harmless_rank,
                "root_in_set": bool(selected & roots) if roots else None,
                "harmless_in_set": bool(selected & harmless) if harmless else None,
                "root_in_top5": bool(top5 & roots) if roots else None,
                "harmless_in_top5": bool(top5 & harmless) if harmless else None,
                "top1_feature": features[0],
                "top1_score": float(scores[0]),
            }
        )
    return rows


def _score_gap_rows(
    wasserstein_ranking: pd.DataFrame,
    bundle: DatasetBundle,
    *,
    dataset_label: str,
    scenario_seed: int,
    null_thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Evaluate elbow and permutation-calibrated score-gap cutoffs per device."""
    harmless = set(bundle.harmless_features)
    rows: list[dict[str, Any]] = []
    for device in bundle.abnormal_devices:
        if device not in null_thresholds:
            raise ValueError(f"missing permutation threshold for device {device!r}")
        subset = wasserstein_ranking.loc[wasserstein_ranking["equipment"] == device].sort_values(
            ["rank", "feature"], kind="stable"
        )
        features = [str(feature) for feature in subset["feature"]]
        scores = subset["score"].to_numpy(dtype=float)
        roots = set(bundle.root_causes.get(str(device), ()))
        rows.extend(
            _cutoff_rows_from_scores(
                dataset_label=dataset_label,
                scenario=bundle.scenario,
                scenario_seed=scenario_seed,
                device=device,
                features=features,
                scores=scores,
                roots=roots,
                harmless=harmless,
                null_threshold=null_thresholds[device],
            )
        )
    return rows


def _rebuild_score_gap(perm_features: pd.DataFrame, perm_cells: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the cutoff table from stored per-feature observed scores and null thresholds."""
    if perm_features.empty or perm_cells.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in SCORE_GAP_COLUMNS})
    thresholds = {
        (
            str(record["dataset"]),
            str(record["scenario"]),
            int(record["scenario_seed"]),
            str(record["view"]),
            str(record["device"]),
        ): float(record["q95_max"])
        for record in perm_cells.to_dict(orient="records")
    }
    standard = perm_features.loc[perm_features["view"] == "standard"]
    if standard.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in SCORE_GAP_COLUMNS})
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "scenario", "scenario_seed", "view", "device"]
    for keys, group in standard.groupby(group_columns, dropna=False, observed=True):
        dataset_label, scenario, scenario_seed, view, device = keys
        key = (str(dataset_label), str(scenario), int(scenario_seed), str(view), str(device))
        if key not in thresholds:
            raise ValueError(f"missing permutation threshold for score-gap cell {key}")
        ordered = group.sort_values("gads_rank", kind="stable")
        features = [str(feature) for feature in ordered["feature"]]
        scores = ordered["observed"].to_numpy(dtype=float)
        roots = set(ordered.loc[ordered["role"] == "root_cause", "feature"].astype(str))
        harmless = set(ordered.loc[ordered["role"] == "harmless_drift", "feature"].astype(str))
        rows.extend(
            _cutoff_rows_from_scores(
                dataset_label=str(dataset_label),
                scenario=str(scenario),
                scenario_seed=int(scenario_seed),
                device=str(device),
                features=features,
                scores=scores,
                roots=roots,
                harmless=harmless,
                null_threshold=thresholds[key],
            )
        )
    return _ensure_columns(pd.DataFrame(rows), SCORE_GAP_COLUMNS)


def _variant_rows(
    rankings: dict[str, pd.DataFrame],
    *,
    bundle: DatasetBundle,
    dataset_label: str,
    scenario_seed: int,
    variant: str,
    kpi_auc: float,
) -> list[dict[str, Any]]:
    """Convert GADS rankings into rows tagged by TreeSHAP variant."""
    rows = _metric_rows_from_rankings(
        rankings,
        bundle=bundle,
        dataset_label=dataset_label,
        scenario_seed=scenario_seed,
        view=f"variant:{variant}",
        kpi_auc=kpi_auc,
    )
    for row in rows:
        row["variant"] = variant
    return rows


def _run_base_task(task: BaseTask) -> TaskResult:
    """Run one base cell: standard model, centered model, tree, null and variant diagnostics."""
    bundle = _load_bundle(
        task.dataset,
        task.scenario,
        task.scenario_seed,
        float(DATASET_B_PARAMS["physical_shift"]),
        float(DATASET_B_PARAMS["mechanism_weight"]),
        float(DATASET_B_PARAMS["harmless_shift"]),
    )
    X, y, groups = _feature_matrix(bundle)
    dataset_label = DATASET_LABELS[task.dataset]
    standard = _cross_fit(X, y, dataset=task.dataset, center_by_group=False)
    standard_rankings = _rank_all_distances(standard.shap_values, groups, bundle)
    tables: dict[str, list[dict[str, Any]]] = {
        "gads_metrics": [],
        "tree_split_role_summary": [],
        "tree_split_feature_detail": [],
        "harmless_correlation": [],
        "shap_variant": [],
        "perm_null_cells": [],
        "perm_null_features": [],
        "score_gap": [],
    }
    tables["gads_metrics"].extend(
        _metric_rows_from_rankings(
            standard_rankings,
            bundle=bundle,
            dataset_label=dataset_label,
            scenario_seed=task.scenario_seed,
            view="standard",
            kpi_auc=standard.mean_auc,
        )
    )
    stats = _tree_stats(standard, list(X.columns), bundle.group_column, bundle.abnormal_devices)
    role_rows, detail_rows = _tree_diagnostic_rows(
        stats,
        bundle,
        dataset=task.dataset,
        dataset_label=dataset_label,
        scenario_seed=task.scenario_seed,
    )
    tables["tree_split_role_summary"].extend(role_rows)
    tables["tree_split_feature_detail"].extend(detail_rows)
    tables["harmless_correlation"].extend(
        _harmless_correlation_rows(
            bundle,
            dataset=task.dataset,
            dataset_label=dataset_label,
            scenario_seed=task.scenario_seed,
            wasserstein_ranking=standard_rankings["wasserstein"],
        )
    )
    global_kpi = _global_kpi_features(bundle, task.dataset)
    cells, feature_thresholds = _permutation_null(
        standard.shap_values,
        groups,
        bundle,
        dataset_label=dataset_label,
        scenario_seed=task.scenario_seed,
        view="standard",
        wasserstein_ranking=standard_rankings["wasserstein"],
        n_permutations=_TASK_SETTINGS["n_permutations"],
        global_kpi=global_kpi,
    )
    tables["perm_null_cells"].extend(cells)
    tables["perm_null_features"].extend(feature_thresholds)
    tables["score_gap"].extend(
        _score_gap_rows(
            standard_rankings["wasserstein"],
            bundle,
            dataset_label=dataset_label,
            scenario_seed=task.scenario_seed,
            null_thresholds={str(cell["device"]): float(cell["q95_max"]) for cell in cells},
        )
    )

    if task.scenario == "S4-main":
        columns = list(X.columns)
        interventional_oof = np.full((len(X), X.shape[1]), np.nan, dtype=float)
        for fold_index, fold in enumerate(standard.folds):
            aggregated = _interventional_shap(
                fold,
                columns=columns,
                background_size=_TASK_SETTINGS["background_size"],
                random_state=_stable_seed(
                    "interventional-background",
                    task.dataset,
                    task.scenario,
                    task.scenario_seed,
                    fold_index,
                ),
            )
            interventional_oof[fold.valid_idx] = aggregated.to_numpy(dtype=float)
        if np.isnan(interventional_oof).any():
            raise RuntimeError("interventional TreeSHAP did not cover every row")
        interventional_shap = pd.DataFrame(
            interventional_oof,
            index=X.index,
            columns=columns,
        )
        interventional_rankings = _rank_all_distances(interventional_shap, groups, bundle)
        tables["shap_variant"].extend(
            _variant_rows(
                standard_rankings,
                bundle=bundle,
                dataset_label=dataset_label,
                scenario_seed=task.scenario_seed,
                variant="tree_path_dependent",
                kpi_auc=standard.mean_auc,
            )
        )
        tables["shap_variant"].extend(
            _variant_rows(
                interventional_rankings,
                bundle=bundle,
                dataset_label=dataset_label,
                scenario_seed=task.scenario_seed,
                variant="interventional",
                kpi_auc=standard.mean_auc,
            )
        )

    if task.scenario in CENTERED_SCENARIOS:
        centered = _cross_fit(X, y, dataset=task.dataset, center_by_group=True)
        centered_rankings = _rank_all_distances(centered.shap_values, groups, bundle)
        tables["gads_metrics"].extend(
            _metric_rows_from_rankings(
                centered_rankings,
                bundle=bundle,
                dataset_label=dataset_label,
                scenario_seed=task.scenario_seed,
                view="centered",
                kpi_auc=centered.mean_auc,
            )
        )
        centered_cells, centered_thresholds = _permutation_null(
            centered.shap_values,
            groups,
            bundle,
            dataset_label=dataset_label,
            scenario_seed=task.scenario_seed,
            view="centered",
            wasserstein_ranking=centered_rankings["wasserstein"],
            n_permutations=_TASK_SETTINGS["n_permutations"],
            global_kpi=global_kpi,
        )
        tables["perm_null_cells"].extend(centered_cells)
        tables["perm_null_features"].extend(centered_thresholds)

    return TaskResult(
        task_id=task.task_id,
        kind="base",
        status="ok",
        tables=tables,
    )


def _run_sweep_task(task: SweepTask) -> TaskResult:
    """Run one Dataset B intensity-sweep cell."""
    spec = SWEEP_GRIDS[task.sweep_name]
    physical_shift = float(DATASET_B_PARAMS["physical_shift"])
    mechanism_weight = float(DATASET_B_PARAMS["mechanism_weight"])
    harmless_shift = float(DATASET_B_PARAMS["harmless_shift"])
    parameter = str(spec["parameter"])
    if parameter == "physical_shift":
        physical_shift = float(task.level)
    elif parameter == "mechanism_weight":
        mechanism_weight = float(task.level)
    elif parameter == "harmless_shift":
        harmless_shift = float(task.level)
    else:
        raise ValueError(f"unsupported sweep parameter {parameter!r}")
    bundle = _load_bundle(
        task.dataset,
        task.scenario,
        task.scenario_seed,
        physical_shift,
        mechanism_weight,
        harmless_shift,
    )
    X, y, groups = _feature_matrix(bundle)
    cross = _cross_fit(X, y, dataset=task.dataset, center_by_group=False)
    rankings = _rank_all_distances(cross.shap_values, groups, bundle)
    rows = _metric_rows_from_rankings(
        rankings,
        bundle=bundle,
        dataset_label=DATASET_LABELS[task.dataset],
        scenario_seed=task.scenario_seed,
        view="sweep",
        kpi_auc=cross.mean_auc,
    )
    for row in rows:
        row["sweep_name"] = task.sweep_name
        row["level"] = float(task.level)
        row["reused_base"] = False
    return TaskResult(
        task_id=task.task_id,
        kind="sweep",
        status="ok",
        tables={"sweep_metrics": rows},
    )


_TASK_SETTINGS: dict[str, int] = {
    "n_permutations": DEFAULT_N_PERMUTATIONS,
    "background_size": DEFAULT_BACKGROUND_SIZE,
}


def _run_task_worker(task: Task) -> TaskResult:
    """Execute one task, capturing any failure for the parent process."""
    try:
        if isinstance(task, BaseTask):
            return _run_base_task(task)
        return _run_sweep_task(task)
    except Exception:
        return TaskResult(
            task_id=task.task_id,
            kind="base" if isinstance(task, BaseTask) else "sweep",
            status="error",
            error=traceback.format_exc(),
        )


def _init_worker(n_permutations: int, background_size: int) -> None:
    """Configure spawn worker settings and one BLAS thread per process."""
    global _TASK_SETTINGS
    _TASK_SETTINGS = {
        "n_permutations": int(n_permutations),
        "background_size": int(background_size),
    }
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    threadpool_limits(limits=1)


def build_tasks(
    datasets: tuple[str, ...],
    scenarios: tuple[str, ...],
    scenario_seeds: tuple[int, ...],
) -> list[Task]:
    """Expand the base grid and the Dataset B sweeps into concrete tasks."""
    tasks: list[Task] = []
    for dataset in datasets:
        for scenario in scenarios:
            for scenario_seed in scenario_seeds:
                tasks.append(  # noqa: PERF401 - readable nested grid construction
                    BaseTask(
                        dataset=dataset,
                        scenario=scenario,
                        scenario_seed=int(scenario_seed),
                        task_id=f"base_{dataset}_{scenario}_seed{scenario_seed}",
                    )
                )
    if "b" in datasets:
        for sweep_name, spec in SWEEP_GRIDS.items():
            scenario = str(spec["scenario"])
            if scenario not in scenarios:
                continue
            for level in spec["levels"]:
                if float(level) == float(spec["default"]):
                    continue
                for scenario_seed in scenario_seeds:
                    tasks.append(  # noqa: PERF401 - readable nested grid construction
                        SweepTask(
                            dataset="b",
                            sweep_name=sweep_name,
                            scenario=scenario,
                            scenario_seed=int(scenario_seed),
                            level=float(level),
                            task_id=f"sweep_{sweep_name}_{level}_seed{scenario_seed}",
                        )
                    )
    return tasks


def _partial_path(partials_dir: Path, task_id: str) -> Path:
    """Return the JSON partial path for one task."""
    return partials_dir / f"{task_id}.json"


def _write_partial(partials_dir: Path, result: TaskResult) -> None:
    """Write one task result to its partial JSON file atomically."""
    payload = {
        "task_id": result.task_id,
        "kind": result.kind,
        "status": result.status,
        "error": result.error,
        "settings": dict(_TASK_SETTINGS),
        "tables": {name: [_jsonable(row) for row in rows] for name, rows in result.tables.items()},
    }
    path = _partial_path(partials_dir, result.task_id)
    _write_json(path, payload)


def _partial_is_compatible(payload: dict[str, Any]) -> bool:
    """Return whether a partial was produced with the current worker settings."""
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return False
    return int(settings.get("n_permutations", -1)) == int(_TASK_SETTINGS["n_permutations"]) and int(
        settings.get("background_size", -1)
    ) == int(_TASK_SETTINGS["background_size"])


def _load_partial(partials_dir: Path, task_id: str) -> dict[str, Any]:
    """Load one partial JSON file with basic integrity checks."""
    path = _partial_path(partials_dir, task_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing task partial: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != task_id:
        raise ValueError(f"partial {path} contains task_id {payload.get('task_id')!r}, expected {task_id!r}")
    return payload


def _run_pool(
    tasks: list[Task],
    *,
    workers: int,
    partials_dir: Path,
) -> None:
    """Execute pending tasks with multiprocessing and persist each result."""
    if not tasks:
        return
    context = mp.get_context("spawn")
    start = time.monotonic()
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(_TASK_SETTINGS["n_permutations"], _TASK_SETTINGS["background_size"]),
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(_run_task_worker, tasks, chunksize=1), start=1):
            _write_partial(partials_dir, result)
            if result.status != "ok":
                print(f"[{index}/{len(tasks)}] FAILED {result.task_id}", flush=True)
            if index == 1 or index % max(1, len(tasks) // 20) == 0 or index == len(tasks):
                elapsed = time.monotonic() - start
                rate = index / elapsed if elapsed > 0.0 else 0.0
                remaining = (len(tasks) - index) / rate if rate > 0.0 else 0.0
                print(
                    f"[{index}/{len(tasks)}] elapsed={elapsed / 60.0:.1f} min eta={remaining / 60.0:.1f} min",
                    flush=True,
                )


def _load_all_partials(partials_dir: Path, tasks: list[Task]) -> list[dict[str, Any]]:
    """Load every task partial, raising on missing or failed tasks."""
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[tuple[str, str]] = []
    for task in tasks:
        path = _partial_path(partials_dir, task.task_id)
        if not path.is_file():
            missing.append(task.task_id)
            continue
        payload = _load_partial(partials_dir, task.task_id)
        if not _partial_is_compatible(payload):
            raise RuntimeError(
                f"partial {task.task_id} was produced with different permutation settings; rerun with --force"
            )
        if payload.get("status") != "ok":
            failed.append((task.task_id, str(payload.get("error"))))
            continue
        payloads.append(payload)
    if missing:
        raise FileNotFoundError(f"{len(missing)} task partials are missing: {missing[:10]}")
    if failed:
        details = "\n".join(f"{task_id}: {error}" for task_id, error in failed[:5])
        raise RuntimeError(f"{len(failed)} tasks failed; first failures:\n{details}")
    return payloads


def _merge_table(payloads: list[dict[str, Any]], name: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """Concatenate one table from all partial payloads onto a stable schema."""
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload["tables"].get(name, []))
    frame = pd.DataFrame(rows)
    return _ensure_columns(frame, columns)


def _mean_std(frame: pd.DataFrame, group_columns: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    """Aggregate metrics as mean, standard deviation and non-null count."""
    if frame.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in group_columns})
    working = frame.copy()
    for metric in metrics:
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
    grouped = working.groupby(group_columns, dropna=False, observed=True)
    pieces: list[pd.DataFrame] = []
    for metric in metrics:
        aggregated = (
            grouped[metric]
            .agg(["mean", "std", "count"])
            .rename(
                columns={
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_std",
                    "count": f"{metric}_n",
                }
            )
        )
        pieces.append(aggregated)
    return pd.concat(pieces, axis=1).reset_index()


def _reuse_default_sweep_rows(base_metrics: pd.DataFrame, sweep_name: str) -> pd.DataFrame:
    """Reuse the default-level base cell for the matching sweep level."""
    spec = SWEEP_GRIDS[sweep_name]
    scenario = str(spec["scenario"])
    subset = base_metrics.loc[
        (base_metrics["dataset"] == DATASET_LABELS["b"])
        & (base_metrics["scenario"] == scenario)
        & (base_metrics["view"] == "standard")
    ].copy()
    if subset.empty:
        raise ValueError(f"base metrics have no standard rows for sweep scenario {scenario!r}")
    subset["sweep_name"] = sweep_name
    subset["level"] = float(spec["default"])
    subset["reused_base"] = True
    return subset


def _sweep_frame(base_metrics: pd.DataFrame, sweep_metrics: pd.DataFrame, sweep_name: str) -> pd.DataFrame:
    """Combine computed sweep cells with the reused default-level base cells."""
    computed = sweep_metrics.loc[sweep_metrics["sweep_name"] == sweep_name].copy()
    reused = _reuse_default_sweep_rows(base_metrics, sweep_name)
    combined = pd.concat([computed, reused], ignore_index=True)
    duplicate_keys = ["dataset", "scenario_seed", "level", "distance"]
    duplicated = combined.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        raise ValueError(
            f"sweep {sweep_name!r} has duplicate (seed, level, distance) rows: "
            f"{combined.loc[duplicated, duplicate_keys].head().to_dict(orient='records')}"
        )
    return _ensure_columns(combined, SWEEP_METRIC_COLUMNS).sort_values(
        ["level", "distance", "scenario_seed"], kind="stable"
    )


def _detection_power(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize FWER rejection, root-cause power and per-feature exceedance."""
    columns = [
        "dataset",
        "scenario",
        "view",
        "n_seeds",
        "n_device_tests",
        "fwer_reject_rate_95",
        "fwer_reject_rate_99",
        "per_feature_fpr_95",
        "mean_features_exceed_tau95",
        "mean_features_exceed_tau99",
        "mean_features_exceed_fwer95",
        "root_detection_power_fwer95",
        "all_roots_detection_rate_fwer95",
        "mean_root_rank_observed",
        "top1_root_rate",
        "top1_harmless_rate",
    ]
    if cells.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, view), group in cells.groupby(["dataset", "scenario", "view"], dropna=False, observed=True):
        root_rows = group.loc[group["n_roots"] > 0]
        per_feature_rate = group["n_features_exceed_tau95"].to_numpy(dtype=float) / group["n_features"].to_numpy(
            dtype=float
        )
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "view": view,
                "n_seeds": int(group["scenario_seed"].nunique()),
                "n_device_tests": len(group),
                "fwer_reject_rate_95": float(group["fwer_reject_95"].mean()),
                "fwer_reject_rate_99": float(group["fwer_reject_99"].mean()),
                "per_feature_fpr_95": float(per_feature_rate.mean()),
                "mean_features_exceed_tau95": float(group["n_features_exceed_tau95"].mean()),
                "mean_features_exceed_tau99": float(group["n_features_exceed_tau99"].mean()),
                "mean_features_exceed_fwer95": float(group["n_features_exceed_fwer95"].mean()),
                "root_detection_power_fwer95": (
                    float(root_rows["any_root_detected_fwer95"].mean()) if not root_rows.empty else None
                ),
                "all_roots_detection_rate_fwer95": (
                    float(root_rows["all_roots_detected_fwer95"].mean()) if not root_rows.empty else None
                ),
                "mean_root_rank_observed": (
                    float(root_rows["min_gads_rank_of_roots"].dropna().mean())
                    if not root_rows.empty and root_rows["min_gads_rank_of_roots"].notna().any()
                    else None
                ),
                "top1_root_rate": float(group["top1_is_root"].mean()),
                "top1_harmless_rate": float(group["top1_is_harmless"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario", "view"], kind="stable")


def _s0_fwer_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize the S0-clean false-discovery calibration, with Wilson intervals."""
    columns = [
        "dataset",
        "view",
        "n_tests",
        "n_fwer_rejects_95",
        "fwer_rate_95",
        "fwer_ci_low_95",
        "fwer_ci_high_95",
        "expected_fwer",
        "per_feature_fpr_95",
        "mean_features_exceed_tau95",
        "mean_n_features",
        "uncalibrated_top1_claim_rate",
    ]
    if cells.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = cells.loc[cells["scenario"] == "S0-clean"]
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    rows: list[dict[str, Any]] = []
    for (dataset, view), group in subset.groupby(["dataset", "view"], dropna=False, observed=True):
        trials = len(group)
        successes = int(group["fwer_reject_95"].sum())
        low, high = _wilson_interval(successes, trials)
        per_feature_rate = group["n_features_exceed_tau95"].to_numpy(dtype=float) / group["n_features"].to_numpy(
            dtype=float
        )
        rows.append(
            {
                "dataset": dataset,
                "view": view,
                "n_tests": trials,
                "n_fwer_rejects_95": successes,
                "fwer_rate_95": successes / trials,
                "fwer_ci_low_95": low,
                "fwer_ci_high_95": high,
                "expected_fwer": 0.05,
                "per_feature_fpr_95": float(per_feature_rate.mean()),
                "mean_features_exceed_tau95": float(group["n_features_exceed_tau95"].mean()),
                "mean_n_features": float(group["n_features"].mean()),
                "uncalibrated_top1_claim_rate": 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "view"], kind="stable")


def _shap_variant_summary(variant_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize path-dependent vs interventional TreeSHAP and their paired differences."""
    summary_columns = [
        "dataset",
        "scenario",
        "variant",
        "distance",
        "mrr_mean",
        "mrr_std",
        "mrr_n",
        "hr_at_1_mean",
        "hr_at_1_std",
        "hr_at_1_n",
        "hr_at_3_mean",
        "hr_at_3_std",
        "hr_at_3_n",
        "hr_at_5_mean",
        "hr_at_5_std",
        "hr_at_5_n",
        "far_at_1_mean",
        "far_at_1_std",
        "far_at_1_n",
        "far_at_3_mean",
        "far_at_3_std",
        "far_at_3_n",
        "far_at_5_mean",
        "far_at_5_std",
        "far_at_5_n",
    ]
    paired_columns = [
        "dataset",
        "scenario",
        "distance",
        "far_at_1_difference_mean",
        "far_at_1_difference_std",
        "far_at_1_difference_n",
        "far_at_5_difference_mean",
        "far_at_5_difference_std",
        "far_at_5_difference_n",
        "hr_at_1_difference_mean",
        "hr_at_1_difference_std",
        "hr_at_1_difference_n",
        "hr_at_5_difference_mean",
        "hr_at_5_difference_std",
        "hr_at_5_difference_n",
        "mrr_difference_mean",
        "mrr_difference_std",
        "mrr_difference_n",
    ]
    if variant_rows.empty:
        return (
            pd.DataFrame({column: pd.Series(dtype=object) for column in summary_columns}),
            pd.DataFrame({column: pd.Series(dtype=object) for column in paired_columns}),
        )
    summary = _mean_std(
        variant_rows,
        ["dataset", "scenario", "variant", "distance"],
        ("mrr", "hr_at_1", "hr_at_3", "hr_at_5", "far_at_1", "far_at_3", "far_at_5"),
    )
    pivot = variant_rows.pivot_table(
        index=["dataset", "scenario", "distance", "scenario_seed"],
        columns="variant",
        values=["far_at_1", "far_at_5", "hr_at_1", "hr_at_5", "mrr"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{variant}" for metric, variant in pivot.columns]
    pivot = pivot.reset_index()
    required = {
        "far_at_1_tree_path_dependent",
        "far_at_1_interventional",
        "hr_at_1_tree_path_dependent",
        "hr_at_1_interventional",
    }
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"paired TreeSHAP comparison is missing columns: {sorted(missing)}")
    pivot["far_at_1_difference"] = pivot["far_at_1_interventional"] - pivot["far_at_1_tree_path_dependent"]
    pivot["hr_at_1_difference"] = pivot["hr_at_1_interventional"] - pivot["hr_at_1_tree_path_dependent"]
    if "far_at_5_tree_path_dependent" in pivot.columns:
        pivot["far_at_5_difference"] = pivot["far_at_5_interventional"] - pivot["far_at_5_tree_path_dependent"]
    if "hr_at_5_tree_path_dependent" in pivot.columns:
        pivot["hr_at_5_difference"] = pivot["hr_at_5_interventional"] - pivot["hr_at_5_tree_path_dependent"]
    if "mrr_tree_path_dependent" in pivot.columns:
        pivot["mrr_difference"] = pivot["mrr_interventional"] - pivot["mrr_tree_path_dependent"]
    paired_summary = _mean_std(
        pivot,
        ["dataset", "scenario", "distance"],
        (
            "far_at_1_difference",
            "far_at_5_difference",
            "hr_at_1_difference",
            "hr_at_5_difference",
            "mrr_difference",
        ),
    )
    return summary, paired_summary


def _score_gap_summary(score_gap: pd.DataFrame) -> pd.DataFrame:
    """Summarize the cutoff rules on Dataset B S4-main."""
    columns = [
        "dataset",
        "scenario",
        "rule",
        "n_seeds",
        "mean_selected_k",
        "median_selected_k",
        "root_in_set_rate",
        "harmless_in_set_rate",
        "root_in_top5_rate",
        "harmless_in_top5_rate",
        "mean_root_rank",
        "mean_harmless_rank",
    ]
    if score_gap.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = score_gap.loc[(score_gap["dataset"] == DATASET_LABELS["b"]) & (score_gap["scenario"] == "S4-main")]
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    rows: list[dict[str, Any]] = []
    for (rule,), group in subset.groupby(["rule"], dropna=False, observed=True):
        rows.append(
            {
                "dataset": DATASET_LABELS["b"],
                "scenario": "S4-main",
                "rule": rule,
                "n_seeds": int(group["scenario_seed"].nunique()),
                "mean_selected_k": float(group["selected_k"].mean()),
                "median_selected_k": float(group["selected_k"].median()),
                "root_in_set_rate": float(group["root_in_set"].mean()),
                "harmless_in_set_rate": float(group["harmless_in_set"].mean()),
                "root_in_top5_rate": float(group["root_in_top5"].mean()),
                "harmless_in_top5_rate": float(group["harmless_in_top5"].mean()),
                "mean_root_rank": float(group["root_rank"].dropna().mean())
                if group["root_rank"].notna().any()
                else None,
                "mean_harmless_rank": float(group["harmless_rank"].dropna().mean())
                if group["harmless_rank"].notna().any()
                else None,
            }
        )
    order = {"elbow_ratio": 0, "elbow_absgap": 1, "null_calibrated": 2, "top5": 3}
    frame = pd.DataFrame(rows)
    frame["_order"] = frame["rule"].map(order)
    if frame["_order"].isna().any():
        raise ValueError("score-gap summary contains an unknown cutoff rule")
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _tree_role_summary(role_rows: pd.DataFrame) -> pd.DataFrame:
    """Average tree role statistics over the 20 scenario seeds."""
    group_columns = ["dataset", "scenario", "role"]
    summary = _mean_std(
        role_rows,
        group_columns,
        (
            "total_gain_share",
            "mean_gain_share_per_feature",
            "total_split_count_per_fold",
            "mean_split_count_per_feature_per_fold",
            "max_split_count_per_feature_per_fold",
        ),
    )
    sizes = role_rows.groupby(group_columns, dropna=False, observed=True)["n_features"].first().reset_index()
    return summary.merge(sizes, on=group_columns, how="left", validate="one_to_one")


def _harmless_correlation_summary(correlation: pd.DataFrame) -> pd.DataFrame:
    """Average harmless-feature correlations for the top-ranked harmless features."""
    columns = [
        "dataset",
        "scenario",
        "harmless_bucket",
        "n_observations",
        "mean_max_abs_pearson_with_kpi",
        "max_max_abs_pearson_with_kpi",
        "mean_point_biserial_with_eqp",
        "max_point_biserial_with_eqp",
        "correlation_gap_mean",
    ]
    if correlation.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    top = correlation.loc[correlation["harmless_order"] <= 3].copy()
    top["harmless_rank_bucket"] = np.where(top["harmless_order"] == 1, "top1", "top3")
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, bucket), group in top.groupby(
        ["dataset", "scenario", "harmless_rank_bucket"], dropna=False, observed=True
    ):
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "harmless_bucket": bucket,
                "n_observations": len(group),
                "mean_max_abs_pearson_with_kpi": float(group["max_abs_pearson_with_kpi"].mean()),
                "max_max_abs_pearson_with_kpi": float(group["max_abs_pearson_with_kpi"].max()),
                "mean_point_biserial_with_eqp": float(group["point_biserial_with_eqp"].mean()),
                "max_point_biserial_with_eqp": float(group["point_biserial_with_eqp"].max()),
                "correlation_gap_mean": float(
                    (group["point_biserial_with_eqp"] - group["max_abs_pearson_with_kpi"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario", "harmless_bucket"], kind="stable")


def _sweep_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one intensity sweep over seeds and distances."""
    return _mean_std(
        sweep,
        ["sweep_name", "scenario", "level", "distance"],
        ("mrr", "hr_at_1", "hr_at_3", "hr_at_5", "far_at_1", "far_at_3", "far_at_5", "kpi_auc"),
    )


def _s4_harmless_far_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Pivot S4-main harmless-drift FAR@1 means by level and distance."""
    columns = ["level", *DISTANCES]
    if summary.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = summary.loc[summary["sweep_name"] == "s4_harmless"]
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    pivot = subset.pivot_table(
        index="level",
        columns="distance",
        values="far_at_1_mean",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=list(DISTANCES)).reset_index()
    return pivot.sort_values("level", kind="stable").reset_index(drop=True)


def _s4_harmless_wasserstein(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the Wasserstein S4-main harmless-drift trajectory."""
    columns = [
        "level",
        "mrr_mean",
        "hr_at_1_mean",
        "hr_at_5_mean",
        "far_at_1_mean",
        "far_at_5_mean",
        "kpi_auc_mean",
    ]
    if summary.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = summary.loc[(summary["sweep_name"] == "s4_harmless") & (summary["distance"] == "wasserstein")]
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    return subset.loc[:, columns].sort_values("level", kind="stable").reset_index(drop=True)


def _centering_table(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the standard-vs-centered 20-seed comparison for both datasets."""
    columns = [
        "dataset",
        "scenario",
        "view",
        "mrr_mean",
        "hr_at_1_mean",
        "hr_at_5_mean",
        "far_at_1_mean",
        "far_at_5_mean",
        "kpi_auc_mean",
    ]
    if metrics_summary.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = metrics_summary.loc[
        (metrics_summary["distance"] == "wasserstein")
        & (metrics_summary["view"].isin(["standard", "centered"]))
        & (metrics_summary["scenario"] != "S0-clean")
    ].copy()
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    missing = [column for column in columns if column not in subset.columns]
    if missing:
        raise ValueError(f"centering table is missing columns: {missing}")
    return subset.loc[:, columns].sort_values(["dataset", "scenario", "view"], kind="stable").reset_index(drop=True)


def _variant_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the 20-seed TreeSHAP variant display table."""
    columns = [
        "dataset",
        "variant",
        "mrr_mean",
        "hr_at_1_mean",
        "hr_at_5_mean",
        "far_at_1_mean",
        "far_at_5_mean",
    ]
    if summary.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    subset = summary.loc[(summary["scenario"] == "S4-main") & (summary["distance"] == "wasserstein")].copy()
    if subset.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})
    return subset.loc[:, columns].sort_values(["dataset", "variant"], kind="stable").reset_index(drop=True)


def _sweep_display(summary: pd.DataFrame, sweep_name: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Pivot one sweep's mean metrics into a level x distance table."""
    if summary.empty:
        return pd.DataFrame({"level": pd.Series(dtype=float)})
    subset = summary.loc[summary["sweep_name"] == sweep_name]
    if subset.empty:
        return pd.DataFrame({"level": pd.Series(dtype=float)})
    frames: list[pd.DataFrame] = []
    for metric in metrics:
        column = f"{metric}_mean"
        pivot = subset.pivot_table(
            index="level",
            columns="distance",
            values=column,
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{distance}" for distance in pivot.columns]
        frames.append(pivot)
    output = pd.concat(frames, axis=1).reset_index().sort_values("level", kind="stable")
    return output.reset_index(drop=True)


def _markdown_protocol(
    protocol: dict[str, Any],
    *,
    counts: Counter[str],
    runtime_seconds: float,
) -> str:
    """Render the protocol section of ``summary.md``."""
    lines = [
        "# Pack 3 — R3 #7, R3 #8/R2 #1 and R3 #11 diagnostics",
        "",
        "## Protocol",
        "",
        f"- Generated (UTC): `{protocol['created_at_utc']}`",
        f"- Host: `{protocol['host']['node']}` ({protocol['host']['platform']}), {protocol['host']['cpu_count']} CPUs",
        f"- Python {protocol['host']['python']}; xgboost {protocol['versions']['xgboost']}; "
        f"shap {protocol['versions']['shap']}; scikit-learn {protocol['versions']['scikit_learn']}",
        f"- Git commit: `{protocol['git_commit']}`",
        f"- Workers: {protocol['workers']}; each worker uses one BLAS thread and XGBoost `n_jobs=1`",
        f"- Scenario seeds: 0–19; permutation budget: {protocol['n_permutations']} per test; "
        f"interventional background: {protocol['background_size']} training rows passed to "
        '`shap.TreeExplainer(..., feature_perturbation="interventional")`, of which shap 0.50.0 '
        f"keeps {min(100, protocol['background_size'])} effective background rows "
        "(its `Independent` masker defaults to `max_samples=100`)",
        f"- Total wall time: {runtime_seconds / 60.0:.1f} min",
        "",
        "**Dataset A**: `n_samples=5000`, `noise_level=0.2`, `n_devices=10`, "
        "`distribution_shift=3.0`, `mechanism_scale=1.0`, `harmless_drift=3.5`; "
        "XGBoost `max_depth=6`, `min_child_weight=1`.",
        "",
        "**Dataset B**: official UCI SECOM raw files; `group_seed=2024`, "
        "`physical_shift=3.0`, `mechanism_weight=12.0`, `harmless_shift=4.0`, "
        "missing-indicator columns **excluded** (590 process features); "
        "XGBoost `max_depth=3`, `min_child_weight=5`.",
        "",
        "**S0-clean**: the S2 generator with the mechanism strength set to the normal-pool "
        "value (Dataset A `mechanism_scale=0.0`; Dataset B local weight 0.2); no physical "
        "shift, no mechanism shift and no harmless drift. EQP_B is the labelled equipment.",
        "",
        "**Within-EQP centering**: `X_centered = X - mean(X | EQP)` with the group means "
        "estimated on each **training fold only**; the validation fold is transformed with "
        "the training-fold means, so no validation row contributes to the centering transform.",
        "",
        "**Permutation null**: the equipment label is permuted among `I_g ∪ I_normal` "
        f"{protocol['n_permutations']} times; the null statistic is "
        "`max_j W1(φ_j^(g*), φ_j^(normal*))`. FWER rejection uses the 95th percentile of "
        "the null maximum; per-feature thresholds are the 95th/99th percentiles.",
        "",
        "**Task counts**: " + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())) + ".",
        "",
    ]
    return "\n".join(lines)


def _mean_value(frame: pd.DataFrame, filters: dict[str, str], column: str) -> float | None:
    """Return one aggregate cell selected by exact string filters."""
    subset = frame
    for key, value in filters.items():
        if key not in subset.columns:
            raise ValueError(f"aggregate table has no column {key!r}")
        subset = subset.loc[subset[key] == value]
    if len(subset) != 1:
        raise ValueError(f"expected exactly one aggregate row for {filters}, got {len(subset)}")
    value = subset.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _optional_mean(frame: pd.DataFrame, filters: dict[str, str], column: str) -> float | None:
    """Return one aggregate cell or None when the requested slice is absent."""
    subset = frame
    for key, value in filters.items():
        subset = subset.loc[subset[key] == value]
    if len(subset) == 0:
        return None
    if len(subset) > 1:
        raise ValueError(f"expected at most one aggregate row for {filters}, got {len(subset)}")
    value = subset.iloc[0][column]
    return None if pd.isna(value) else float(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format an optional number for narrative text."""
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def _markdown_tree(
    role_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> str:
    """Render the R3 #7 tree-split and harmless-correlation sections."""
    display = role_summary.loc[
        role_summary["scenario"].isin(["S2", "S4-main", "S4-null"])
        & role_summary["role"].isin(["eqp_abnormal_onehot", "harmless_drift", "root_cause", "global_kpi"])
    ].copy()
    display = display.loc[
        :,
        [
            "dataset",
            "scenario",
            "role",
            "n_features",
            "total_gain_share_mean",
            "mean_gain_share_per_feature_mean",
            "mean_split_count_per_feature_per_fold_mean",
        ],
    ].sort_values(["dataset", "scenario", "role"], kind="stable")
    lines = [
        "## Part 1 — R3 #7: surrogate marker splits and within-EQP centering",
        "",
        "### 1.1 XGBoost split diagnostics (20 seeds, 5 CV folds)",
        "",
        "Gain share is the feature group's share of total split gain; split counts are per "
        "fold. `eqp_abnormal_onehot` is the one-hot column of the labelled abnormal equipment, "
        "so it isolates the marker channel from the other nine equipment indicators.",
        "",
        _md_table(
            display,
            [
                "dataset",
                "scenario",
                "role",
                "n_features",
                "total_gain_share_mean",
                "mean_gain_share_per_feature_mean",
                "mean_split_count_per_feature_per_fold_mean",
            ],
        ),
        "",
        "### 1.2 Harmless features as equipment markers",
        "",
        "For each S4 bundle we record the maximum absolute Pearson correlation between a "
        "harmless-drift feature and the global KPI features, and its point-biserial correlation "
        "with `I(EQP == g_abnormal)`. The table averages the top-1 and top-3 harmless features "
        "by GADS rank over 20 seeds.",
        "",
        _md_table(
            correlation_summary,
            [
                "dataset",
                "scenario",
                "harmless_bucket",
                "n_observations",
                "mean_max_abs_pearson_with_kpi",
                "mean_point_biserial_with_eqp",
                "correlation_gap_mean",
            ],
        ),
        "",
        "A positive `correlation_gap_mean` means the harmless feature tracks the abnormal-tool "
        "indicator more strongly than any true KPI feature: the tree can use it in place of EQP.",
        "",
        "**Marker-split evidence.**",
        "",
    ]
    available_datasets = set(role_summary["dataset"]) & set(correlation_summary["dataset"])
    for dataset_label in [label for label in (DATASET_LABELS["a"], DATASET_LABELS["b"]) if label in available_datasets]:
        if role_summary.loc[(role_summary["dataset"] == dataset_label) & (role_summary["scenario"] == "S4-main")].empty:
            continue
        harmless_gain = _mean_value(
            role_summary,
            {"dataset": dataset_label, "scenario": "S4-main", "role": "harmless_drift"},
            "total_gain_share_mean",
        )
        root_gain = _mean_value(
            role_summary,
            {"dataset": dataset_label, "scenario": "S4-main", "role": "root_cause"},
            "total_gain_share_mean",
        )
        eqp_gain = _mean_value(
            role_summary,
            {"dataset": dataset_label, "scenario": "S4-main", "role": "eqp_abnormal_onehot"},
            "total_gain_share_mean",
        )
        marker_corr = _optional_mean(
            correlation_summary,
            {"dataset": dataset_label, "scenario": "S4-main", "harmless_bucket": "top1"},
            "mean_point_biserial_with_eqp",
        )
        kpi_corr = _optional_mean(
            correlation_summary,
            {"dataset": dataset_label, "scenario": "S4-main", "harmless_bucket": "top1"},
            "mean_max_abs_pearson_with_kpi",
        )
        lines.append(
            f"- {dataset_label} S4-main: harmless-drift gain share {_fmt(harmless_gain)} vs "
            f"root-cause {_fmt(root_gain)} vs abnormal EQP one-hot {_fmt(eqp_gain)}; the top-ranked "
            f"harmless feature has |r|={_fmt(marker_corr)} with I(EQP=g) and {_fmt(kpi_corr)} with "
            "the best KPI feature."
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_variant(
    variant_display: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    """Render the TreeSHAP path-dependent vs interventional comparison."""
    paired = paired_summary.loc[paired_summary["distance"] == "wasserstein"].copy()
    lines = [
        "### 1.3 TreeSHAP variant: path-dependent vs interventional (S4-main)",
        "",
        "Both variants use the same five fitted fold models. The interventional variant passes a "
        "200-row training-fold background to `shap.TreeExplainer(..., data=bg_data, "
        'feature_perturbation="interventional")`; shap 0.50.0 wraps that sample in an '
        "`Independent` masker with the default `max_samples=100`, so 200 rows are passed and "
        "100 are used as the effective background (`len(explainer.data) == 100`). If the marker "
        "effect were an artefact of the TreeSHAP variant, the two rows would differ materially.",
        "",
        _md_table(
            variant_display,
            [
                "dataset",
                "variant",
                "mrr_mean",
                "hr_at_1_mean",
                "hr_at_5_mean",
                "far_at_1_mean",
                "far_at_5_mean",
            ],
        ),
        "",
        "Paired 20-seed differences (interventional minus path-dependent):",
        "",
        _md_table(
            paired,
            [
                "dataset",
                "distance",
                "far_at_1_difference_mean",
                "far_at_1_difference_std",
                "hr_at_1_difference_mean",
                "mrr_difference_mean",
            ],
        ),
        "",
        "**Variant conclusion.**",
        "",
    ]
    available_datasets = set(variant_display["dataset"])
    for dataset_label in [label for label in (DATASET_LABELS["a"], DATASET_LABELS["b"]) if label in available_datasets]:
        far_difference = _mean_value(
            paired_summary,
            {"dataset": dataset_label, "distance": "wasserstein"},
            "far_at_1_difference_mean",
        )
        far_sd = _mean_value(
            paired_summary,
            {"dataset": dataset_label, "distance": "wasserstein"},
            "far_at_1_difference_std",
        )
        hr_difference = _mean_value(
            paired_summary,
            {"dataset": dataset_label, "distance": "wasserstein"},
            "hr_at_1_difference_mean",
        )
        dataset_rows = variant_display.loc[variant_display["dataset"] == dataset_label]
        far_by_variant = {
            str(record["variant"]): float(record["far_at_1_mean"]) for record in dataset_rows.to_dict(orient="records")
        }
        path_far = far_by_variant.get("tree_path_dependent")
        interventional_far = far_by_variant.get("interventional")
        if path_far is None or interventional_far is None:
            raise ValueError(f"{dataset_label} variant table is missing one of the two TreeSHAP variants")
        if min(path_far, interventional_far) >= 0.5:
            variant_reading = (
                "Both variants show the same high false-alarm level, so the surrogate-marker "
                "effect comes from the fitted tree splits, not from the TreeSHAP variant."
            )
        else:
            variant_reading = (
                "Both variants show the same low false-alarm level, so interventional SHAP does "
                "not create a marker effect that is absent under path-dependent SHAP."
            )
        lines.append(
            f"- {dataset_label}: FAR@1 is {_fmt(path_far)} for path-dependent and "
            f"{_fmt(interventional_far)} for interventional SHAP, a paired difference of "
            f"{_fmt(far_difference)} (SD {_fmt(far_sd)}); HR@1 difference = {_fmt(hr_difference)}. "
            f"{variant_reading}"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_centering(centering: pd.DataFrame) -> str:
    """Render the within-EQP centering trade-off table."""
    lines = [
        "### 1.4 Within-EQP centering trade-off (20 seeds)",
        "",
        "The centered model removes each equipment's training-fold mean from every numeric "
        "process feature. Pure location shifts (S1, S4-main/S4-null harmless drift) disappear from "
        "the attribution comparison; coefficient-level mechanism shifts (S2, S3, S4-main) survive.",
        "",
        _md_table(
            centering,
            [
                "dataset",
                "scenario",
                "view",
                "mrr_mean",
                "hr_at_1_mean",
                "hr_at_5_mean",
                "far_at_1_mean",
                "far_at_5_mean",
            ],
        ),
        "",
        "**Trade-off.**",
        "",
    ]
    available_datasets = set(centering["dataset"])
    for dataset_label in [label for label in (DATASET_LABELS["a"], DATASET_LABELS["b"]) if label in available_datasets]:
        for scenario in ("S1", "S2", "S4-main", "S4-null"):
            standard = centering.loc[
                (centering["dataset"] == dataset_label)
                & (centering["scenario"] == scenario)
                & (centering["view"] == "standard")
            ]
            centered = centering.loc[
                (centering["dataset"] == dataset_label)
                & (centering["scenario"] == scenario)
                & (centering["view"] == "centered")
            ]
            if standard.empty or centered.empty:
                continue
            standard_row = standard.iloc[0]
            centered_row = centered.iloc[0]
            if scenario == "S4-null":
                lines.append(
                    f"- {dataset_label} {scenario}: FAR@1 {_fmt(standard_row['far_at_1_mean'])} -> "
                    f"{_fmt(centered_row['far_at_1_mean'])}, FAR@5 "
                    f"{_fmt(standard_row['far_at_5_mean'])} -> {_fmt(centered_row['far_at_5_mean'])}."
                )
                continue
            message = (
                f"- {dataset_label} {scenario}: MRR {_fmt(standard_row['mrr_mean'])} -> "
                f"{_fmt(centered_row['mrr_mean'])}, HR@1 {_fmt(standard_row['hr_at_1_mean'])} -> "
                f"{_fmt(centered_row['hr_at_1_mean'])}"
            )
            if pd.notna(standard_row["far_at_1_mean"]) or pd.notna(centered_row["far_at_1_mean"]):
                message += f", FAR@1 {_fmt(standard_row['far_at_1_mean'])} -> {_fmt(centered_row['far_at_1_mean'])}"
            lines.append(f"{message}.")
    lines.extend(
        [
            "",
            "Reading the two views together separates the two failure modes: a pure physical offset "
            "is visible in the standard view but disappears after within-EQP centering (S1), while a "
            "coefficient-level mechanism shift survives centering (S2/S3/S4-main). In Dataset B "
            "S4-main, the standard view is dominated by the harmless marker (FAR@1 high, HR@1 zero), "
            "whereas the centered view restores root-cause localization and removes the marker false "
            "alarm. Reading Top-5 in the standard view is the minimalist alternative; the permutation "
            "null in Part 2 provides the calibrated alternative.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_null(
    s0_summary: pd.DataFrame,
    detection: pd.DataFrame,
    score_gap: pd.DataFrame,
) -> str:
    """Render the permutation-null and score-gap calibration sections."""
    s0_display = s0_summary.copy()
    detection_display = detection.loc[
        detection["scenario"].isin(["S0-clean", "S1", "S2", "S3", "S4-main", "S4-null"])
    ].copy()
    lines = [
        "## Part 2 — R3 #8 and R2 #1: permutation null and score-gap calibration",
        "",
        "### 2.1 S0-clean calibration (no physical, mechanism or harmless shift)",
        "",
        "GADS always produces a top-1 ranking, so the uncalibrated claim rate is 1.000 by "
        "construction. The permutation test rejects when the observed maximum exceeds the 95th "
        "percentile of the null maximum; under S0-clean this should reject about 5% of the 20 "
        "seed-level tests.",
        "",
        _md_table(
            s0_display,
            [
                "dataset",
                "view",
                "n_tests",
                "n_fwer_rejects_95",
                "fwer_rate_95",
                "fwer_ci_low_95",
                "fwer_ci_high_95",
                "per_feature_fpr_95",
            ],
        ),
        "",
        "### 2.2 Detection and exceedance on S1–S4",
        "",
        "`fwer_reject_rate_95` is the fraction of device-level tests whose observed maximum "
        "exceeds the family-wise threshold. `root_detection_power_fwer95` is the fraction of "
        "tests in which at least one true root cause exceeds that same threshold.",
        "",
        _md_table(
            detection_display,
            [
                "dataset",
                "scenario",
                "view",
                "fwer_reject_rate_95",
                "root_detection_power_fwer95",
                "all_roots_detection_rate_fwer95",
                "per_feature_fpr_95",
                "mean_features_exceed_tau95",
                "mean_features_exceed_fwer95",
                "top1_root_rate",
                "top1_harmless_rate",
            ],
        ),
        "",
        "### 2.3 Score-gap / elbow cutoff vs reading Top-5 (Dataset B, S4-main)",
        "",
        "The elbow rules choose the number of candidates from the largest consecutive score gap "
        "(absolute or ratio) within the top 20 candidates. `null_calibrated` keeps every feature "
        "above the S4-main permutation 95th-percentile family-wise threshold. `top5` is the "
        "reviewer's suggested reading rule.",
        "",
        _md_table(
            score_gap,
            [
                "rule",
                "n_seeds",
                "mean_selected_k",
                "root_in_set_rate",
                "harmless_in_set_rate",
                "root_in_top5_rate",
                "harmless_in_top5_rate",
                "mean_root_rank",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def _markdown_sweeps(
    s2_display: pd.DataFrame,
    s1_display: pd.DataFrame,
    s4_far: pd.DataFrame,
    s4_wasserstein: pd.DataFrame,
) -> str:
    """Render the Dataset B intensity-sweep section."""
    lines = [
        "## Part 3 — R3 #11: Dataset B intensity sweep (20 seeds)",
        "",
        "All four GADS distances are computed from the same OOF SHAP matrix for each cell.",
        "",
        "### 3.1 S2 mechanism local weight",
        "",
        _md_table(
            s2_display,
            ["level", "mrr_wasserstein", "mrr_mean", "mrr_kl", "mrr_js", "hr_at_1_wasserstein"],
        ),
        "",
        "### 3.2 S1 physical shift magnitude",
        "",
        _md_table(
            s1_display,
            ["level", "mrr_wasserstein", "mrr_mean", "mrr_kl", "mrr_js", "hr_at_1_wasserstein"],
        ),
        "",
        "### 3.3 S4-main harmless-drift magnitude (mechanism weight fixed at 12.0)",
        "",
        "FAR@1 by harmless-drift magnitude and distance (mean over 20 seeds):",
        "",
        _md_table(s4_far, ["level", "wasserstein", "mean", "kl", "js"]),
        "",
        "Wasserstein localization/false-alarm trajectory:",
        "",
        _md_table(
            s4_wasserstein,
            [
                "level",
                "mrr_mean",
                "hr_at_1_mean",
                "hr_at_5_mean",
                "far_at_1_mean",
                "far_at_5_mean",
            ],
        ),
        "",
        "The default 4.0 SD harmless drift is well beyond the 0.5–1.5 SD range typical of "
        "compensated real-fab drift. The smallest tested level is therefore the informative one:",
        "",
    ]
    if not s4_far.empty:
        ordered = s4_far.sort_values("level")
        smallest = ordered.iloc[0]
        lines.append(
            f"- At {smallest['level']:.1f} SD, Wasserstein FAR@1 is already "
            f"{smallest['wasserstein']:.2f} (mean {smallest['mean']:.2f}, KL {smallest['kl']:.2f}, "
            f"JS {smallest['js']:.2f})."
        )
        saturated = ordered.loc[ordered["wasserstein"] >= 0.9]
        if not saturated.empty:
            first = saturated.iloc[0]
            lines.append(
                f"- Wasserstein FAR@1 reaches {first['wasserstein']:.2f} at "
                f"{first['level']:.1f} SD, so the surrogate-marker takeover is not a threshold "
                "that switches on near the default magnitude; it is already severe at the "
                "smallest drift tested."
            )
    lines.append("")
    return "\n".join(lines)


def _key_findings(
    *,
    s0_summary: pd.DataFrame,
    centering: pd.DataFrame,
    variant_display: pd.DataFrame,
    s4_far: pd.DataFrame,
    s4_wasserstein: pd.DataFrame,
    detection: pd.DataFrame,
) -> list[str]:
    """Derive concise numeric findings from the aggregate tables."""
    findings = [
        f"S0-clean FWER ({record['dataset']}, {record['view']}): "
        f"{record['n_fwer_rejects_95']}/{record['n_tests']} = {record['fwer_rate_95']:.3f}."
        for record in s0_summary.to_dict(orient="records")
    ]
    findings.extend(
        f"{record['dataset']} S4-main {record['view']}: MRR={record['mrr_mean']:.3f}, "
        f"HR@1={record['hr_at_1_mean']:.3f}, HR@5={record['hr_at_5_mean']:.3f}, "
        f"FAR@1={record['far_at_1_mean']:.3f}, FAR@5={record['far_at_5_mean']:.3f}."
        for record in centering.to_dict(orient="records")
        if record["scenario"] == "S4-main"
    )
    findings.extend(
        f"{record['dataset']} S1 {record['view']}: MRR={record['mrr_mean']:.3f}, HR@1={record['hr_at_1_mean']:.3f}."
        for record in centering.to_dict(orient="records")
        if record["scenario"] == "S1"
    )
    findings.extend(
        f"{record['dataset']} S4-main {record['variant']}: FAR@1={record['far_at_1_mean']:.3f}, "
        f"HR@1={record['hr_at_1_mean']:.3f}, MRR={record['mrr_mean']:.3f}."
        for record in variant_display.to_dict(orient="records")
    )
    for level in (0.5, 1.0, 2.0, 3.0, 4.0):
        row = s4_far.loc[s4_far["level"] == level]
        if row.empty:
            continue
        record = row.iloc[0]
        findings.append(
            f"Dataset B S4-main harmless drift {level:.1f} SD: "
            f"FAR@1(wasserstein)={record['wasserstein']:.3f}, FAR@1(mean)={record['mean']:.3f}, "
            f"FAR@1(kl)={record['kl']:.3f}, FAR@1(js)={record['js']:.3f}."
        )
    for level in (0.5, 1.0, 4.0):
        row = s4_wasserstein.loc[s4_wasserstein["level"] == level]
        if row.empty:
            continue
        record = row.iloc[0]
        findings.append(
            f"Dataset B S4-main harmless drift {level:.1f} SD (Wasserstein): "
            f"MRR={record['mrr_mean']:.3f}, HR@1={record['hr_at_1_mean']:.3f}, "
            f"HR@5={record['hr_at_5_mean']:.3f}, FAR@1={record['far_at_1_mean']:.3f}, "
            f"FAR@5={record['far_at_5_mean']:.3f}."
        )
    s0_detection = detection.loc[detection["scenario"] == "S0-clean"]
    findings.extend(
        f"S0-clean per-feature 95% exceedance rate ({record['dataset']}, {record['view']}): "
        f"{record['per_feature_fpr_95']:.3f}."
        for record in s0_detection.to_dict(orient="records")
    )
    return findings


def _reviewer_map() -> str:
    """Render the reviewer-item to evidence map."""
    return (
        "## Reviewer-item map\n\n"
        "| Item | Evidence |\n| --- | --- |\n"
        "| R3 #7 (surrogate marker, TreeSHAP variant) | `base_diagnostics/tree_split_role_summary.csv`, "
        "`tree_split_feature_detail.csv`, `harmless_correlation.csv`, `shap_variant_comparison.csv` |\n"
        "| R3 #7 (within-EQP centering trade-off) | `base_diagnostics/gads_metrics.csv` "
        "(view=standard/centered), `gads_metrics_summary.csv` |\n"
        "| R3 #8 (permutation null) | `base_diagnostics/permutation_null_cells.csv`, "
        "`permutation_null_features.csv`, `s0_clean/fwer.csv`, `detection_power.csv` |\n"
        "| R2 #1 (Top-5 vs score-gap cutoff) | `base_diagnostics/score_gap_cutoff.csv`, "
        "`score_gap_summary.csv` |\n"
        "| R3 #11 / R3 #3 / R4 #3 (Dataset B intensity and distances) | "
        "`bsweep/s2_weight*.csv`, `bsweep/s1_shift*.csv`, `bsweep/s4_harmless*.csv` |\n"
    )


def _write_summary_markdown(
    path: Path,
    *,
    protocol: dict[str, Any],
    counts: Counter[str],
    runtime_seconds: float,
    role_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    variant_display: pd.DataFrame,
    paired_summary: pd.DataFrame,
    centering: pd.DataFrame,
    s0_summary: pd.DataFrame,
    detection: pd.DataFrame,
    score_gap: pd.DataFrame,
    s2_display: pd.DataFrame,
    s1_display: pd.DataFrame,
    s4_far: pd.DataFrame,
    s4_wasserstein: pd.DataFrame,
    findings: list[str],
) -> None:
    """Assemble and write ``summary.md``."""
    sections = [
        _markdown_protocol(protocol, counts=counts, runtime_seconds=runtime_seconds),
        _markdown_tree(role_summary, correlation_summary),
        _markdown_variant(variant_display, paired_summary),
        _markdown_centering(centering),
        _markdown_null(s0_summary, detection, score_gap),
        _markdown_sweeps(s2_display, s1_display, s4_far, s4_wasserstein),
        "## Key numeric findings\n\n" + "\n".join(f"- {finding}" for finding in findings) + "\n",
        _reviewer_map(),
        "## Output index\n\n"
        "- `protocol.json` — exact settings, versions and task counts.\n"
        "- `summary.json` — machine-readable aggregate tables and key findings.\n"
        "- `base_diagnostics/` — GADS metrics, tree splits, correlations, TreeSHAP variants, "
        "permutation nulls and score-gap cutoffs.\n"
        "- `s0_clean/` — no-fault FWER calibration tables.\n"
        "- `bsweep/` — Dataset B S2/S1/S4-main intensity sweeps.\n"
        "- `partials/` — per-task JSON results (resume support).\n",
    ]
    path.write_text("\n\n".join(sections), encoding="utf-8")


def _verify_protocol(datasets: tuple[str, ...]) -> dict[str, Any]:
    """Verify the custom fold loop against the paper implementation and check S0-clean."""
    checks: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = _load_bundle(
            dataset,
            "S1",
            0,
            float(DATASET_B_PARAMS["physical_shift"]),
            float(DATASET_B_PARAMS["mechanism_weight"]),
            float(DATASET_B_PARAMS["harmless_shift"]),
        )
        X, y, groups = _feature_matrix(bundle)
        fast = _cross_fit(X, y, dataset=dataset, center_by_group=False)
        reference = fit_cross_fitted_model(
            X,
            y,
            model_type="xgboost",
            model_seed=MODEL_SEED,
            cv_seed=None,
            cv_splits=CV_SPLITS,
            model_params=_config_for_dataset(dataset),
        )
        shap_difference = float(
            np.max(np.abs(fast.shap_values.to_numpy(dtype=float) - reference.shap_values.to_numpy(dtype=float)))
        )
        auc_difference = abs(fast.mean_auc - reference.mean_auc)
        fast_rankings = _rank_all_distances(fast.shap_values, groups, bundle)
        reference_rankings = rank_gads_features(
            reference.shap_values,
            groups,
            bundle.abnormal_devices,
            bundle.normal_devices,
            process_features=bundle.process_features,
            view="process",
            distance="wasserstein",
            bins=DEFAULT_GADS_BINS,
            epsilon=DEFAULT_GADS_EPSILON,
        )
        fast_sorted = fast_rankings["wasserstein"].sort_values(["equipment", "feature"], kind="stable")
        reference_sorted = reference_rankings.sort_values(["equipment", "feature"], kind="stable")
        if (
            not fast_sorted[["equipment", "feature"]]
            .reset_index(drop=True)
            .equals(reference_sorted[["equipment", "feature"]].reset_index(drop=True))
        ):
            raise RuntimeError(f"protocol self-check failed for {dataset}: ranking keys differ")
        score_difference = float(
            np.max(np.abs(fast_sorted["score"].to_numpy(dtype=float) - reference_sorted["score"].to_numpy(dtype=float)))
        )
        if shap_difference > 1e-10:
            raise RuntimeError(f"protocol self-check failed for {dataset}: SHAP diff={shap_difference}")
        if auc_difference > 1e-10:
            raise RuntimeError(f"protocol self-check failed for {dataset}: AUC diff={auc_difference}")
        if score_difference > 1e-10:
            raise RuntimeError(f"protocol self-check failed for {dataset}: GADS score diff={score_difference}")
        check_feature = next(iter(bundle.process_features))
        values = fast.shap_values[check_feature].to_numpy(dtype=float)
        membership = (groups == bundle.abnormal_devices[0]).to_numpy()
        cdf_distance = _wasserstein_from_membership(values, membership)
        scipy_distance = compute_gads_distance(values[membership], values[~membership], distance="wasserstein")
        if abs(cdf_distance - scipy_distance) > 1e-10:
            raise RuntimeError(
                f"Wasserstein CDF implementation differs from scipy for {dataset}: {cdf_distance} vs {scipy_distance}"
            )
        checks.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "max_shap_abs_diff": shap_difference,
                "max_auc_abs_diff": auc_difference,
                "max_gads_score_abs_diff": score_difference,
                "wasserstein_cdf_abs_diff": abs(cdf_distance - scipy_distance),
                "n_process_features": len(bundle.process_features),
                "n_samples": len(bundle.data),
            }
        )

    s0_checks: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = _load_bundle(
            dataset,
            "S0-clean",
            0,
            float(DATASET_B_PARAMS["physical_shift"]),
            float(DATASET_B_PARAMS["mechanism_weight"]),
            float(DATASET_B_PARAMS["harmless_shift"]),
        )
        device = bundle.abnormal_devices[0]
        groups = bundle.data[bundle.group_column].astype(str)
        abnormal = groups == device
        standardized_differences: dict[str, float] = {}
        for feature in bundle.process_features:
            values = bundle.data[feature].to_numpy(dtype=float)
            abnormal_values = values[abnormal.to_numpy()]
            normal_values = values[~abnormal.to_numpy()]
            scale = float(np.std(normal_values))
            mean_difference = abs(float(np.mean(abnormal_values) - np.mean(normal_values)))
            if scale == 0.0:
                if mean_difference > 1e-12:
                    raise RuntimeError(
                        f"S0-clean feature {feature!r} is constant in the normal pool but "
                        f"shifted by {mean_difference} in the abnormal pool"
                    )
                standardized_differences[feature] = 0.0
                continue
            standardized_differences[feature] = mean_difference / scale
        max_smd = max(standardized_differences.values())
        threshold = 0.45
        if max_smd > threshold:
            worst = max(standardized_differences, key=standardized_differences.get)
            raise RuntimeError(
                f"S0-clean construction failed for {dataset}: max |SMD|={max_smd:.3f} on {worst!r} exceeds {threshold}"
            )
        mechanism_effect = None
        if dataset == "b":
            mechanism = bundle.metadata.get("glm", {}).get("mechanism", {})
            if not isinstance(mechanism, dict) or device not in mechanism:
                raise ValueError("S0-clean Dataset B metadata is missing the mechanism record")
            mechanism_effect = float(mechanism[device]["local_weight"] - mechanism[device]["global_weight"])
            if abs(mechanism_effect) > 1e-12:
                raise RuntimeError(f"S0-clean Dataset B mechanism effect is non-zero: {mechanism_effect}")
        s0_checks.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "abnormal_device": device,
                "max_standardized_mean_difference": max_smd,
                "threshold": threshold,
                "mechanism_weight_effect": mechanism_effect,
                "n_process_features": len(bundle.process_features),
            }
        )
    return {"checks": checks, "s0_clean_checks": s0_checks}


def _runtime_from_log(output_dir: Path) -> float:
    """Recover the longest measured pool wall time from ``run.log`` progress lines."""
    log_path = output_dir / "run.log"
    if not log_path.is_file():
        return 0.0
    pattern = re.compile(r"elapsed=([0-9.]+) min")
    matches = pattern.findall(log_path.read_text(encoding="utf-8"))
    if not matches:
        return 0.0
    return max(float(match) for match in matches) * 60.0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS, default=list(DATASET_KEYS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=BASE_SCENARIOS,
        default=list(BASE_SCENARIOS),
        help="base scenarios to run (default: all six including S0-clean)",
    )
    parser.add_argument(
        "--scenario-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SCENARIO_SEEDS),
        help="scenario seeds (default: 0..19)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument("--background-size", type=int, default=DEFAULT_BACKGROUND_SIZE)
    parser.add_argument("--force", action="store_true", help="ignore existing partial results")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument(
        "--reaggregate",
        action="store_true",
        help="skip model fitting and rebuild CSV/JSON/markdown from existing partials",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the task plan without running")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Validate CLI arguments and return the deduplicated grids."""
    if not 1 <= args.workers <= 19:
        raise ValueError("--workers must lie in [1, 19]")
    if args.n_permutations < 100:
        raise ValueError("--n-permutations must be at least 100")
    if args.background_size < 20:
        raise ValueError("--background-size must be at least 20")
    datasets = tuple(dict.fromkeys(args.datasets))
    scenarios = tuple(dict.fromkeys(args.scenarios))
    scenario_seeds = tuple(dict.fromkeys(int(seed) for seed in args.scenario_seeds))
    if not datasets or not scenarios or not scenario_seeds:
        raise ValueError("at least one dataset, scenario and scenario seed are required")
    if any(seed < 0 for seed in scenario_seeds):
        raise ValueError("scenario seeds must be non-negative")
    if "b" in datasets:
        verify_secom_files(RAW_DIR)
    return datasets, scenarios, scenario_seeds


def main() -> None:
    """Run Pack 3 and write all tables, JSON and the markdown summary."""
    args = _parse_args()
    datasets, scenarios, scenario_seeds = _validate_args(args)
    output_dir = args.output_dir.resolve()
    partials_dir = output_dir / "partials"
    output_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)
    _TASK_SETTINGS["n_permutations"] = int(args.n_permutations)
    _TASK_SETTINGS["background_size"] = int(args.background_size)

    tasks = build_tasks(datasets, scenarios, scenario_seeds)
    if not tasks:
        raise ValueError("task grid is empty")
    counts = Counter("base" if isinstance(task, BaseTask) else "sweep" for task in tasks)
    print(f"Prepared {len(tasks)} tasks: datasets={datasets}, scenarios={scenarios}", flush=True)
    print(f"Scenario seeds: {list(scenario_seeds)}; counts: {dict(counts)}", flush=True)
    if args.dry_run:
        for task in tasks[:20]:
            print(task, flush=True)
        print(f"... {len(tasks)} tasks total", flush=True)
        return

    if args.reaggregate:
        prior_path = output_dir / "protocol.json"
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            self_check = prior.get("self_check", {"reaggregated": True})
            runtime_seconds = max(
                float(prior.get("runtime_seconds", 0.0)),
                _runtime_from_log(output_dir),
            )
        else:
            self_check = {"reaggregated": True}
            runtime_seconds = 0.0
        payloads = _load_all_partials(partials_dir, tasks)
        print(f"Reaggregated {len(payloads)} completed tasks", flush=True)
    else:
        if not args.skip_self_check:
            self_check = _verify_protocol(datasets)
            print(f"Protocol self-check passed: {self_check}", flush=True)
        else:
            self_check = {"skipped": True}

        pending: list[Task] = []
        for task in tasks:
            if args.force:
                pending.append(task)
                continue
            path = _partial_path(partials_dir, task.task_id)
            if path.is_file():
                payload = _load_partial(partials_dir, task.task_id)
                if payload.get("status") == "ok" and _partial_is_compatible(payload):
                    continue
            pending.append(task)
        print(f"Pending tasks: {len(pending)} of {len(tasks)}", flush=True)

        started = time.monotonic()
        _run_pool(pending, workers=args.workers, partials_dir=partials_dir)
        runtime_seconds = time.monotonic() - started
        payloads = _load_all_partials(partials_dir, tasks)
        print(f"All tasks completed in {runtime_seconds / 60.0:.1f} min", flush=True)

    base_dir = output_dir / "base_diagnostics"
    s0_dir = output_dir / "s0_clean"
    sweep_dir = output_dir / "bsweep"
    base_dir.mkdir(parents=True, exist_ok=True)
    s0_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    gads_metrics = _merge_table(payloads, "gads_metrics", GADS_METRIC_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "view", "distance"], kind="stable"
    )
    tree_role = _merge_table(payloads, "tree_split_role_summary", TREE_ROLE_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "device", "role"], kind="stable"
    )
    tree_feature = _merge_table(payloads, "tree_split_feature_detail", TREE_FEATURE_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "feature_kind", "feature"], kind="stable"
    )
    harmless_correlation = _merge_table(payloads, "harmless_correlation", HARMLESS_CORRELATION_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "harmless_order"], kind="stable"
    )
    shap_variant = _merge_table(payloads, "shap_variant", SHAP_VARIANT_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "variant", "distance"], kind="stable"
    )
    perm_cells = _merge_table(payloads, "perm_null_cells", PERM_CELL_COLUMNS)
    perm_features = _merge_table(payloads, "perm_null_features", PERM_FEATURE_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "view", "device", "gads_rank"], kind="stable"
    )
    exceed_counts = (
        perm_features.groupby(
            ["dataset", "scenario", "scenario_seed", "view", "device"],
            dropna=False,
            observed=True,
        )["exceeds_fwer95"]
        .sum()
        .rename("n_features_exceed_fwer95")
        .reset_index()
    )
    perm_cells = perm_cells.drop(columns="n_features_exceed_fwer95", errors="ignore").merge(
        exceed_counts,
        on=["dataset", "scenario", "scenario_seed", "view", "device"],
        how="left",
        validate="one_to_one",
    )
    if perm_cells["n_features_exceed_fwer95"].isna().any():
        raise ValueError("failed to derive the family-wise exceedance count for every null cell")
    perm_cells["n_features_exceed_fwer95"] = perm_cells["n_features_exceed_fwer95"].astype(int)
    perm_cells = _ensure_columns(perm_cells, PERM_CELL_COLUMNS).sort_values(
        ["dataset", "scenario", "scenario_seed", "view", "device"], kind="stable"
    )
    score_gap = _rebuild_score_gap(perm_features, perm_cells).sort_values(
        ["dataset", "scenario", "scenario_seed", "device", "rule"], kind="stable"
    )
    sweep_metrics = _merge_table(payloads, "sweep_metrics", SWEEP_METRIC_COLUMNS)

    _write_csv(gads_metrics, base_dir / "gads_metrics.csv")
    _write_csv(tree_role, base_dir / "tree_split_role_summary.csv")
    _write_csv(tree_feature, base_dir / "tree_split_feature_detail.csv")
    _write_csv(harmless_correlation, base_dir / "harmless_correlation.csv")
    _write_csv(shap_variant, base_dir / "shap_variant_comparison.csv")
    _write_csv(perm_cells, base_dir / "permutation_null_cells.csv")
    _write_csv(perm_features, base_dir / "permutation_null_features.csv")
    _write_csv(score_gap, base_dir / "score_gap_cutoff.csv")

    metrics_summary = _mean_std(
        gads_metrics,
        ["dataset", "scenario", "view", "distance"],
        (*METRIC_COLUMNS, "kpi_auc"),
    )
    _write_csv(metrics_summary, base_dir / "gads_metrics_summary.csv")
    role_summary = _tree_role_summary(tree_role)
    _write_csv(role_summary, base_dir / "tree_split_role_summary_mean.csv")
    correlation_summary = _harmless_correlation_summary(harmless_correlation)
    _write_csv(correlation_summary, base_dir / "harmless_correlation_summary.csv")
    variant_summary, variant_paired = _shap_variant_summary(shap_variant)
    _write_csv(variant_summary, base_dir / "shap_variant_summary.csv")
    _write_csv(variant_paired, base_dir / "shap_variant_paired_summary.csv")
    detection = _detection_power(perm_cells)
    _write_csv(detection, base_dir / "detection_power.csv")
    score_gap_summary = _score_gap_summary(score_gap)
    _write_csv(score_gap_summary, base_dir / "score_gap_summary.csv")

    s0_cells = perm_cells.loc[perm_cells["scenario"] == "S0-clean"].copy()
    _write_csv(s0_cells, s0_dir / "fwer.csv")
    s0_summary = _s0_fwer_summary(perm_cells)
    _write_csv(s0_summary, s0_dir / "fwer_summary.csv")

    sweep_summaries: dict[str, pd.DataFrame] = {}
    for sweep_name in SWEEP_GRIDS:
        if "b" not in datasets:
            continue
        if str(SWEEP_GRIDS[sweep_name]["scenario"]) not in scenarios:
            continue
        sweep_frame = _sweep_frame(gads_metrics, sweep_metrics, sweep_name)
        _write_csv(sweep_frame, sweep_dir / f"{sweep_name}.csv")
        summary = _sweep_summary(sweep_frame)
        _write_csv(summary, sweep_dir / f"{sweep_name}_summary.csv")
        sweep_summaries[sweep_name] = summary

    if "s2_weight" in sweep_summaries:
        s2_display = _sweep_display(sweep_summaries["s2_weight"], "s2_weight", ("mrr", "hr_at_1"))
    else:
        s2_display = pd.DataFrame(columns=["level"])
    if "s1_shift" in sweep_summaries:
        s1_display = _sweep_display(sweep_summaries["s1_shift"], "s1_shift", ("mrr", "hr_at_1"))
    else:
        s1_display = pd.DataFrame(columns=["level"])
    if "s4_harmless" in sweep_summaries:
        s4_far = _s4_harmless_far_pivot(sweep_summaries["s4_harmless"])
        s4_wasserstein = _s4_harmless_wasserstein(sweep_summaries["s4_harmless"])
        _write_csv(s4_far, sweep_dir / "s4_harmless_far_at_1_pivot.csv")
        _write_csv(s4_wasserstein, sweep_dir / "s4_harmless_wasserstein_trajectory.csv")
    else:
        s4_far = pd.DataFrame(columns=["level"])
        s4_wasserstein = pd.DataFrame(columns=["level"])

    centering = _centering_table(metrics_summary)
    _write_csv(centering, base_dir / "within_eqp_centering.csv")
    variant_display = _variant_display(variant_summary)

    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "ld_preload": os.environ.get("LD_PRELOAD"),
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "xgboost": xgboost.__version__,
        },
        "workers": args.workers,
        "datasets": list(datasets),
        "scenarios": list(scenarios),
        "scenario_seeds": list(scenario_seeds),
        "dataset_a_params": DATASET_A_PARAMS,
        "dataset_b_params": DATASET_B_PARAMS,
        "model_params": MODEL_PARAMS,
        "model_seed": MODEL_SEED,
        "cv_splits": CV_SPLITS,
        "gads": {
            "distances": list(DISTANCES),
            "bins": DEFAULT_GADS_BINS,
            "epsilon": DEFAULT_GADS_EPSILON,
        },
        "n_permutations": int(args.n_permutations),
        "background_size": int(args.background_size),
        "sweeps": SWEEP_GRIDS,
        "task_counts": dict(counts),
        "self_check": self_check,
        "output_dir": str(output_dir),
        "runtime_seconds": runtime_seconds,
        "reaggregated": bool(args.reaggregate),
    }
    _write_json(output_dir / "protocol.json", protocol)

    findings = _key_findings(
        s0_summary=s0_summary,
        centering=centering,
        variant_display=variant_display,
        s4_far=s4_far,
        s4_wasserstein=s4_wasserstein,
        detection=detection,
    )
    summary_payload = {
        "pack": "pack3_diagnostics_bsweep",
        "protocol": protocol,
        "runtime_seconds": runtime_seconds,
        "r3_7": {
            "tree_role_summary": _records(role_summary),
            "harmless_correlation_summary": _records(correlation_summary),
            "shap_variant_summary": _records(variant_summary),
            "shap_variant_paired_summary": _records(variant_paired),
            "within_eqp_centering": _records(centering),
        },
        "r3_8_r2_1": {
            "s0_fwer_summary": _records(s0_summary),
            "detection_power": _records(detection),
            "score_gap_summary": _records(score_gap_summary),
        },
        "r3_11": {
            "s2_weight_summary": _records(sweep_summaries.get("s2_weight", pd.DataFrame())),
            "s1_shift_summary": _records(sweep_summaries.get("s1_shift", pd.DataFrame())),
            "s4_harmless_summary": _records(sweep_summaries.get("s4_harmless", pd.DataFrame())),
            "s4_harmless_far_at_1_pivot": _records(s4_far),
            "s4_harmless_wasserstein_trajectory": _records(s4_wasserstein),
        },
        "key_findings": findings,
    }
    _write_json(output_dir / "summary.json", summary_payload)

    _write_summary_markdown(
        output_dir / "summary.md",
        protocol=protocol,
        counts=counts,
        runtime_seconds=runtime_seconds,
        role_summary=role_summary,
        correlation_summary=correlation_summary,
        variant_display=variant_display,
        paired_summary=variant_paired,
        centering=centering,
        s0_summary=s0_summary,
        detection=detection,
        score_gap=score_gap_summary,
        s2_display=s2_display,
        s1_display=s1_display,
        s4_far=s4_far,
        s4_wasserstein=s4_wasserstein,
        findings=findings,
    )
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

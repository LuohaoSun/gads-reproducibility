#!/usr/bin/env python3
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportReturnType=false
"""Pack 1 revision experiments: model/CV seeds, Dataset B hyperparameters and stability.

The script is self-contained (it imports the paper's ``src`` package but does not
modify it) and reproduces the exact paper protocol:

* Dataset A: n=5000, noise=0.2, 10 devices, shift=3.0, mechanism_scale=1.0,
  harmless_drift=3.5; XGBoost ``max_depth=6`` / ``min_child_weight=1``.
* Dataset B: SECOM, group_seed=2024, physical_shift=3.0, mechanism_weight=12.0,
  harmless_shift=4.0, missing indicators excluded; XGBoost ``max_depth=3`` /
  ``min_child_weight=5``.
* ``eqp_mode="contextual_process"``, GADS Wasserstein distance with 50 bins and
  epsilon 1e-10, 5-fold cross-fitting whose CV seed equals the model seed.

Outputs (CSV + JSON + ``summary.md``) go to
``results/bdcc_revision/pack1_seeds_hyperparams/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
import traceback
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap, kendalltau, norm, spearmanr
import shap
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import xgboost

from shap_diff_analysis.baselines import ALL_METHODS, rank_wasserstein_x
from shap_diff_analysis.experiment_runner import ExperimentSettings, git_commit, run_experiment_bundle
from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.gads import DEFAULT_GADS_BINS, DEFAULT_GADS_EPSILON, rank_gads_features
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.metrics import add_result_context, evaluate_rankings
from shap_diff_analysis.modeling import (
    _aggregate_shap,
    _class_weight,
    _extract_binary_shap,
    _make_estimator,
    _make_preprocessor,
    fit_cross_fitted_model,
)
from shap_diff_analysis.secom import DEFAULT_RAW_DIR, generate_dataset_b, verify_secom_files
from shap_diff_analysis.statistics import holm_adjust, metric_applicable, paired_permutation_test

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/bdcc_revision/pack1_seeds_hyperparams"

SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4-main", "S4-null")
DEFAULT_SCENARIO_SEEDS: tuple[int, ...] = tuple(range(20))
DEFAULT_MODEL_SEEDS: tuple[int, ...] = (42, 101, 202, 303, 404)
DATASET_LABELS: dict[str, str] = {"a": "Dataset A", "b": "Dataset B"}
METRIC_COLUMNS: tuple[str, ...] = ("mrr", "hr_at_1", "hr_at_5", "far_at_1", "far_at_5")
FAR_METRICS: frozenset[str] = frozenset({"far_at_1", "far_at_5"})
AUC_METRICS: tuple[str, ...] = ("kpi_auc", "train_auc", "train_oof_auc_gap")
EFFECT_METRICS: tuple[str, ...] = (*METRIC_COLUMNS, *AUC_METRICS)
TOP_K: int = 5

_CHECKPOINT_DIR: Path | None = None

JobName = Literal["model_seed", "hyperparam", "missing_indicator", "main", "wasserstein_scale"]


@dataclass(frozen=True)
class ParamConfig:
    """One XGBoost hyperparameter configuration."""

    label: str
    max_depth: int
    min_child_weight: int

    def as_model_params(self) -> dict[str, int]:
        """Return the estimator override dictionary."""
        return {"max_depth": self.max_depth, "min_child_weight": self.min_child_weight}


DATASET_A_CONFIG = ParamConfig("a_default_d6_mcw1", 6, 1)
DATASET_B_CONFIG = ParamConfig("d3_mcw5", 3, 5)
DATASET_B_GRID: tuple[ParamConfig, ...] = (
    ParamConfig("d3_mcw1", 3, 1),
    ParamConfig("d3_mcw5", 3, 5),
    ParamConfig("d3_mcw10", 3, 10),
    ParamConfig("d4_mcw5", 4, 5),
    ParamConfig("d6_mcw1", 6, 1),
    ParamConfig("d6_mcw5", 6, 5),
)
CONFIG_ORDER: dict[str, int] = {config.label: index for index, config in enumerate(DATASET_B_GRID)}
CONFIG_ORDER[DATASET_A_CONFIG.label] = -1


@dataclass(frozen=True)
class RunTask:
    """One independently executable experiment run."""

    job: JobName
    dataset: str
    scenario: str
    scenario_seed: int
    model_seed: int
    config_label: str
    max_depth: int
    min_child_weight: int
    include_missing_indicators: bool
    variant: str = ""


@dataclass(frozen=True)
class RunEvaluation:
    """Metrics and rankings produced by one run."""

    metrics: pd.DataFrame
    rankings: pd.DataFrame | None
    train_auc: float | None
    oof_auc: float | None
    n_samples: int
    n_features: int


@dataclass
class RunResult:
    """Result or captured error for one task."""

    task: RunTask
    evaluation: RunEvaluation | None = None
    error: str | None = None


def _config_for_dataset(dataset: str) -> ParamConfig:
    if dataset == "a":
        return DATASET_A_CONFIG
    if dataset == "b":
        return DATASET_B_CONFIG
    raise ValueError(f"unknown dataset key {dataset!r}; expected 'a' or 'b'")


def _generate_bundle(
    dataset: str,
    scenario: str,
    scenario_seed: int,
    include_missing_indicators: bool,
) -> DatasetBundle:
    if dataset == "a":
        return generate_dataset_bundle(
            scenario=scenario,
            n_samples=5_000,
            seed=scenario_seed,
            noise_level=0.2,
            n_devices=10,
            distribution_shift=3.0,
            mechanism_scale=1.0,
            harmless_drift=3.5,
        )
    if dataset == "b":
        return generate_dataset_b(
            scenario=scenario,  # type: ignore[arg-type]
            seed=scenario_seed,
            group_seed=2024,
            raw_dir=RAW_DIR,
            physical_shift_magnitude=3.0,
            mechanism_local_weight=12.0,
            harmless_shift_magnitude=4.0,
            include_missing_indicators=include_missing_indicators,
        )
    raise ValueError(f"unknown dataset key {dataset!r}; expected 'a' or 'b'")


@lru_cache(maxsize=4)
def load_bundle(
    dataset: str,
    scenario: str,
    scenario_seed: int,
    include_missing_indicators: bool,
) -> DatasetBundle:
    """Generate and cache a bundle inside one worker process.

    The returned frames are never mutated by the experiment code, so caching a
    handful of bundles per worker is safe and removes repeated SECOM parsing.
    """
    return _generate_bundle(dataset, scenario, scenario_seed, include_missing_indicators)


def _feature_matrix(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    columns = [*bundle.process_features, bundle.group_column]
    X = bundle.data.loc[:, columns]
    y = pd.to_numeric(bundle.data[bundle.target_column], errors="raise").astype(int)
    groups = bundle.data[bundle.group_column].astype(str)
    return X, y, groups


def _add_ranking_context(
    rankings: pd.DataFrame,
    *,
    bundle: DatasetBundle,
    dataset_label: str,
    scenario_seed: int,
    model_seed: int,
) -> pd.DataFrame:
    result = rankings.copy()
    result.insert(0, "dataset", dataset_label)
    result.insert(1, "scenario", bundle.scenario)
    result.insert(2, "scenario_seed", scenario_seed)
    result.insert(3, "model_seed", model_seed)
    roots = {str(device): set(features) for device, features in bundle.root_causes.items()}
    result["is_root_cause"] = [
        feature in roots.get(str(device), set())
        for device, feature in zip(result["equipment"], result["feature"], strict=True)
    ]
    result["is_harmless"] = result["feature"].isin(bundle.harmless_features)
    result["eqp_mode"] = "contextual_process"
    return result


def _assert_applicable_metrics(metrics: pd.DataFrame, *, require_auc: bool = True) -> None:
    for scenario, group in metrics.groupby("scenario", sort=False):
        for metric in METRIC_COLUMNS:
            applicable, _note = metric_applicable(str(scenario), metric)
            if applicable and not np.isfinite(group[metric].to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite {metric} in applicable scenario {scenario!r}")
        if not require_auc:
            continue
        for metric in AUC_METRICS:
            if metric in group.columns and not np.isfinite(group[metric].to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite {metric} in scenario {scenario!r}")


def _fit_oof_attributions(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_seed: int,
    cv_splits: int,
    model_params: dict[str, int],
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Fit the five OOF folds only (no final all-data model) for GADS metrics.

    This mirrors ``fit_cross_fitted_model``'s fold loop exactly, including the
    fold preprocessor, estimator seed, SHAP extraction and encoded-column
    aggregation.  The final pipeline is omitted because GADS metrics and OOF
    attributions never use it; the final model is still fitted for the
    hyperparameter runs that report a train-vs-OOF AUC gap.
    """
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_seed)
    shap_oof = np.full((len(X), X.shape[1]), np.nan, dtype=float)
    predictions = np.full(len(X), np.nan, dtype=float)
    fold_aucs: list[float] = []
    encoded_feature_names: tuple[str, ...] | None = None
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        preprocessor = _make_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)
        if encoded_feature_names is None:
            encoded_feature_names = tuple(preprocessor.get_feature_names_out())
        elif tuple(preprocessor.get_feature_names_out()) != encoded_feature_names:
            raise ValueError("fold preprocessors produced inconsistent feature names")
        estimator = _make_estimator(
            "xgboost",
            random_state=model_seed + fold,
            scale_pos_weight=_class_weight(y_train),
            model_params=model_params,
        )
        estimator.fit(X_train_t, y_train)
        predictions[valid_idx] = estimator.predict_proba(X_valid_t)[:, 1]
        shap_encoded = _extract_binary_shap(estimator, np.asarray(X_valid_t))
        shap_oof[valid_idx] = _aggregate_shap(shap_encoded, preprocessor, list(X.columns)).to_numpy()
        fold_aucs.append(float(roc_auc_score(y_valid, predictions[valid_idx])))
    if np.isnan(predictions).any() or np.isnan(shap_oof).any():
        raise RuntimeError("cross-fitting did not produce complete OOF predictions/attributions")
    return (
        pd.DataFrame(shap_oof, index=X.index, columns=X.columns),
        pd.Series(predictions, index=X.index, name="oof_prediction"),
        float(np.mean(fold_aucs)),
    )


def evaluate_gads_run(
    bundle: DatasetBundle,
    *,
    dataset_label: str,
    model_seed: int,
    model_params: dict[str, int],
    fit_final_model: bool = True,
) -> RunEvaluation:
    """Fit the paper KPI model and compute GADS process-view metrics."""
    X, y, groups = _feature_matrix(bundle)
    if fit_final_model:
        attribution = fit_cross_fitted_model(
            X,
            y,
            model_type="xgboost",
            model_seed=model_seed,
            cv_seed=None,
            cv_splits=5,
            model_params=model_params,
        )
        shap_values = attribution.shap_values
        mean_auc = attribution.mean_auc
        train_auc = float(roc_auc_score(y, attribution.model.predict_proba(X)[:, 1]))
    else:
        shap_values, _predictions, mean_auc = _fit_oof_attributions(
            X,
            y,
            model_seed=model_seed,
            cv_splits=5,
            model_params=model_params,
        )
        train_auc = np.nan
    rankings = rank_gads_features(
        shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
        bins=DEFAULT_GADS_BINS,
        epsilon=DEFAULT_GADS_EPSILON,
    )
    rankings = _add_ranking_context(
        rankings,
        bundle=bundle,
        dataset_label=dataset_label,
        scenario_seed=bundle.seed,
        model_seed=model_seed,
    )
    metrics = evaluate_rankings(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=(1, 3, 5),
    )
    metrics = add_result_context(
        metrics,
        dataset=dataset_label,
        scenario=bundle.scenario,
        seed=bundle.seed,
    ).rename(columns={"seed": "scenario_seed"})
    metrics["model_seed"] = model_seed
    metrics["kpi_auc"] = mean_auc
    metrics["train_auc"] = train_auc
    metrics["train_oof_auc_gap"] = train_auc - mean_auc
    metrics["status"] = "valid"
    metrics["eqp_mode"] = "contextual_process"
    metrics["n_samples"] = len(X)
    metrics["n_model_features"] = X.shape[1]
    metrics["n_process_features"] = len(bundle.process_features)
    _assert_applicable_metrics(metrics, require_auc=fit_final_model)
    return RunEvaluation(
        metrics=metrics,
        rankings=rankings,
        train_auc=train_auc if fit_final_model else None,
        oof_auc=mean_auc,
        n_samples=len(X),
        n_features=X.shape[1],
    )


def evaluate_main_run(
    bundle: DatasetBundle,
    *,
    dataset_label: str,
    model_seed: int,
    model_params: dict[str, int],
) -> RunEvaluation:
    """Run the exact paper method matrix (GADS plus five classical baselines)."""
    settings = ExperimentSettings(
        methods=ALL_METHODS,
        model_type="xgboost",
        model_seed=model_seed,
        cv_seed=None,
        cv_splits=5,
        model_params=model_params,
        top_ks=(1, 3, 5),
        auc_threshold=None,
        psi_bins=10,
        gads_distance="wasserstein",
        gads_bins=DEFAULT_GADS_BINS,
        gads_epsilon=DEFAULT_GADS_EPSILON,
        eqp_mode="contextual_process",
    )
    result = run_experiment_bundle(bundle, settings=settings)
    metrics = result.metrics.copy()
    metrics["dataset"] = dataset_label
    metrics = metrics.rename(columns={"seed": "scenario_seed"})
    metrics["model_seed"] = model_seed
    metrics["n_samples"] = len(bundle.data)
    metrics["n_model_features"] = len(bundle.process_features) + 1
    metrics["n_process_features"] = len(bundle.process_features)
    rankings = result.rankings.copy()
    rankings["dataset"] = dataset_label
    rankings = rankings.rename(columns={"seed": "scenario_seed"})
    rankings["model_seed"] = model_seed
    rankings = rankings.loc[rankings["view"] == "process"].copy()
    _assert_applicable_metrics(metrics)
    return RunEvaluation(
        metrics=metrics,
        rankings=rankings,
        train_auc=None,
        oof_auc=float(result.metrics["kpi_auc"].iloc[0]),
        n_samples=len(bundle.data),
        n_features=len(bundle.process_features) + 1,
    )


def _evaluate_wasserstein_scale(task: RunTask) -> RunEvaluation:
    bundle = load_bundle("a", task.scenario, task.scenario_seed, False)
    features = list(bundle.process_features)
    X_raw = bundle.data.loc[:, features]
    groups = bundle.data[bundle.group_column].astype(str)
    if task.variant == "raw":
        X = X_raw
    elif task.variant == "normal_pool_zscore":
        normal_mask = groups.isin(bundle.normal_devices).to_numpy()
        normal = X_raw.loc[normal_mask]
        means = normal.mean(axis=0)
        stds = normal.std(axis=0, ddof=1)
        if (stds <= 0.0).any():
            raise ValueError(f"normal-pool standardization encountered zero scale: {task.scenario_seed}")
        X = (X_raw - means) / stds
    else:
        raise ValueError(f"unknown Wasserstein-X variant {task.variant!r}")
    rankings = rank_wasserstein_x(
        X,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
    )
    rankings = _add_ranking_context(
        rankings,
        bundle=bundle,
        dataset_label=DATASET_LABELS["a"],
        scenario_seed=bundle.seed,
        model_seed=42,
    )
    rankings["variant"] = task.variant
    metrics = evaluate_rankings(
        rankings,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        top_ks=(1, 3, 5),
    )
    metrics = add_result_context(
        metrics,
        dataset=DATASET_LABELS["a"],
        scenario=bundle.scenario,
        seed=bundle.seed,
    ).rename(columns={"seed": "scenario_seed"})
    metrics["model_seed"] = 42
    metrics["variant"] = task.variant
    metrics["kpi_auc"] = np.nan
    metrics["train_auc"] = np.nan
    metrics["train_oof_auc_gap"] = np.nan
    metrics["status"] = "valid"
    metrics["eqp_mode"] = "contextual_process"
    metrics["n_samples"] = len(bundle.data)
    metrics["n_model_features"] = np.nan
    metrics["n_process_features"] = len(bundle.process_features)
    _assert_applicable_metrics(metrics, require_auc=False)
    return RunEvaluation(
        metrics=metrics,
        rankings=rankings,
        train_auc=None,
        oof_auc=None,
        n_samples=len(bundle.data),
        n_features=len(bundle.process_features),
    )


def _attach_task_context(frame: pd.DataFrame, task: RunTask) -> pd.DataFrame:
    result = frame.copy()
    result["job"] = task.job
    result["config_label"] = task.config_label
    result["max_depth"] = task.max_depth
    result["min_child_weight"] = task.min_child_weight
    result["include_missing_indicators"] = task.include_missing_indicators
    return result


def _task_key(task: RunTask) -> str:
    variant = task.variant or "base"
    return (
        f"{task.job}_{task.dataset}_{task.scenario}_s{task.scenario_seed}_m{task.model_seed}_"
        f"{task.config_label}_ind{int(task.include_missing_indicators)}_{variant}"
    )


def _checkpoint_path(directory: Path, task: RunTask) -> Path:
    return directory / f"{_task_key(task)}.pkl"


def _write_checkpoint(result: RunResult) -> None:
    """Persist one successful task as plain data (no custom classes in the pickle)."""
    if _CHECKPOINT_DIR is None:
        raise RuntimeError("checkpoint directory is not configured in this process")
    evaluation = result.evaluation
    payload = {
        "task": asdict(result.task),
        "metrics": None if evaluation is None else evaluation.metrics,
        "rankings": None if evaluation is None else evaluation.rankings,
        "train_auc": None if evaluation is None else evaluation.train_auc,
        "oof_auc": None if evaluation is None else evaluation.oof_auc,
        "n_samples": None if evaluation is None else evaluation.n_samples,
        "n_features": None if evaluation is None else evaluation.n_features,
        "error": result.error,
    }
    path = _checkpoint_path(_CHECKPOINT_DIR, result.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def _load_checkpoint_results(
    directory: Path,
    tasks: list[RunTask],
) -> tuple[list[RunResult], list[RunTask]]:
    completed: list[RunResult] = []
    remaining: list[RunTask] = []
    for task in tasks:
        path = _checkpoint_path(directory, task)
        if not path.is_file():
            remaining.append(task)
            continue
        payload = pickle.loads(path.read_bytes())
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint {path} does not contain a payload dictionary")
        checkpoint_task = RunTask(**payload["task"])
        if checkpoint_task != task:
            raise ValueError(f"checkpoint {path} task does not match the requested task")
        if payload["error"] is not None:
            raise RuntimeError(f"checkpoint {path} recorded a failed task: {payload['error']}")
        evaluation = RunEvaluation(
            metrics=payload["metrics"],
            rankings=payload["rankings"],
            train_auc=payload["train_auc"],
            oof_auc=payload["oof_auc"],
            n_samples=payload["n_samples"],
            n_features=payload["n_features"],
        )
        completed.append(RunResult(task=checkpoint_task, evaluation=evaluation))
    return completed, remaining


def run_task(task: RunTask) -> RunResult:
    """Execute one task, checkpoint it, and capture any failure for the parent."""
    try:
        if task.job == "wasserstein_scale":
            evaluation = _evaluate_wasserstein_scale(task)
        elif task.job == "main":
            bundle = load_bundle(task.dataset, task.scenario, task.scenario_seed, task.include_missing_indicators)
            evaluation = evaluate_main_run(
                bundle,
                dataset_label=DATASET_LABELS[task.dataset],
                model_seed=task.model_seed,
                model_params={"max_depth": task.max_depth, "min_child_weight": task.min_child_weight},
            )
        else:
            bundle = load_bundle(task.dataset, task.scenario, task.scenario_seed, task.include_missing_indicators)
            evaluation = evaluate_gads_run(
                bundle,
                dataset_label=DATASET_LABELS[task.dataset],
                model_seed=task.model_seed,
                model_params={"max_depth": task.max_depth, "min_child_weight": task.min_child_weight},
                fit_final_model=task.job != "model_seed",
            )
        metrics = _attach_task_context(evaluation.metrics, task)
        rankings = None if evaluation.rankings is None else evaluation.rankings.copy()
        result = RunResult(
            task=task,
            evaluation=RunEvaluation(
                metrics=metrics,
                rankings=rankings,
                train_auc=evaluation.train_auc,
                oof_auc=evaluation.oof_auc,
                n_samples=evaluation.n_samples,
                n_features=evaluation.n_features,
            ),
        )
        _write_checkpoint(result)
        return result
    except Exception:
        return RunResult(task=task, error=traceback.format_exc())


def _init_worker(checkpoint_dir: str) -> None:
    """Limit every worker to one BLAS thread and configure checkpointing."""
    global _CHECKPOINT_DIR
    _CHECKPOINT_DIR = Path(checkpoint_dir)
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
    model_seeds: tuple[int, ...],
) -> list[RunTask]:
    """Build every Pack 1 task exactly once."""
    tasks: list[RunTask] = [
        RunTask(
            job="model_seed",
            dataset=dataset,
            scenario=scenario,
            scenario_seed=scenario_seed,
            model_seed=model_seed,
            config_label=_config_for_dataset(dataset).label,
            max_depth=_config_for_dataset(dataset).max_depth,
            min_child_weight=_config_for_dataset(dataset).min_child_weight,
            include_missing_indicators=False,
        )
        for dataset, scenario, scenario_seed, model_seed in product(datasets, scenarios, scenario_seeds, model_seeds)
    ]
    if "b" in datasets:
        tasks.extend(
            RunTask(
                job="hyperparam",
                dataset="b",
                scenario=scenario,
                scenario_seed=scenario_seed,
                model_seed=42,
                config_label=config.label,
                max_depth=config.max_depth,
                min_child_weight=config.min_child_weight,
                include_missing_indicators=False,
            )
            for config, scenario, scenario_seed in product(DATASET_B_GRID, scenarios, scenario_seeds)
        )
        tasks.extend(
            RunTask(
                job="missing_indicator",
                dataset="b",
                scenario=scenario,
                scenario_seed=scenario_seed,
                model_seed=42,
                config_label=DATASET_B_CONFIG.label,
                max_depth=DATASET_B_CONFIG.max_depth,
                min_child_weight=DATASET_B_CONFIG.min_child_weight,
                include_missing_indicators=True,
            )
            for scenario, scenario_seed in product(scenarios, scenario_seeds)
        )
    tasks.extend(
        RunTask(
            job="main",
            dataset=dataset,
            scenario=scenario,
            scenario_seed=scenario_seed,
            model_seed=42,
            config_label=_config_for_dataset(dataset).label,
            max_depth=_config_for_dataset(dataset).max_depth,
            min_child_weight=_config_for_dataset(dataset).min_child_weight,
            include_missing_indicators=False,
        )
        for dataset, scenario, scenario_seed in product(datasets, scenarios, scenario_seeds)
    )
    if "a" in datasets:
        tasks.extend(
            RunTask(
                job="wasserstein_scale",
                dataset="a",
                scenario="S2",
                scenario_seed=scenario_seed,
                model_seed=42,
                config_label="wasserstein_x",
                max_depth=0,
                min_child_weight=0,
                include_missing_indicators=False,
                variant=variant,
            )
            for scenario_seed, variant in product(scenario_seeds, ("raw", "normal_pool_zscore"))
        )
    return tasks


def _collect_frames(
    results: list[RunResult],
    job: JobName,
    *,
    include_rankings: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        result.evaluation.metrics for result in results if result.task.job == job and result.evaluation is not None
    ]
    rankings = [
        result.evaluation.rankings
        for result in results
        if result.task.job == job and result.evaluation is not None and result.evaluation.rankings is not None
    ]
    if not metrics:
        raise ValueError(f"no metrics collected for job {job!r}")
    metrics_frame = pd.concat(metrics, ignore_index=True)
    if not include_rankings:
        return metrics_frame, pd.DataFrame()
    rankings_frame = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return metrics_frame, rankings_frame


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    return frame.sort_values(present, kind="stable").reset_index(drop=True)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
        raise ValueError("arrays differ in their missing-value pattern")
    mask = ~np.isnan(left_values)
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(left_values[mask] - right_values[mask])))


def _verify_protocol(dataset_keys: tuple[str, ...]) -> dict[str, Any]:
    """Check the lean GADS path against ``run_experiment_bundle`` exactly."""
    records: list[dict[str, Any]] = []
    for dataset in dataset_keys:
        config = _config_for_dataset(dataset)
        bundle = load_bundle(dataset, "S1", 0, False)
        lean = evaluate_gads_run(
            bundle,
            dataset_label=DATASET_LABELS[dataset],
            model_seed=42,
            model_params=config.as_model_params(),
            fit_final_model=False,
        )
        reference = run_experiment_bundle(
            bundle,
            settings=ExperimentSettings(
                methods=("GADS",),
                model_type="xgboost",
                model_seed=42,
                cv_seed=None,
                cv_splits=5,
                model_params=config.as_model_params(),
                top_ks=(1, 3, 5),
                auc_threshold=None,
                psi_bins=10,
                gads_distance="wasserstein",
                gads_bins=DEFAULT_GADS_BINS,
                gads_epsilon=DEFAULT_GADS_EPSILON,
                eqp_mode="contextual_process",
            ),
        )
        metric_columns = ["mrr", "hr_at_1", "hr_at_3", "hr_at_5", "far_at_1", "far_at_3", "far_at_5", "kpi_auc"]
        metric_difference = max(
            _max_abs_difference(
                lean.metrics[column].to_numpy(dtype=float),
                reference.metrics[column].to_numpy(dtype=float),
            )
            for column in metric_columns
        )
        lean_rankings = lean.rankings.sort_values(["equipment", "feature"], kind="stable").reset_index(drop=True)
        reference_rankings = (
            reference.rankings.loc[reference.rankings["view"] == "process"]
            .sort_values(["equipment", "feature"], kind="stable")
            .reset_index(drop=True)
        )
        score_difference = _max_abs_difference(
            lean_rankings["score"].to_numpy(dtype=float),
            reference_rankings["score"].to_numpy(dtype=float),
        )
        rank_difference = _max_abs_difference(
            lean_rankings["rank"].to_numpy(dtype=float),
            reference_rankings["rank"].to_numpy(dtype=float),
        )
        passed = metric_difference <= 1e-10 and score_difference <= 1e-10 and rank_difference == 0.0
        records.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "max_metric_abs_diff": metric_difference,
                "max_score_abs_diff": score_difference,
                "max_rank_abs_diff": rank_difference,
                "passed": passed,
            }
        )
    if not all(record["passed"] for record in records):
        raise RuntimeError(f"protocol self-check failed: {records}")
    return {"passed": True, "checks": records}


def _melt_metrics(metrics: pd.DataFrame, value_vars: tuple[str, ...]) -> pd.DataFrame:
    identifier_columns = [
        column
        for column in (
            "dataset",
            "scenario",
            "scenario_seed",
            "model_seed",
            "method",
            "equipment",
            "config_label",
            "max_depth",
            "min_child_weight",
            "include_missing_indicators",
            "variant",
        )
        if column in metrics.columns
    ]
    return metrics.melt(
        id_vars=identifier_columns,
        value_vars=list(value_vars),
        var_name="metric",
        value_name="value",
    )


def _per_model_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    long = _melt_metrics(metrics, METRIC_COLUMNS)
    summary = long.groupby(["dataset", "scenario", "metric", "model_seed"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        minimum=("value", "min"),
        maximum=("value", "max"),
        n=("value", "count"),
    )
    return _sort_frame(summary.loc[summary["n"] > 0], ["dataset", "scenario", "metric", "model_seed"])


def _variance_decomposition(metrics: pd.DataFrame) -> pd.DataFrame:
    long = _melt_metrics(metrics, METRIC_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, metric), group in long.groupby(["dataset", "scenario", "metric"], sort=False):
        applicable, _note = metric_applicable(str(scenario), str(metric))
        if not applicable:
            continue
        pivot = group.pivot_table(index="scenario_seed", columns="model_seed", values="value", aggfunc="first")
        if pivot.isna().to_numpy().any():
            raise ValueError(f"missing values in applicable metric {dataset}/{scenario}/{metric}")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            raise ValueError(f"variance decomposition needs >=2 scenario and model seeds: {dataset}/{scenario}")
        within_variance = float(pivot.var(axis=1, ddof=1).mean())
        scenario_means = pivot.mean(axis=1)
        between_variance = float(scenario_means.var(ddof=1))
        total_variance = within_variance + between_variance
        model_seed_means = pivot.mean(axis=0)
        seed_42_deviation = (
            float((model_seed_means - model_seed_means.loc[42]).abs().max()) if 42 in model_seed_means else np.nan
        )
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "metric": metric,
                "grand_mean": float(pivot.to_numpy().mean()),
                "within_var": within_variance,
                "within_sd": math.sqrt(within_variance),
                "between_var": between_variance,
                "between_sd": math.sqrt(between_variance),
                "total_var": total_variance,
                "between_share": between_variance / total_variance if total_variance > 0.0 else np.nan,
                "within_share": within_variance / total_variance if total_variance > 0.0 else np.nan,
                "between_within_ratio": between_variance / within_variance if within_variance > 0.0 else np.nan,
                "model_seed_sd": float(model_seed_means.std(ddof=1)),
                "model_seed_range": float(model_seed_means.max() - model_seed_means.min()),
                "max_abs_deviation_from_seed42": seed_42_deviation,
                "n_scenario_seeds": int(pivot.shape[0]),
                "n_model_seeds": int(pivot.shape[1]),
                "degenerate": total_variance == 0.0,
            }
        )
    if not rows:
        raise ValueError("variance decomposition produced no rows")
    return _sort_frame(pd.DataFrame(rows), ["dataset", "scenario", "metric"])


def _absolute_precision_ci(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    model_seeds: tuple[int, ...],
) -> pd.DataFrame:
    """Return 95% CIs for absolute metric means of the model-seed sweep.

    Two scopes are reported per dataset × scenario × metric cell:

    * ``model_seed_42`` — the 20 scenario-seed values at the paper model seed.
    * ``scenario_cluster_5_model_seeds`` — each ``scenario_seed`` is one
      independent cluster: the five model seeds are averaged within the cluster
      first, then the interval bootstraps over the 20 scenario clusters. This is
      the only valid unit of resampling for the 5-model-seed × 20-scenario-seed
      table, because model seeds share the same scenario realisation.
    """
    long = _melt_metrics(metrics, (*METRIC_COLUMNS, "kpi_auc"))
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, metric), group in long.groupby(["dataset", "scenario", "metric"], sort=False):
        applicable, note = metric_applicable(str(scenario), str(metric))
        if not applicable:
            continue
        pivot = group.pivot_table(index="scenario_seed", columns="model_seed", values="value", aggfunc="first")
        missing = [seed for seed in model_seeds if seed not in pivot.columns]
        if missing:
            raise ValueError(f"model seeds missing from {dataset}/{scenario}/{metric}: {missing}")
        pivot = pivot.reindex(columns=list(model_seeds))
        if pivot.isna().to_numpy().any():
            raise ValueError(f"non-finite values in {dataset}/{scenario}/{metric}")
        if 42 not in pivot.columns:
            raise ValueError("absolute precision requires model seed 42")
        scopes = (
            ("model_seed_42", pivot[42].to_numpy(dtype=float), "absolute_seed42"),
            ("scenario_cluster_5_model_seeds", pivot.mean(axis=1).to_numpy(dtype=float), "absolute_cluster5"),
        )
        for scope, values, seed_tag in scopes:
            intervals = _bootstrap_mean_ci(
                values,
                n_resamples=bootstrap_samples,
                confidence=0.95,
                seed=_comparison_seed(str(dataset), str(scenario), str(metric), seed_tag),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "metric": metric,
                    "scope": scope,
                    "n_clusters": int(values.size),
                    "n_model_seeds": 1 if scope == "model_seed_42" else len(model_seeds),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    **intervals,
                    "applicability_note": note,
                }
            )
    if not rows:
        raise ValueError("absolute precision table produced no rows")
    result = _sort_frame(pd.DataFrame(rows), ["dataset", "scenario", "metric", "scope"])
    metric_order = {metric: index for index, metric in enumerate((*METRIC_COLUMNS, "kpi_auc"))}
    result["metric_order"] = result["metric"].map(metric_order)
    result = result.sort_values(["dataset", "scenario", "metric_order", "scope"], kind="stable")
    return result.drop(columns="metric_order").reset_index(drop=True)


def _rank_correlations(rankings: pd.DataFrame, model_seeds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "scenario", "scenario_seed", "equipment"]
    for key, group in rankings.groupby(group_columns, sort=False):
        dataset, scenario, scenario_seed, equipment = key
        pivot = group.pivot_table(index="feature", columns="model_seed", values="score", aggfunc="first")
        missing = [seed for seed in model_seeds if seed not in pivot.columns]
        if missing:
            raise ValueError(f"model seeds missing from ranking slice {key}: {missing}")
        if pivot.isna().to_numpy().any():
            raise ValueError(f"feature scores missing across model seeds in slice {key}")
        for left_index, left_seed in enumerate(model_seeds):
            for right_seed in model_seeds[left_index + 1 :]:
                rho = float(spearmanr(pivot[left_seed], pivot[right_seed]).statistic)
                tau = float(kendalltau(pivot[left_seed], pivot[right_seed]).statistic)
                if not (np.isfinite(rho) and np.isfinite(tau)):
                    raise ValueError(f"non-finite rank correlation in slice {key} for seeds {left_seed}/{right_seed}")
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "scenario_seed": int(scenario_seed),
                        "equipment": equipment,
                        "model_seed_left": int(left_seed),
                        "model_seed_right": int(right_seed),
                        "spearman": rho,
                        "kendall": tau,
                        "n_features": int(pivot.shape[0]),
                    }
                )
    if not rows:
        raise ValueError("no model-seed rank correlations computed")
    pairs = _sort_frame(pd.DataFrame(rows), ["dataset", "scenario", "scenario_seed", "equipment", "model_seed_left"])
    summary = (
        pairs.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            q25_spearman=("spearman", lambda values: values.quantile(0.25)),
            q75_spearman=("spearman", lambda values: values.quantile(0.75)),
            max_spearman=("spearman", "max"),
            frac_spearman_ge_0_8=("spearman", lambda values: float((values >= 0.8).mean())),
            mean_kendall=("kendall", "mean"),
            median_kendall=("kendall", "median"),
            min_kendall=("kendall", "min"),
            frac_kendall_ge_0_6=("kendall", lambda values: float((values >= 0.6).mean())),
            n_pairs=("spearman", "size"),
            n_scenario_seeds=("scenario_seed", "nunique"),
            n_equipment=("equipment", "nunique"),
        )
        .reset_index(drop=True)
    )
    overall = (
        pairs.groupby(["dataset"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            mean_kendall=("kendall", "mean"),
            median_kendall=("kendall", "median"),
            min_kendall=("kendall", "min"),
            frac_spearman_ge_0_8=("spearman", lambda values: float((values >= 0.8).mean())),
            frac_kendall_ge_0_6=("kendall", lambda values: float((values >= 0.6).mean())),
            n_pairs=("spearman", "size"),
        )
        .reset_index(drop=True)
    )
    return pairs, pd.concat(
        [summary.assign(scope="scenario"), overall.assign(scope="dataset", scenario="ALL")],
        ignore_index=True,
    )


def _top1_stability(
    rankings: pd.DataFrame,
    *,
    group_columns: list[str],
    seed_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = rankings.sort_values([*group_columns, seed_column, "rank", "feature"], kind="stable")
    top1 = ordered.groupby([*group_columns, seed_column], sort=False).head(1).copy()
    rows: list[dict[str, Any]] = []
    for key, group in top1.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values, strict=True))
        counts = group["feature"].value_counts()
        modal_top1 = str(counts.index[0])
        modal_share = float(counts.iloc[0] / len(group))
        n_distinct = int(counts.size)
        applicable, _note = metric_applicable(str(record["scenario"]), "mrr")
        top1_root_rate = float(group["is_root_cause"].mean()) if applicable else np.nan
        modal_is_root = (
            bool(group.loc[group["feature"] == modal_top1, "is_root_cause"].iloc[0]) if applicable else np.nan
        )
        record.update(
            {
                "n_seeds": len(group),
                "modal_top1": modal_top1,
                "modal_share": modal_share,
                "n_distinct_top1": n_distinct,
                "all_same": bool(n_distinct == 1),
                "top1_root_rate": top1_root_rate,
                "modal_is_root": modal_is_root,
            }
        )
        rows.append(record)
    if not rows:
        raise ValueError("top-1 stability produced no rows")
    per_group = _sort_frame(pd.DataFrame(rows), [*group_columns])
    summary_columns = [*group_columns]
    if "equipment" in summary_columns:
        summary_columns.remove("equipment")
    summary = (
        per_group.groupby(summary_columns, as_index=False)
        .agg(
            n_groups=("modal_share", "size"),
            mean_modal_share=("modal_share", "mean"),
            min_modal_share=("modal_share", "min"),
            mean_n_distinct_top1=("n_distinct_top1", "mean"),
            mean_top1_root_rate=("top1_root_rate", "mean"),
            fraction_modal_is_root=("modal_is_root", "mean"),
            fraction_all_same=("all_same", "mean"),
        )
        .reset_index(drop=True)
    )
    return per_group, summary


def _topk_jaccard(
    rankings: pd.DataFrame,
    *,
    group_columns: list[str],
    seed_column: str,
    top_k: int,
    seed_kind: str,
    summary_group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = rankings.sort_values([*group_columns, seed_column, "rank", "feature"], kind="stable")
    top = ordered.groupby([*group_columns, seed_column], sort=False).head(top_k)
    feature_sets = (
        top.groupby([*group_columns, seed_column], sort=False)["feature"]
        .agg(lambda values: frozenset(values))
        .reset_index(name="features")
    )
    rows: list[dict[str, Any]] = []
    for key, group in feature_sets.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values, strict=True))
        values = dict(zip(group[seed_column], group["features"], strict=True))
        seeds = sorted(values)
        if len(seeds) < 2:
            continue
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                left_set = values[left_seed]
                right_set = values[right_seed]
                union = left_set | right_set
                jaccard = float(len(left_set & right_set) / len(union)) if union else np.nan
                rows.append(
                    {
                        **record,
                        "seed_kind": seed_kind,
                        "top_k": top_k,
                        "seed_left": int(left_seed),
                        "seed_right": int(right_seed),
                        "jaccard": jaccard,
                    }
                )
    if not rows:
        raise ValueError("no Jaccard pairs computed")
    pairs = _sort_frame(pd.DataFrame(rows), [*group_columns, "seed_left", "seed_right"])
    if summary_group_columns is None:
        summary_columns = [*group_columns]
        if "equipment" in summary_columns:
            summary_columns.remove("equipment")
    else:
        summary_columns = list(summary_group_columns)
    summary = (
        pairs.groupby(summary_columns, as_index=False)
        .agg(
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            min_jaccard=("jaccard", "min"),
            q25_jaccard=("jaccard", lambda values: values.quantile(0.25)),
            q75_jaccard=("jaccard", lambda values: values.quantile(0.75)),
            max_jaccard=("jaccard", "max"),
            n_pairs=("jaccard", "size"),
        )
        .reset_index(drop=True)
    )
    return pairs, summary


def _top_rank_composition(
    main_rankings: pd.DataFrame,
    model_rankings: pd.DataFrame,
    *,
    dataset: str = "Dataset A",
    scenarios: tuple[str, ...] = ("S1", "S2", "S3", "S4-main"),
    rank_min: int = 2,
    rank_max: int = 5,
) -> pd.DataFrame:
    """Classify the feature families that occupy ranks ``rank_min..rank_max``.

    The Top-5 Jaccard is low largely because the runner-up ranks churn.  This
    table shows what actually fills those ranks: the ten KPI-linked
    ``feat_useful_*`` features (whose linear logit weights are redrawn for every
    scenario seed), the ten ``feat_useless_*`` harmless companions, and the
    remaining pure-noise ``feat_useless_*`` features.  Both the single-model
    main matrix (model seed 42) and the five-model-seed sweep are reported, plus
    their pooled combination.
    """
    columns = [
        "dataset",
        "scenario",
        "feature",
        "rank",
        "is_root_cause",
        "is_harmless",
        "method",
    ]
    sources = {
        "main_rankings": main_rankings.loc[:, columns],
        "model_seed_sweep": model_rankings.loc[:, columns],
    }
    classified: list[pd.DataFrame] = []
    for source, frame in sources.items():
        subset = frame.loc[
            (frame["dataset"] == dataset)
            & (frame["method"] == "GADS")
            & (frame["scenario"].isin(scenarios))
            & (frame["rank"].between(rank_min, rank_max))
        ].copy()
        if subset.empty:
            raise ValueError(f"no rank {rank_min}-{rank_max} rows for {source} on {dataset}")
        subset["source"] = source
        classified.append(subset)
    rows: list[dict[str, Any]] = []
    sources_frame = pd.concat(classified, ignore_index=True)
    sources_frame["is_useful"] = sources_frame["feature"].str.startswith("feat_useful_")
    sources_frame["is_harmless"] = sources_frame["is_harmless"].astype(bool)
    sources_frame["family"] = np.select(
        [
            sources_frame["is_useful"],
            sources_frame["is_harmless"],
            sources_frame["is_root_cause"].astype(bool),
        ],
        ["feat_useful", "feat_harmless", "root_cause"],
        default="feat_noise",
    )
    for source, frame in [
        *[(name, sources_frame.loc[sources_frame["source"] == name]) for name in sources],
        ("combined", sources_frame),
    ]:
        groups = [(scenario, frame.loc[frame["scenario"] == scenario]) for scenario in scenarios]
        groups.append(("ALL", frame))
        for scenario, group in groups:
            counts = group["family"].value_counts()
            n_positions = len(group)
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "scenario": scenario,
                    "rank_min": rank_min,
                    "rank_max": rank_max,
                    "n_positions": n_positions,
                    "n_feat_useful": int(counts.get("feat_useful", 0)),
                    "frac_feat_useful": float(counts.get("feat_useful", 0) / n_positions),
                    "n_feat_harmless": int(counts.get("feat_harmless", 0)),
                    "frac_feat_harmless": float(counts.get("feat_harmless", 0) / n_positions),
                    "n_feat_noise": int(counts.get("feat_noise", 0)),
                    "frac_feat_noise": float(counts.get("feat_noise", 0) / n_positions),
                    "n_root_cause": int(counts.get("root_cause", 0)),
                    "frac_root_cause": float(counts.get("root_cause", 0) / n_positions),
                }
            )
    return _sort_frame(pd.DataFrame(rows), ["dataset", "source", "scenario"])


def _aggregate_top1_summary(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        summary.groupby(group_columns, as_index=False)
        .agg(
            n_groups=("n_groups", "sum"),
            mean_modal_share=("mean_modal_share", "mean"),
            min_modal_share=("min_modal_share", "min"),
            mean_n_distinct_top1=("mean_n_distinct_top1", "mean"),
            mean_top1_root_rate=("mean_top1_root_rate", "mean"),
            fraction_modal_is_root=("fraction_modal_is_root", "mean"),
            fraction_all_same=("fraction_all_same", "mean"),
        )
        .reset_index(drop=True)
    )


def _rank_summary(ranks: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        ranks.groupby(group_columns, as_index=False)
        .agg(
            n_obs=("root_cause_rank", "size"),
            median_rank=("root_cause_rank", "median"),
            q1_rank=("root_cause_rank", lambda values: values.quantile(0.25)),
            q3_rank=("root_cause_rank", lambda values: values.quantile(0.75)),
            mean_rank=("root_cause_rank", "mean"),
            min_rank=("root_cause_rank", "min"),
            max_rank=("root_cause_rank", "max"),
        )
        .reset_index(drop=True)
    )
    summary["iqr_rank"] = summary["q3_rank"] - summary["q1_rank"]
    rates = (
        ranks.assign(
            top1=(ranks["root_cause_rank"] <= 1).astype(float),
            top3=(ranks["root_cause_rank"] <= 3).astype(float),
            top5=(ranks["root_cause_rank"] <= 5).astype(float),
        )
        .groupby(group_columns, as_index=False)[["top1", "top3", "top5"]]
        .mean()
        .rename(columns={"top1": "top1_rate", "top3": "top3_rate", "top5": "top5_rate"})
    )
    return summary.merge(rates, on=group_columns, validate="one_to_one")


def _root_cause_rank_tables(rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranks = rankings.loc[
        rankings["is_root_cause"],
        [
            "dataset",
            "scenario",
            "scenario_seed",
            "model_seed",
            "method",
            "equipment",
            "feature",
            "rank",
        ],
    ].rename(columns={"feature": "root_cause_feature", "rank": "root_cause_rank"})
    if ranks.empty:
        raise ValueError("no annotated root-cause ranks found")
    per_scenario = _rank_summary(ranks, ["dataset", "scenario", "method"])
    overall = _rank_summary(ranks, ["dataset", "method"])
    return ranks, per_scenario, overall.assign(scenario="ALL")


def _comparison_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | str]:
    """Return percentile and BCa bootstrap intervals for a mean.

    BCa delegates to ``scipy.stats.bootstrap``. SciPy's bias correction uses the
    mid-rank percentile ``(count(< theta_hat) + 0.5 * count(== theta_hat)) / B``,
    which is the correct tie handling for discrete metrics such as HR@1 and
    FAR@1; a plain ``count(< theta_hat) / B`` distorts ``z0`` when many
    replicates tie with the observed statistic. The same resampling stream is
    reused for the percentile bounds, so for any input ``x`` and seed the BCa
    interval of ``-x`` is the exact negation of the interval of ``x``.

    Constant input (zero variance) or a degenerate jackknife acceleration makes
    BCa undefined; those rows fall back to the percentile interval and are
    marked in ``bootstrap_method``.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("bootstrap interval requires at least two finite values")
    if not np.isfinite(data).all():
        raise ValueError("bootstrap interval requires finite values")
    observed = float(data.mean())
    alpha = (1.0 - confidence) / 2.0
    if float(np.ptp(data)) == 0.0:
        return {
            "ci95_percentile_low": observed,
            "ci95_percentile_high": observed,
            "ci95_bca_low": observed,
            "ci95_bca_high": observed,
            "bootstrap_bias_correction": 0.0,
            "bootstrap_acceleration": 0.0,
            "bootstrap_method": "degenerate_constant",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bootstrap(
            (data,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="BCa",
            random_state=np.random.default_rng(seed),
        )
    bootstrap_means = np.asarray(result.bootstrap_distribution, dtype=float)
    percentile_low, percentile_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    bca_low = float(result.confidence_interval.low)
    bca_high = float(result.confidence_interval.high)
    ties = np.count_nonzero(bootstrap_means == observed)
    proportion = (np.count_nonzero(bootstrap_means < observed) + 0.5 * ties) / bootstrap_means.size
    bias_correction = float(norm.ppf(proportion))
    n_observations = data.size
    jackknife = (data.sum() - data) / (n_observations - 1)
    deviations = float(jackknife.mean()) - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    acceleration = float(np.sum(deviations**3) / denominator) if denominator > 0.0 else float("nan")
    if math.isfinite(bca_low) and math.isfinite(bca_high):
        method = "bca"
    else:
        bca_low, bca_high = float(percentile_low), float(percentile_high)
        method = "percentile_fallback"
    return {
        "ci95_percentile_low": float(percentile_low),
        "ci95_percentile_high": float(percentile_high),
        "ci95_bca_low": bca_low,
        "ci95_bca_high": bca_high,
        "bootstrap_bias_correction": bias_correction,
        "bootstrap_acceleration": acceleration,
        "bootstrap_method": method,
    }


def _metric_higher_is_better(metric: str) -> bool:
    return metric not in FAR_METRICS and metric != "train_oof_auc_gap"


def _effect_sizes(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> pd.DataFrame:
    long = _melt_metrics(metrics, METRIC_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, metric), group in long.groupby(["dataset", "scenario", "metric"], sort=False):
        applicable, note = metric_applicable(str(scenario), str(metric))
        if not applicable:
            continue
        pivot = group.pivot_table(index="scenario_seed", columns="method", values="value", aggfunc="first")
        missing_methods = set(ALL_METHODS) - set(pivot.columns)
        if missing_methods:
            raise ValueError(f"methods missing from main matrix: {sorted(missing_methods)}")
        if pivot.isna().to_numpy().any():
            raise ValueError(f"non-finite main-matrix metric for {dataset}/{scenario}/{metric}")
        baselines = sorted(set(pivot.columns) - {"GADS"})
        group_rows: list[dict[str, Any]] = []
        for baseline in baselines:
            gads_values = pivot["GADS"].to_numpy(dtype=float)
            baseline_values = pivot[baseline].to_numpy(dtype=float)
            differences = gads_values - baseline_values
            mean_difference = float(differences.mean())
            median_difference = float(np.median(differences))
            sd_difference = float(differences.std(ddof=1))
            higher_is_better = _metric_higher_is_better(str(metric))
            comparison_seed = _comparison_seed(str(dataset), str(scenario), str(metric), str(baseline))
            intervals = _bootstrap_mean_ci(
                differences,
                n_resamples=bootstrap_samples,
                confidence=0.95,
                seed=comparison_seed,
            )
            _observed, p_value = paired_permutation_test(
                gads_values,
                baseline_values,
                n_resamples=permutation_samples,
                seed=comparison_seed,
            )
            ci_excludes_zero = intervals["ci95_bca_low"] > 0.0 or intervals["ci95_bca_high"] < 0.0
            if not ci_excludes_zero:
                favors = "inconclusive"
            elif (mean_difference > 0.0) == higher_is_better:
                favors = "GADS"
            else:
                favors = "baseline"
            group_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "baseline": baseline,
                    "n_pairs": int(differences.size),
                    "gads_mean": float(gads_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "mean_diff": mean_difference,
                    "median_diff": median_difference,
                    "sd_diff": sd_difference,
                    "cohen_dz": mean_difference / sd_difference if sd_difference > 0.0 else np.nan,
                    **intervals,
                    "p_value_permutation": float(p_value),
                    "ci_excludes_zero": bool(ci_excludes_zero),
                    "favors": favors,
                    "applicability_note": note,
                }
            )
        adjusted = holm_adjust([row["p_value_permutation"] for row in group_rows])
        for row, p_adjusted in zip(group_rows, adjusted, strict=True):
            row["p_value_holm"] = float(p_adjusted)
        rows.extend(group_rows)
    if not rows:
        raise ValueError("effect-size table produced no rows")
    return _sort_frame(pd.DataFrame(rows), ["dataset", "scenario", "metric", "baseline"])


def _hyperparameter_metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    long = _melt_metrics(metrics, EFFECT_METRICS)
    summary = long.groupby(
        [
            "dataset",
            "config_label",
            "max_depth",
            "min_child_weight",
            "include_missing_indicators",
            "scenario",
            "metric",
        ],
        as_index=False,
    ).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        minimum=("value", "min"),
        maximum=("value", "max"),
        n=("value", "count"),
    )
    return _sort_frame(summary, ["config_label", "scenario", "metric"])


def _config_overall(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["config_label", "max_depth", "min_child_weight", "include_missing_indicators"]
    for key, group in metrics.groupby(group_columns, sort=False):
        config_label, max_depth, min_child_weight, include_missing = key
        record: dict[str, Any] = {
            "config_label": str(config_label),
            "max_depth": int(max_depth),
            "min_child_weight": int(min_child_weight),
            "include_missing_indicators": bool(include_missing),
            "n_runs": len(group),
            "kpi_auc": float(group["kpi_auc"].mean()),
            "train_auc": float(group["train_auc"].mean()),
            "train_oof_auc_gap": float(group["train_oof_auc_gap"].mean()),
        }
        for metric in METRIC_COLUMNS:
            applicable_scenarios = {scenario for scenario in SCENARIOS if metric_applicable(scenario, metric)[0]}
            applicability = group["scenario"].isin(applicable_scenarios)
            applicable_values = group.loc[applicability, metric]
            record[metric] = float(applicable_values.mean()) if not applicable_values.empty else np.nan
            record[f"n_{metric}_obs"] = int(applicable_values.notna().sum())
        rows.append(record)
    result = pd.DataFrame(rows)
    result["config_order"] = result["config_label"].map(CONFIG_ORDER)
    return _sort_frame(result, ["config_order", "include_missing_indicators"])


def _missing_indicator_comparison(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
) -> pd.DataFrame:
    long = _melt_metrics(metrics, EFFECT_METRICS)
    included = long.loc[long["include_missing_indicators"]]
    excluded = long.loc[(~long["include_missing_indicators"]) & (long["config_label"] == DATASET_B_CONFIG.label)]
    if included.empty or excluded.empty:
        raise ValueError("missing-indicator comparison requires both indicator settings")
    merged = included.merge(
        excluded,
        on=["dataset", "scenario", "scenario_seed", "metric"],
        suffixes=("_included", "_excluded"),
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, metric), group in merged.groupby(["dataset", "scenario", "metric"], sort=False):
        if metric in METRIC_COLUMNS:
            applicable, note = metric_applicable(str(scenario), str(metric))
        else:
            applicable, note = True, ""
        if not applicable:
            continue
        included_values = group["value_included"].to_numpy(dtype=float)
        excluded_values = group["value_excluded"].to_numpy(dtype=float)
        if not (np.isfinite(included_values).all() and np.isfinite(excluded_values).all()):
            raise ValueError(f"non-finite missing-indicator comparison for {dataset}/{scenario}/{metric}")
        differences = included_values - excluded_values
        comparison_seed = _comparison_seed(str(dataset), str(scenario), str(metric), "missing_indicator")
        intervals = _bootstrap_mean_ci(
            differences,
            n_resamples=bootstrap_samples,
            confidence=0.95,
            seed=comparison_seed,
        )
        _observed, p_value = paired_permutation_test(
            included_values,
            excluded_values,
            n_resamples=permutation_samples,
            seed=comparison_seed,
        )
        higher_is_better = _metric_higher_is_better(str(metric))
        mean_difference = float(differences.mean())
        ci_excludes_zero = intervals["ci95_bca_low"] > 0.0 or intervals["ci95_bca_high"] < 0.0
        if not ci_excludes_zero:
            favors = "inconclusive"
        elif (mean_difference > 0.0) == higher_is_better:
            favors = "include_missing_indicators"
        else:
            favors = "exclude_missing_indicators"
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "n_pairs": int(differences.size),
                "included_mean": float(included_values.mean()),
                "excluded_mean": float(excluded_values.mean()),
                "mean_diff": mean_difference,
                **intervals,
                "p_value_permutation": float(p_value),
                "ci_excludes_zero": bool(ci_excludes_zero),
                "favors": favors,
                "applicability_note": note,
            }
        )
    if not rows:
        raise ValueError("missing-indicator comparison produced no rows")
    return _sort_frame(pd.DataFrame(rows), ["dataset", "scenario", "metric"])


def _wasserstein_scale_tables(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = _melt_metrics(metrics, METRIC_COLUMNS)
    by_variant = (
        long.groupby(["dataset", "scenario", "variant", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            minimum=("value", "min"),
            maximum=("value", "max"),
            n=("value", "count"),
        )
        .reset_index(drop=True)
    )
    by_variant = by_variant.loc[by_variant["n"] > 0].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, metric), group in long.groupby(["dataset", "scenario", "metric"], sort=False):
        applicable, _note = metric_applicable(str(scenario), str(metric))
        if not applicable:
            continue
        pivot = group.pivot_table(index="scenario_seed", columns="variant", values="value", aggfunc="first")
        required = {"raw", "normal_pool_zscore"}
        if not required <= set(pivot.columns):
            raise ValueError(f"Wasserstein-X variants missing for {dataset}/{scenario}/{metric}")
        if pivot.isna().to_numpy().any():
            raise ValueError(f"non-finite Wasserstein-X metric for {dataset}/{scenario}/{metric}")
        raw_values = pivot["raw"].to_numpy(dtype=float)
        standardized_values = pivot["normal_pool_zscore"].to_numpy(dtype=float)
        differences = standardized_values - raw_values
        comparison_seed = _comparison_seed(str(dataset), str(scenario), str(metric), "wasserstein_scale")
        intervals = _bootstrap_mean_ci(
            differences,
            n_resamples=bootstrap_samples,
            confidence=0.95,
            seed=comparison_seed,
        )
        _observed, p_value = paired_permutation_test(
            standardized_values,
            raw_values,
            n_resamples=permutation_samples,
            seed=comparison_seed,
        )
        higher_is_better = _metric_higher_is_better(str(metric))
        mean_difference = float(differences.mean())
        ci_excludes_zero = intervals["ci95_bca_low"] > 0.0 or intervals["ci95_bca_high"] < 0.0
        if not ci_excludes_zero:
            favors = "inconclusive"
        elif (mean_difference > 0.0) == higher_is_better:
            favors = "standardized"
        else:
            favors = "raw"
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "n_pairs": int(differences.size),
                "raw_mean": float(raw_values.mean()),
                "standardized_mean": float(standardized_values.mean()),
                "mean_diff_standardized_minus_raw": mean_difference,
                **intervals,
                "p_value_permutation": float(p_value),
                "ci_excludes_zero": bool(ci_excludes_zero),
                "favors": favors,
            }
        )
    if not rows:
        raise ValueError("Wasserstein-X scale comparison produced no rows")
    return by_variant, _sort_frame(pd.DataFrame(rows), ["scenario", "metric"])


def _validate_main_against_model_sweep(main_metrics: pd.DataFrame, sweep_metrics: pd.DataFrame) -> dict[str, Any]:
    main_gads = main_metrics.loc[main_metrics["method"] == "GADS"]
    sweep_gads = sweep_metrics.loc[sweep_metrics["model_seed"] == 42]
    keys = ["dataset", "scenario", "scenario_seed"]
    merged = main_gads.merge(sweep_gads, on=keys, suffixes=("_main", "_sweep"), validate="one_to_one")
    if merged.empty:
        raise ValueError("main matrix and model-seed sweep do not overlap")
    differences = {
        metric: _max_abs_difference(
            merged[f"{metric}_main"].to_numpy(dtype=float),
            merged[f"{metric}_sweep"].to_numpy(dtype=float),
        )
        for metric in [*METRIC_COLUMNS, "kpi_auc"]
    }
    max_difference = max(differences.values())
    if max_difference > 1e-10:
        raise RuntimeError(f"recomputed main-matrix GADS differs from the model-seed sweep: {differences}")
    return {"passed": True, "n_runs": len(merged), "max_abs_diff": max_difference, "per_metric": differences}


def _to_builtin(value: Any) -> Any:
    if value is None:
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


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return _to_builtin(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _to_builtin(value) for key, value in record.items()} for record in frame.to_dict(orient="records")
    ]


def _md_table(frame: pd.DataFrame, columns: list[str], *, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "_No rows._"
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"markdown table is missing columns: {missing}")
    display = frame.loc[:, columns].astype(object)
    display = display.where(pd.notna(display), None)
    return display.to_markdown(index=False, floatfmt=floatfmt, missingval="--")


def _pivot_metric(
    per_model: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    model_seeds: tuple[int, ...],
) -> pd.DataFrame:
    subset = per_model.loc[(per_model["dataset"] == dataset) & (per_model["metric"] == metric)]
    present_scenarios = [scenario for scenario in SCENARIOS if scenario in set(subset["scenario"])]
    pivot = subset.pivot_table(index="scenario", columns="model_seed", values="mean", aggfunc="first").reindex(
        present_scenarios
    )
    pivot = pivot.reindex(columns=list(model_seeds))
    pivot = pivot.reset_index()
    return pivot.rename(columns={seed: f"seed {seed}" for seed in model_seeds})


def _pivot_far(
    per_model: pd.DataFrame,
    *,
    model_seeds: tuple[int, ...],
) -> pd.DataFrame:
    subset = per_model.loc[(per_model["metric"] == "far_at_1") & (per_model["scenario"].isin(["S4-main", "S4-null"]))]
    pivot = subset.pivot_table(index=["dataset", "scenario"], columns="model_seed", values="mean", aggfunc="first")
    pivot = pivot.reindex(columns=list(model_seeds)).reset_index()
    return pivot.rename(columns={seed: f"seed {seed}" for seed in model_seeds})


def _select_effect_summary(effect_sizes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, metric), group in effect_sizes.groupby(["dataset", "metric"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "n_comparisons": len(group),
                "ci_excludes_zero": int(group["ci_excludes_zero"].sum()),
                "gads_favored": int((group["favors"] == "GADS").sum()),
                "baseline_favored": int((group["favors"] == "baseline").sum()),
                "inconclusive": int((group["favors"] == "inconclusive").sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_findings(
    per_model: pd.DataFrame,
    variance: pd.DataFrame,
    rank_correlation: pd.DataFrame,
    config_overall: pd.DataFrame,
    missing_comparison: pd.DataFrame,
    wasserstein_paired: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    absolute_precision: pd.DataFrame,
    root_summary: pd.DataFrame,
    top1_model_summary: pd.DataFrame,
    top1_scenario_summary: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    rank_composition: pd.DataFrame,
    model_seeds: tuple[int, ...],
) -> dict[str, Any]:
    mrr_per_model = per_model.loc[per_model["metric"] == "mrr"]
    mrr_nested: dict[str, dict[str, dict[str, float]]] = {}
    for (dataset, scenario), group in mrr_per_model.groupby(["dataset", "scenario"], sort=False):
        mrr_nested.setdefault(str(dataset), {})[str(scenario)] = {
            str(int(row.model_seed)): float(row.mean) for row in group.itertuples()
        }
    mrr_ranges = {
        str(dataset): {
            str(scenario): float(group["mean"].max() - group["mean"].min())
            for scenario, group in dataset_frame.groupby("scenario")
        }
        for dataset, dataset_frame in mrr_per_model.groupby("dataset")
    }
    variance_finite = variance.dropna(subset=["between_share"])
    variance_extremes = {}
    if not variance_finite.empty:
        max_row = variance_finite.loc[variance_finite["between_share"].idxmax()]
        min_row = variance_finite.loc[variance_finite["between_share"].idxmin()]
        variance_extremes = {
            "max_between_share": {
                "dataset": str(max_row["dataset"]),
                "scenario": str(max_row["scenario"]),
                "metric": str(max_row["metric"]),
                "between_share": float(max_row["between_share"]),
            },
            "min_between_share": {
                "dataset": str(min_row["dataset"]),
                "scenario": str(min_row["scenario"]),
                "metric": str(min_row["metric"]),
                "between_share": float(min_row["between_share"]),
            },
        }
    dataset_correlation = rank_correlation.loc[rank_correlation["scope"] == "dataset"]
    absolute_precision_findings = {
        f"{row.scope}|{row.dataset}|{row.scenario}|{row.metric}": {
            "mean": float(row.mean),
            "sd": float(row.sd),
            "n_clusters": int(row.n_clusters),
            "n_model_seeds": int(row.n_model_seeds),
            "ci95_percentile_low": float(row.ci95_percentile_low),
            "ci95_percentile_high": float(row.ci95_percentile_high),
            "ci95_bca_low": float(row.ci95_bca_low),
            "ci95_bca_high": float(row.ci95_bca_high),
            "bootstrap_method": str(row.bootstrap_method),
        }
        for row in absolute_precision.itertuples()
    }
    correlation_findings = {
        str(row.dataset): {
            "mean_spearman": float(row.mean_spearman),
            "min_spearman": float(row.min_spearman),
            "mean_kendall": float(row.mean_kendall),
            "min_kendall": float(row.min_kendall),
            "frac_spearman_ge_0_8": float(row.frac_spearman_ge_0_8),
            "frac_kendall_ge_0_6": float(row.frac_kendall_ge_0_6),
        }
        for row in dataset_correlation.itertuples()
    }
    config_findings = {
        f"{row.config_label}|missing_indicators={'included' if row.include_missing_indicators else 'excluded'}": {
            "include_missing_indicators": bool(row.include_missing_indicators),
            "max_depth": int(row.max_depth),
            "min_child_weight": int(row.min_child_weight),
            "kpi_auc": float(row.kpi_auc),
            "train_auc": float(row.train_auc),
            "train_oof_auc_gap": float(row.train_oof_auc_gap),
            "mrr": float(row.mrr),
            "hr_at_1": float(row.hr_at_1),
            "hr_at_5": float(row.hr_at_5),
            "far_at_1": float(row.far_at_1),
            "far_at_5": float(row.far_at_5),
        }
        for row in config_overall.itertuples()
    }
    missing_findings = {
        f"{row.scenario}|{row.metric}": {
            "mean_diff_included_minus_excluded": float(row.mean_diff),
            "ci95_bca_low": float(row.ci95_bca_low),
            "ci95_bca_high": float(row.ci95_bca_high),
            "favors": str(row.favors),
        }
        for row in missing_comparison.itertuples()
    }
    wasserstein_findings = {
        f"{row.scenario}|{row.metric}": {
            "raw_mean": float(row.raw_mean),
            "standardized_mean": float(row.standardized_mean),
            "mean_diff_standardized_minus_raw": float(row.mean_diff_standardized_minus_raw),
            "ci95_bca_low": float(row.ci95_bca_low),
            "ci95_bca_high": float(row.ci95_bca_high),
            "favors": str(row.favors),
        }
        for row in wasserstein_paired.itertuples()
    }
    effect_summary = _select_effect_summary(effect_sizes)
    effect_findings = {
        f"{row.dataset}|{row.metric}": {
            "n_comparisons": int(row.n_comparisons),
            "ci_excludes_zero": int(row.ci_excludes_zero),
            "gads_favored": int(row.gads_favored),
            "baseline_favored": int(row.baseline_favored),
            "inconclusive": int(row.inconclusive),
        }
        for row in effect_summary.itertuples()
    }
    gads_root = root_summary.loc[root_summary["method"] == "GADS"]
    stability_findings = {
        "gads_root_cause_rank": _records(gads_root),
        "gads_top1_model_seed": _records(top1_model_summary.loc[top1_model_summary["dataset"].notna()]),
        "gads_top1_scenario_seed": _records(
            top1_scenario_summary.loc[top1_scenario_summary["method"] == "GADS"]
            if "method" in top1_scenario_summary.columns
            else top1_scenario_summary
        ),
        "gads_top5_jaccard": _records(
            jaccard_summary.loc[jaccard_summary["method"] == "GADS"]
            if "method" in jaccard_summary.columns
            else jaccard_summary
        ),
        "gads_top5_rank_composition": _records(rank_composition),
    }
    return {
        "model_seed_sweep": {
            "intervals": (
                "percentile and scipy BCa; the scenario-cluster scope averages the five model "
                "seeds within each scenario_seed before resampling"
            ),
            "mrr_mean_by_dataset_scenario_model_seed": mrr_nested,
            "mrr_model_seed_range_by_dataset_scenario": mrr_ranges,
            "absolute_metric_ci": absolute_precision_findings,
            "variance_decomposition_extremes": variance_extremes,
        },
        "rank_correlation_model_seeds": correlation_findings,
        "dataset_b_hyperparameters": {
            "configs": config_findings,
            "missing_indicators": missing_findings,
        },
        "wasserstein_x_scale": wasserstein_findings,
        "effect_sizes": effect_findings,
        "stability": stability_findings,
    }


def _markdown_protocol(
    *,
    protocol: dict[str, Any],
    counts: Counter[str],
    runtime_seconds: float,
    self_check: dict[str, Any],
) -> str:
    lines = [
        "# Pack 1 — Model/CV seeds, Dataset B hyperparameters, and stability statistics",
        "",
        "## Protocol",
        "",
        f"- Generated (UTC): `{protocol['created_at_utc']}`",
        f"- Host: `{protocol['host']['node']}` ({protocol['host']['platform']}), {protocol['host']['cpu_count']} CPUs",
        f"- Python {protocol['host']['python']}; xgboost {protocol['versions']['xgboost']}; "
        f"shap {protocol['versions']['shap']}; scikit-learn {protocol['versions']['scikit_learn']}",
        f"- Git commit: `{protocol['git_commit']}`",
        f"- Workers: {protocol['workers']}; each worker uses one BLAS thread and XGBoost `n_jobs=1`",
        f"- Total wall time: {runtime_seconds / 60.0:.1f} min",
        "",
        "**Dataset A**: `n_samples=5000`, `noise_level=0.2`, `n_devices=10`, "
        "`distribution_shift=3.0`, `mechanism_scale=1.0`, `harmless_drift=3.5`; "
        "XGBoost `max_depth=6`, `min_child_weight=1`.",
        "",
        "**Dataset B**: official UCI SECOM raw files; `group_seed=2024`, "
        "`physical_shift=3.0`, `mechanism_weight=12.0`, `harmless_shift=4.0`, "
        "missing-indicator columns **excluded**; XGBoost `max_depth=3`, `min_child_weight=5`.",
        "",
        "**Ranking protocol (both datasets)**: `eqp_mode=contextual_process`; GADS "
        "one-dimensional Wasserstein distance, 50 bins, epsilon 1e-10; 5-fold "
        "cross-fitting whose CV seed equals the model seed; metrics MRR, HR@1, HR@5, "
        "FAR@1, FAR@5; FAR is defined only in S4-main/S4-null and MRR/HR only outside S4-null.",
        "",
        "**Task counts**: " + ", ".join(f"{job}={count}" for job, count in sorted(counts.items())) + ".",
        "",
        f"**Protocol self-check**: {'PASSED' if self_check['passed'] else 'FAILED'} — the lean GADS "
        "path was compared against `run_experiment_bundle` on Dataset A and Dataset B, S1 seed 0.",
        "",
    ]
    lines.extend(
        f"- {check['dataset']}: max metric diff {check['max_metric_abs_diff']:.2e}, "
        f"max score diff {check['max_score_abs_diff']:.2e}, max rank diff {check['max_rank_abs_diff']:.0f}"
        for check in self_check["checks"]
    )
    lines.append("")
    return "\n".join(lines)


def _absolute_precision_table(absolute_precision: pd.DataFrame, scope: str) -> pd.DataFrame:
    subset = absolute_precision.loc[
        (absolute_precision["scope"] == scope)
        & (absolute_precision["metric"].isin(["mrr", "hr_at_1", "far_at_1", "kpi_auc"]))
    ]
    return subset.loc[
        :,
        [
            "dataset",
            "scenario",
            "metric",
            "mean",
            "ci95_percentile_low",
            "ci95_percentile_high",
            "ci95_bca_low",
            "ci95_bca_high",
        ],
    ]


def _markdown_model_seed(
    *,
    per_model: pd.DataFrame,
    variance: pd.DataFrame,
    rank_correlation: pd.DataFrame,
    absolute_precision: pd.DataFrame,
    model_seeds: tuple[int, ...],
) -> str:
    lines = [
        "## Part 1 — Model/CV seed sweep (R1 #3, R2 #3, R3 #12)",
        "",
        "GADS was rerun for every scenario seed 0–19 and every model/CV seed "
        f"{list(model_seeds)} on both datasets (1000 runs). The CV fold assignment changes with "
        "the model seed because `cv_seed` is not fixed independently.",
        "",
    ]
    for dataset in ("Dataset A", "Dataset B"):
        for metric, title in (("mrr", "MRR"), ("hr_at_1", "HR@1")):
            table = _pivot_metric(per_model, dataset=dataset, metric=metric, model_seeds=model_seeds)
            if table.empty:
                continue
            lines.extend(
                [
                    f"### {dataset} — mean {title} over 20 scenario seeds",
                    "",
                    _md_table(table, list(table.columns)),
                    "",
                ]
            )
    far_table = _pivot_far(per_model, model_seeds=model_seeds)
    if not far_table.empty:
        lines.extend(
            [
                "### Mean FAR@1 over 20 scenario seeds (S4-main / S4-null)",
                "",
                _md_table(far_table, list(far_table.columns)),
                "",
            ]
        )
    lines.extend(
        [
            "### Absolute precision: 95% CIs for mean metric levels",
            "",
            "Two resampling units are reported, because the 5-model-seed × 20-scenario-seed "
            "table is not 100 independent observations: the five model seeds share the same "
            "scenario realisation (data, mechanism weights and abnormal devices). "
            "The valid unit is therefore the `scenario_seed` cluster.",
            "",
            "**Model seed 42, 20 scenario seeds** — independent seed-level bootstrap over the "
            "20 scenario seeds (same slice as the main matrix).",
            "",
            _md_table(
                _absolute_precision_table(absolute_precision, "model_seed_42"),
                [
                    "dataset",
                    "scenario",
                    "metric",
                    "mean",
                    "ci95_percentile_low",
                    "ci95_percentile_high",
                    "ci95_bca_low",
                    "ci95_bca_high",
                ],
            ),
            "",
            "**Five model seeds, scenario-seed cluster bootstrap** — the five model seeds are "
            "averaged within each of the 20 `scenario_seed` clusters first, then the interval "
            "bootstraps over the 20 cluster means. The dispersion reflects scenario-to-scenario "
            "variation only; model-seed variation is averaged out inside each cluster.",
            "",
            _md_table(
                _absolute_precision_table(absolute_precision, "scenario_cluster_5_model_seeds"),
                [
                    "dataset",
                    "scenario",
                    "metric",
                    "mean",
                    "ci95_percentile_low",
                    "ci95_percentile_high",
                    "ci95_bca_low",
                    "ci95_bca_high",
                ],
            ),
            "",
            "All intervals use 10,000 resamples. BCa uses `scipy.stats.bootstrap`, whose bias "
            "correction applies the mid-rank tie correction "
            "`(count(< theta_hat) + 0.5 * count(== theta_hat)) / B`; this matters for discrete "
            "metrics such as HR@1 and FAR@1. Cells with constant input or a degenerate jackknife "
            "acceleration fall back to the percentile interval (marked in the CSV).",
            "",
            "### Variance decomposition",
            "",
            "Within-scenario-seed variance is the mean over the 20 scenario seeds of the "
            "variance across the five model seeds. Between-scenario-seed variance is the "
            "variance of the 20 scenario-seed means. `between_share` is "
            "between/(between+within).",
            "",
            _md_table(
                variance,
                [
                    "dataset",
                    "scenario",
                    "metric",
                    "grand_mean",
                    "within_sd",
                    "between_sd",
                    "between_share",
                    "model_seed_sd",
                    "model_seed_range",
                    "max_abs_deviation_from_seed42",
                ],
            ),
            "",
            "### Paired rank correlation of GADS feature scores across model seeds",
            "",
            "For each scenario seed and abnormal device, Spearman rho and Kendall tau were "
            "computed between the full GADS score vectors of every pair of model seeds "
            "(10 pairs per slice).",
            "",
            _md_table(
                rank_correlation,
                [
                    "scope",
                    "dataset",
                    "scenario",
                    "mean_spearman",
                    "median_spearman",
                    "min_spearman",
                    "frac_spearman_ge_0_8",
                    "mean_kendall",
                    "min_kendall",
                    "frac_kendall_ge_0_6",
                    "n_pairs",
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_hyperparameters(
    *,
    hyper_summary: pd.DataFrame,
    config_overall: pd.DataFrame,
    missing_comparison: pd.DataFrame,
    wasserstein_by_variant: pd.DataFrame,
    wasserstein_paired: pd.DataFrame,
) -> str:
    lines = [
        "## Part 2 — Dataset B hyperparameter sensitivity and protocol justification (R2 #3, R3 #14, R3 #9)",
        "",
        "The grid varies `max_depth` and `min_child_weight` around the paper choice "
        "`(3, 5)`. `(6, 1)` is Dataset A's unregularized default. Dataset B has "
        "n=1567 rows and p=590 process features, so `max_depth=6, mcw=1` can overfit.",
        "",
        "### Config × scenario means",
        "",
    ]
    pivot_source = hyper_summary.loc[~hyper_summary["include_missing_indicators"]]
    table_rows: list[pd.DataFrame] = []
    for metric in ("kpi_auc", "train_auc", "train_oof_auc_gap", "mrr", "hr_at_1", "far_at_1"):
        subset = pivot_source.loc[pivot_source["metric"] == metric]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=["config_label", "scenario"], columns="metric", values="mean", aggfunc="first")
        if pivot.empty:
            continue
        table_rows.append(pivot)
    if table_rows:
        combined = pd.concat(table_rows, axis=1).reset_index()
        combined["config_order"] = combined["config_label"].map(CONFIG_ORDER)
        combined = combined.sort_values(["config_order", "scenario"], kind="stable").drop(columns="config_order")
        lines.extend([_md_table(combined, list(combined.columns)), ""])
    lines.extend(
        [
            "### Overall by configuration",
            "",
            "MRR/HR averages use S1–S4-main (S4-null has no root cause); FAR averages use "
            "S4-main and S4-null. AUC averages use all 100 runs.",
            "",
            _md_table(
                config_overall,
                [
                    "config_label",
                    "include_missing_indicators",
                    "max_depth",
                    "min_child_weight",
                    "n_runs",
                    "kpi_auc",
                    "train_auc",
                    "train_oof_auc_gap",
                    "mrr",
                    "hr_at_1",
                    "hr_at_5",
                    "far_at_1",
                    "far_at_5",
                ],
            ),
            "",
            "### Dataset B — missing indicators included vs excluded (`(3, 5)`)",
            "",
            "`mean_diff` is included minus excluded. Positive is not automatically better: "
            "the `favors` column accounts for metric direction.",
            "",
            _md_table(
                missing_comparison.loc[
                    missing_comparison["metric"].isin(
                        [
                            "kpi_auc",
                            "train_auc",
                            "train_oof_auc_gap",
                            "mrr",
                            "hr_at_1",
                            "far_at_1",
                        ]
                    )
                ],
                [
                    "scenario",
                    "metric",
                    "included_mean",
                    "excluded_mean",
                    "mean_diff",
                    "ci95_bca_low",
                    "ci95_bca_high",
                    "n_pairs",
                    "favors",
                ],
            ),
            "",
            "### Dataset A, S2 — raw vs normal-pool-standardized Wasserstein-X",
            "",
            "Standardization z-scores each process feature with the mean and standard "
            "deviation of the normal-device reference pool before the Wasserstein distance. "
            "This removes the artificial scale advantage of `feat_special_imp` (generator "
            "std=2 versus 1 for the other features) and tests whether Wasserstein-X's S2 "
            "performance reflects a physical distribution shift.",
            "",
            _md_table(
                wasserstein_by_variant,
                ["scenario", "variant", "metric", "mean", "std", "minimum", "maximum", "n"],
            ),
            "",
            _md_table(
                wasserstein_paired,
                [
                    "scenario",
                    "metric",
                    "raw_mean",
                    "standardized_mean",
                    "mean_diff_standardized_minus_raw",
                    "ci95_bca_low",
                    "ci95_bca_high",
                    "favors",
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_effect_sizes(*, effect_sizes: pd.DataFrame) -> str:
    summary = _select_effect_summary(effect_sizes)
    mrr = effect_sizes.loc[effect_sizes["metric"] == "mrr"]
    far = effect_sizes.loc[
        (effect_sizes["metric"] == "far_at_1") & (effect_sizes["scenario"].isin(["S4-main", "S4-null"]))
    ]
    lines = [
        "## Part 3 — Effect sizes with bootstrap 95% CIs (R3 #12)",
        "",
        "Each comparison is paired by scenario seed (20 pairs). Intervals are 10,000-resample "
        "percentile and BCa bootstrap intervals for the mean paired difference GADS − baseline. "
        "BCa uses `scipy.stats.bootstrap`, whose bias correction uses the mid-rank tie correction "
        "`(count(< theta_hat) + 0.5 * count(== theta_hat)) / B`, so discrete metrics such as HR@1 "
        "and FAR@1 are not distorted by ties and `CI(-x)` is the exact negation of `CI(x)`. "
        "`p_value_holm` is the Holm-adjusted two-sided paired sign-flip permutation p-value "
        "within each dataset × scenario × metric group (five baselines).",
        "",
        "### How often the BCa interval excludes zero",
        "",
        _md_table(summary, list(summary.columns)),
        "",
        "### MRR: mean paired difference (GADS − baseline)",
        "",
        _md_table(
            mrr,
            [
                "dataset",
                "scenario",
                "baseline",
                "mean_diff",
                "median_diff",
                "cohen_dz",
                "ci95_bca_low",
                "ci95_bca_high",
                "p_value_holm",
                "favors",
            ],
        ),
        "",
        "### FAR@1 in S4-main / S4-null (lower is better)",
        "",
        _md_table(
            far,
            [
                "dataset",
                "scenario",
                "baseline",
                "mean_diff",
                "ci95_bca_low",
                "ci95_bca_high",
                "p_value_holm",
                "favors",
            ],
        ),
        "",
        "Full MRR/HR@1/HR@5/FAR@1/FAR@5 effect sizes are in `effect_sizes_stability/effect_sizes_bootstrap_ci.csv`.",
        "",
    ]
    return "\n".join(lines)


def _markdown_stability(
    *,
    root_summary: pd.DataFrame,
    top1_model_summary: pd.DataFrame,
    top1_scenario_summary: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    rank_composition: pd.DataFrame,
) -> str:
    gads_root = root_summary.loc[root_summary["method"] == "GADS"]
    gads_top1_model = top1_model_summary.loc[top1_model_summary["dataset"].notna()]
    gads_top1_scenario = (
        top1_scenario_summary.loc[top1_scenario_summary["method"] == "GADS"]
        if "method" in top1_scenario_summary.columns
        else top1_scenario_summary
    )
    gads_jaccard = jaccard_summary.loc[jaccard_summary["method"] == "GADS"]
    lines = [
        "## Part 3 (continued) — Root-cause rank distribution and Top-5 Jaccard context (R1 #4, R3 #13)",
        "",
        "The Top-5 Jaccard is low because ranks 2–5 churn while the annotated root cause "
        "stays at rank 1. The tables below separate those two facts.",
        "",
        "### GADS root-cause rank distribution (20 scenario seeds, model seed 42)",
        "",
        _md_table(
            gads_root,
            [
                "dataset",
                "scenario",
                "method",
                "n_obs",
                "median_rank",
                "q1_rank",
                "q3_rank",
                "iqr_rank",
                "top1_rate",
                "top3_rate",
                "top5_rate",
            ],
        ),
        "",
        "### GADS Top-1 stability across the five model seeds",
        "",
        "`mean_modal_share` is the average share of the five model seeds that select the same "
        "Top-1 feature; `fraction_all_same` is the share of slices where all five agree. "
        "`mean_top1_root_rate` is the share of model seeds whose Top-1 is the annotated root cause.",
        "",
        _md_table(
            gads_top1_model,
            [
                "dataset",
                "scenario",
                "n_groups",
                "mean_modal_share",
                "min_modal_share",
                "mean_n_distinct_top1",
                "mean_top1_root_rate",
                "fraction_modal_is_root",
                "fraction_all_same",
            ],
        ),
        "",
        "### GADS Top-1 stability across the 20 scenario seeds",
        "",
        _md_table(
            gads_top1_scenario,
            [
                "dataset",
                "scenario",
                "n_groups",
                "mean_modal_share",
                "min_modal_share",
                "mean_n_distinct_top1",
                "mean_top1_root_rate",
                "fraction_modal_is_root",
                "fraction_all_same",
            ],
        ),
        "",
        "### GADS mean pairwise Top-5 Jaccard (scenario seeds and model seeds)",
        "",
        _md_table(
            gads_jaccard,
            [
                "seed_kind",
                "dataset",
                "scenario",
                "mean_jaccard",
                "median_jaccard",
                "min_jaccard",
                "q25_jaccard",
                "q75_jaccard",
                "max_jaccard",
                "n_pairs",
            ],
        ),
        "",
        "### GADS feature-family composition of ranks 2–5 on Dataset A (S1–S4-main)",
        "",
        "The low Top-5 Jaccard is concentrated in ranks 2–5, which are dominated by the ten "
        "KPI-linked `feat_useful_*` features whose logit weights are redrawn for every scenario "
        "seed. `feat_harmless` are the ten `feat_useless_*` features correlated with the KPI; "
        "`feat_noise` are the remaining pure-noise `feat_useless_*` features. `main_rankings` uses "
        "the 20 scenario seeds at model seed 42; `model_seed_sweep` pools 20 scenario seeds × 5 "
        "model seeds. `ALL` pools S1–S4-main.",
        "",
        _md_table(
            rank_composition,
            [
                "source",
                "scenario",
                "n_positions",
                "frac_feat_useful",
                "frac_feat_harmless",
                "frac_feat_noise",
                "frac_root_cause",
            ],
        ),
        "",
        "The same Jaccard table for every baseline and the model-seed Jaccard are in "
        "`effect_sizes_stability/top5_jaccard_summary.csv`; the composition table is in "
        "`effect_sizes_stability/top5_rank_composition.csv` and "
        "`model_seed_sweep/top1_stability_model_seeds.csv`.",
        "",
    ]
    return "\n".join(lines)


def _write_summary_markdown(
    path: Path,
    *,
    protocol: dict[str, Any],
    counts: Counter[str],
    runtime_seconds: float,
    self_check: dict[str, Any],
    per_model: pd.DataFrame,
    variance: pd.DataFrame,
    rank_correlation: pd.DataFrame,
    hyper_summary: pd.DataFrame,
    config_overall: pd.DataFrame,
    missing_comparison: pd.DataFrame,
    wasserstein_by_variant: pd.DataFrame,
    wasserstein_paired: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    absolute_precision: pd.DataFrame,
    root_summary: pd.DataFrame,
    top1_model_summary: pd.DataFrame,
    top1_scenario_summary: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    rank_composition: pd.DataFrame,
    model_seeds: tuple[int, ...],
) -> Path:
    sections = [
        _markdown_protocol(
            protocol=protocol,
            counts=counts,
            runtime_seconds=runtime_seconds,
            self_check=self_check,
        ),
        _markdown_model_seed(
            per_model=per_model,
            variance=variance,
            rank_correlation=rank_correlation,
            absolute_precision=absolute_precision,
            model_seeds=model_seeds,
        ),
        _markdown_hyperparameters(
            hyper_summary=hyper_summary,
            config_overall=config_overall,
            missing_comparison=missing_comparison,
            wasserstein_by_variant=wasserstein_by_variant,
            wasserstein_paired=wasserstein_paired,
        ),
        _markdown_effect_sizes(effect_sizes=effect_sizes),
        _markdown_stability(
            root_summary=root_summary,
            top1_model_summary=top1_model_summary,
            top1_scenario_summary=top1_scenario_summary,
            jaccard_summary=jaccard_summary,
            rank_composition=rank_composition,
        ),
        "## Reviewer-item map\n\n"
        "| Item | Evidence |\n| --- | --- |\n"
        "| R1 #3, R2 #3, R3 #12 (model/CV seed fixed) | `model_seed_sweep/`; 1,000 runs over "
        "20 scenario seeds × 5 model seeds × 5 scenarios × 2 datasets; "
        "`model_seed_sweep/absolute_precision_ci.csv` holds seed-42 and scenario-cluster CIs |\n"
        "| R1 #4, R3 #13 (weak Top-5 Jaccard) | `model_seed_sweep/rank_correlation_model_seeds*`, "
        "`effect_sizes_stability/root_cause_rank_distribution*`, "
        "`effect_sizes_stability/top1_stability_*`, `effect_sizes_stability/top5_jaccard_summary.csv`, "
        "`effect_sizes_stability/top5_rank_composition.csv` |\n"
        "| R3 #12 (effect sizes with CIs) | `effect_sizes_stability/effect_sizes_bootstrap_ci.csv` "
        "(10,000-resample percentile and scipy BCa CIs with mid-rank tie correction) |\n"
        "| R3 #9 (Wasserstein-X S2 MRR = 0.446) | `hyperparameter_grid/wasserstein_x_scale_*` |\n"
        "| R2 #3, R3 #14 (Dataset B hyperparameters, missing indicators) | "
        "`hyperparameter_grid/metrics_long.csv`, `metrics_summary.csv`, `config_overall.csv`, "
        "`missing_indicators_comparison.csv` |\n",
        "## Output index\n\n"
        "- `summary.json` — machine-readable key findings.\n"
        "- `protocol.json` — exact settings, versions, self-check and task counts.\n"
        "- `model_seed_sweep/` — model/CV seed sweep tables and full rankings.\n"
        "- `hyperparameter_grid/` — Dataset B grid, missing-indicator check and Wasserstein-X scale check.\n"
        "- `effect_sizes_stability/` — effect sizes, root-cause ranks, Top-1 stability and Jaccard tables.\n",
    ]
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--scenario-seeds", nargs="+", type=int, default=list(DEFAULT_SCENARIO_SEEDS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(DEFAULT_MODEL_SEEDS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete existing per-task checkpoints before running (default: resume completed tasks)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild tables, JSON and summary.md from existing checkpoints without running any task",
    )
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    if not 1 <= args.workers <= 6:
        raise ValueError("--workers must lie in [1, 6]")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    if args.permutation_samples < 100:
        raise ValueError("--permutation-samples must be at least 100")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.summary_only and args.fresh:
        raise ValueError("--summary-only and --fresh are mutually exclusive")
    datasets = tuple(dict.fromkeys(args.datasets))
    scenarios = tuple(dict.fromkeys(args.scenarios))
    scenario_seeds = tuple(dict.fromkeys(int(seed) for seed in args.scenario_seeds))
    model_seeds = tuple(dict.fromkeys(int(seed) for seed in args.model_seeds))
    if any(seed < 0 for seed in scenario_seeds):
        raise ValueError("scenario seeds must be non-negative")
    if not scenarios or not scenario_seeds or not model_seeds:
        raise ValueError("at least one scenario, scenario seed and model seed are required")
    if "b" in datasets:
        verify_secom_files(RAW_DIR)
    return datasets, scenarios, scenario_seeds, model_seeds


def _run_pool(tasks: list[RunTask], workers: int, checkpoint_dir: Path) -> list[RunResult]:
    context = mp.get_context("spawn")
    results: list[RunResult] = []
    start = time.monotonic()
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(checkpoint_dir),),
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), start=1):
            results.append(result)
            if index == 1 or index % max(1, len(tasks) // 20) == 0 or index == len(tasks):
                elapsed = time.monotonic() - start
                rate = index / elapsed if elapsed > 0.0 else 0.0
                remaining = (len(tasks) - index) / rate if rate > 0.0 else 0.0
                print(
                    f"[{index}/{len(tasks)}] elapsed={elapsed / 60.0:.1f} min eta={remaining / 60.0:.1f} min",
                    flush=True,
                )
    errors = [result for result in results if result.error is not None]
    if errors:
        details = "\n".join(f"{result.task}: {result.error}" for result in errors[:10])
        raise RuntimeError(f"{len(errors)} of {len(tasks)} tasks failed; first failures:\n{details}")
    counts = Counter(result.task.job for result in results)
    expected = Counter(task.job for task in tasks)
    if counts != expected:
        raise RuntimeError(f"task counts changed during execution: expected {expected}, got {counts}")
    return results


def _load_self_check(output_dir: Path) -> dict[str, Any]:
    """Reuse the protocol self-check recorded by the original full run."""
    protocol_path = output_dir / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"--summary-only requires an existing protocol.json: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    self_check = protocol.get("self_check")
    if not isinstance(self_check, dict) or "checks" not in self_check:
        raise ValueError(f"existing protocol.json has no self_check block: {protocol_path}")
    return self_check


def main() -> None:
    """Run Pack 1 and write all tables, JSON and markdown summaries."""
    args = _parse_args()
    datasets, scenarios, scenario_seeds, model_seeds = _validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    if args.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(datasets, scenarios, scenario_seeds, model_seeds)
    print(f"Prepared {len(tasks)} tasks for datasets={datasets}, scenarios={scenarios}", flush=True)
    print(f"Scenario seeds: {list(scenario_seeds)}; model seeds: {list(model_seeds)}", flush=True)

    self_check = _load_self_check(output_dir) if args.summary_only else _verify_protocol(datasets)
    print(f"Protocol self-check: {self_check['checks']}", flush=True)

    completed, remaining = _load_checkpoint_results(checkpoint_dir, tasks)
    if args.summary_only and remaining:
        raise RuntimeError(f"--summary-only needs every checkpoint: {len(remaining)} of {len(tasks)} tasks are missing")
    if completed:
        print(f"Resuming: {len(completed)} checkpointed tasks found, {len(remaining)} remaining", flush=True)
    started = time.monotonic()
    if remaining:
        _run_pool(remaining, args.workers, checkpoint_dir)
    runtime_seconds = time.monotonic() - started
    results, still_missing = _load_checkpoint_results(checkpoint_dir, tasks)
    if still_missing:
        raise RuntimeError(f"{len(still_missing)} tasks have no checkpoint after execution")
    counts = Counter(result.task.job for result in results)
    print(f"All tasks completed in {runtime_seconds / 60.0:.1f} min: {dict(counts)}", flush=True)

    model_metrics, model_rankings = _collect_frames(results, "model_seed")
    model_metrics = _sort_frame(model_metrics, ["dataset", "scenario", "scenario_seed", "model_seed"])
    model_rankings = _sort_frame(
        model_rankings,
        ["dataset", "scenario", "scenario_seed", "model_seed", "equipment", "rank"],
    )
    per_model = _per_model_seed_summary(model_metrics)
    variance = _variance_decomposition(model_metrics)
    absolute_precision = _absolute_precision_ci(
        model_metrics,
        bootstrap_samples=args.bootstrap_samples,
        model_seeds=model_seeds,
    )
    rank_correlation_pairs, rank_correlation_summary = _rank_correlations(model_rankings, model_seeds)
    top1_model, top1_model_summary = _top1_stability(
        model_rankings,
        group_columns=["dataset", "scenario", "scenario_seed", "equipment"],
        seed_column="model_seed",
    )
    top1_model_overall = _aggregate_top1_summary(top1_model_summary, ["dataset", "scenario"])

    main_metrics, main_rankings = _collect_frames(results, "main")
    main_metrics = _sort_frame(
        main_metrics,
        ["dataset", "scenario", "scenario_seed", "method"],
    )
    main_rankings = _sort_frame(
        main_rankings,
        ["dataset", "scenario", "scenario_seed", "method", "equipment", "rank"],
    )
    if 42 in model_seeds:
        main_consistency = _validate_main_against_model_sweep(main_metrics, model_metrics)
        print(f"Main matrix matches model-seed sweep for GADS: {main_consistency}", flush=True)
    else:
        main_consistency = {
            "passed": False,
            "skipped": "model seed 42 is not in the requested model-seed sweep",
        }
        print(f"Main-matrix cross-check skipped: {main_consistency['skipped']}", flush=True)

    effect_sizes = _effect_sizes(
        main_metrics,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=42,
    )
    root_ranks, root_summary, root_overall = _root_cause_rank_tables(main_rankings)
    top1_scenario_raw, top1_scenario_summary = _top1_stability(
        main_rankings,
        group_columns=["dataset", "scenario", "method", "equipment"],
        seed_column="scenario_seed",
    )
    jaccard_pairs, jaccard_summary = _topk_jaccard(
        main_rankings,
        group_columns=["dataset", "scenario", "method", "equipment"],
        seed_column="scenario_seed",
        top_k=args.top_k,
        seed_kind="scenario_seed",
    )
    model_jaccard_pairs, model_jaccard_summary = _topk_jaccard(
        model_rankings,
        group_columns=["dataset", "scenario", "scenario_seed", "equipment"],
        seed_column="model_seed",
        top_k=args.top_k,
        seed_kind="model_seed",
        summary_group_columns=["dataset", "scenario"],
    )
    rank_composition = _top_rank_composition(main_rankings, model_rankings)
    jaccard_summary = pd.concat(
        [
            jaccard_summary.assign(seed_kind="scenario_seed"),
            model_jaccard_summary.assign(seed_kind="model_seed", method="GADS"),
        ],
        ignore_index=True,
    )
    jaccard_pairs = pd.concat(
        [
            jaccard_pairs.assign(seed_kind="scenario_seed"),
            model_jaccard_pairs.assign(seed_kind="model_seed", method="GADS"),
        ],
        ignore_index=True,
    )

    if "b" in datasets:
        hyper_metrics, _ = _collect_frames(results, "hyperparam", include_rankings=False)
        missing_metrics, _ = _collect_frames(results, "missing_indicator", include_rankings=False)
        hyper_all = pd.concat([hyper_metrics, missing_metrics], ignore_index=True)
        hyper_all = _sort_frame(
            hyper_all,
            ["config_label", "scenario", "scenario_seed"],
        )
        hyper_summary = _hyperparameter_metrics_summary(hyper_all)
        config_overall = _config_overall(hyper_all)
        missing_comparison = _missing_indicator_comparison(
            hyper_all,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
    else:
        hyper_all = pd.DataFrame()
        hyper_summary = pd.DataFrame()
        config_overall = pd.DataFrame()
        missing_comparison = pd.DataFrame()
    if "a" in datasets:
        wasserstein_metrics, _ = _collect_frames(results, "wasserstein_scale", include_rankings=False)
        wasserstein_metrics = _sort_frame(wasserstein_metrics, ["scenario", "scenario_seed", "variant"])
        wasserstein_by_variant, wasserstein_paired = _wasserstein_scale_tables(
            wasserstein_metrics,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
    else:
        wasserstein_metrics = pd.DataFrame()
        wasserstein_by_variant = pd.DataFrame()
        wasserstein_paired = pd.DataFrame()

    model_dir = output_dir / "model_seed_sweep"
    _write_frame(model_metrics, model_dir / "metrics_long.csv")
    _write_frame(per_model, model_dir / "metrics_by_model_seed.csv")
    _write_frame(variance, model_dir / "variance_decomposition.csv")
    _write_frame(absolute_precision, model_dir / "absolute_precision_ci.csv")
    _write_frame(rank_correlation_pairs, model_dir / "rank_correlation_model_seeds.csv")
    _write_frame(rank_correlation_summary, model_dir / "rank_correlation_model_seeds_summary.csv")
    _write_frame(top1_model, model_dir / "top1_stability_model_seeds.csv")
    _write_frame(top1_model_summary, model_dir / "top1_stability_model_seeds_summary.csv")
    _write_frame(top1_model_overall, model_dir / "top1_stability_model_seeds_overall.csv")
    _write_frame(model_rankings, model_dir / "rankings_long.csv")
    _write_parquet(model_rankings, model_dir / "rankings_long.parquet")

    hyper_dir = output_dir / "hyperparameter_grid"
    _write_frame(hyper_all, hyper_dir / "metrics_long.csv")
    _write_frame(hyper_summary, hyper_dir / "metrics_summary.csv")
    _write_frame(config_overall, hyper_dir / "config_overall.csv")
    _write_frame(missing_comparison, hyper_dir / "missing_indicators_comparison.csv")
    _write_frame(wasserstein_metrics, hyper_dir / "wasserstein_x_scale_per_seed.csv")
    _write_frame(wasserstein_by_variant, hyper_dir / "wasserstein_x_scale_by_variant.csv")
    _write_frame(wasserstein_paired, hyper_dir / "wasserstein_x_scale_paired_diff.csv")

    stability_dir = output_dir / "effect_sizes_stability"
    _write_frame(main_metrics, stability_dir / "main_metrics_long.csv")
    _write_frame(main_rankings, stability_dir / "main_rankings_long.csv")
    _write_parquet(main_rankings, stability_dir / "main_rankings_long.parquet")
    _write_frame(effect_sizes, stability_dir / "effect_sizes_bootstrap_ci.csv")
    _write_frame(root_ranks, stability_dir / "root_cause_ranks_per_run.csv")
    _write_frame(root_summary, stability_dir / "root_cause_rank_distribution.csv")
    _write_frame(root_overall, stability_dir / "root_cause_rank_distribution_overall.csv")
    _write_frame(top1_scenario_raw, stability_dir / "top1_stability_scenario_seeds.csv")
    _write_frame(top1_scenario_summary, stability_dir / "top1_stability_scenario_seeds_summary.csv")
    _write_frame(jaccard_pairs, stability_dir / "top5_jaccard_pairs.csv")
    _write_frame(jaccard_summary, stability_dir / "top5_jaccard_summary.csv")
    _write_frame(rank_composition, stability_dir / "top5_rank_composition.csv")

    findings = _build_findings(
        per_model,
        variance,
        rank_correlation_summary,
        config_overall,
        missing_comparison,
        wasserstein_paired,
        effect_sizes,
        absolute_precision,
        root_summary,
        top1_model_overall,
        top1_scenario_summary,
        jaccard_summary,
        rank_composition,
        model_seeds,
    )
    protocol = {
        "pack": "pack1_seeds_and_hyperparams",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "workers": args.workers,
        "fresh_start": bool(args.fresh),
        "summary_only_rebuild": bool(args.summary_only),
        "checkpointed_tasks_at_start": len(completed),
        "datasets": list(datasets),
        "scenarios": list(scenarios),
        "scenario_seeds": list(scenario_seeds),
        "model_seeds": list(model_seeds),
        "settings": {
            "dataset_a": {
                "n_samples": 5000,
                "noise_level": 0.2,
                "n_devices": 10,
                "distribution_shift": 3.0,
                "mechanism_scale": 1.0,
                "harmless_drift": 3.5,
                "model_params": DATASET_A_CONFIG.as_model_params(),
            },
            "dataset_b": {
                "group_seed": 2024,
                "physical_shift": 3.0,
                "mechanism_weight": 12.0,
                "harmless_shift": 4.0,
                "include_missing_indicators": False,
                "model_params": DATASET_B_CONFIG.as_model_params(),
            },
            "model": {"type": "xgboost", "cv_splits": 5, "cv_seed": "equals model_seed", "n_jobs": 1},
            "ranking": {
                "eqp_mode": "contextual_process",
                "gads_distance": "wasserstein",
                "gads_bins": DEFAULT_GADS_BINS,
                "gads_epsilon": DEFAULT_GADS_EPSILON,
                "top_ks": [1, 3, 5],
            },
            "effect_sizes": {
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": 0.95,
                "interval": (
                    "percentile and scipy BCa (mid-rank tie correction); percentile fallback for degenerate cells"
                ),
                "cluster_unit": "scenario_seed for the 5-model-seed absolute precision table",
                "permutation_samples": args.permutation_samples,
            },
        },
        "task_counts": dict(sorted(counts.items())),
        "runtime_seconds": runtime_seconds,
        "self_check": self_check,
        "main_matrix_cross_check": main_consistency,
    }
    summary = {
        "pack": "pack1_seeds_and_hyperparams",
        "created_at_utc": protocol["created_at_utc"],
        "git_commit": protocol["git_commit"],
        "protocol_file": str(output_dir / "protocol.json"),
        "key_findings": findings,
        "outputs": {
            "model_seed_sweep": str(model_dir),
            "hyperparameter_grid": str(hyper_dir),
            "effect_sizes_stability": str(stability_dir),
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(_sanitize_json(protocol), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_sanitize_json(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path = _write_summary_markdown(
        output_dir / "summary.md",
        protocol=protocol,
        counts=counts,
        runtime_seconds=runtime_seconds,
        self_check=self_check,
        per_model=per_model,
        variance=variance,
        rank_correlation=rank_correlation_summary,
        hyper_summary=hyper_summary,
        config_overall=config_overall,
        missing_comparison=missing_comparison,
        wasserstein_by_variant=wasserstein_by_variant,
        wasserstein_paired=wasserstein_paired,
        effect_sizes=effect_sizes,
        absolute_precision=absolute_precision,
        root_summary=root_summary,
        top1_model_summary=top1_model_overall,
        top1_scenario_summary=top1_scenario_summary,
        jaccard_summary=jaccard_summary,
        rank_composition=rank_composition,
        model_seeds=model_seeds,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(summary_path),
                "protocol_json": str(protocol_path),
                "summary_markdown": str(markdown_path),
                "task_counts": dict(sorted(counts.items())),
                "runtime_minutes": round(runtime_seconds / 60.0, 2),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

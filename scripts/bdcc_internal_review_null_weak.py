#!/usr/bin/env python3
"""Independent S0-clean null calibration and weak Dataset B intensity experiments.

This script is a *new, standalone* experiment for the internal review.  It does
not modify or overwrite the existing pack3 outputs.  All generation, model
fitting, GADS ranking, metric and permutation-null definitions are imported from
``scripts.bdcc_pack3_diagnostics_and_bsweep`` so that the two experiment packs
share exactly one implementation.

Two independent experiment blocks are written under
``results/bdcc_internal_review/null_weak/``:

1. ``s0_clean/`` -- S0-clean no-fault calibration for Dataset A and Dataset B,
   200 scenario seeds (0..199), 500 permutations per cell, and both the
   ``standard`` and within-EQP ``centered`` views.  The reported event is
   ``observed_max > q95_max`` where ``q95_max`` is the 95th percentile of the
   permutation maximum over all process features (the "process-max" cutoff).
   The script reports the event rate with a Wilson 95% interval and the
   across-seed distribution of the cutoff thresholds.

2. ``weak_b/`` -- Dataset B S2 mechanism local-weight sweep at
   ``0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5`` with 20 scenario seeds (0..19).
   ``0.2`` equals the normal-pool global coefficient and is the no-fault
   endpoint: its root manifest must be empty and it produces no MRR/HR rows.
   ``0.4``--``1.5`` are injected mechanism faults with a non-empty root
   manifest.  Only the standard view is fitted; TreeSHAP variants, tree-split
   diagnostics and interventional diagnostics are intentionally not rerun.

Statistical unit
----------------
Every reported mean/SD/BCa interval is computed over scenario seeds, not over
rows or devices.  The random-ranking MRR expectation ``H_590 / 590`` is stored
as a *theoretical reference only*; the script never converts a small distance
from that reference into an "indistinguishable" claim without a formal test.

Failure policy
--------------
Unknown datasets, unsupported weights, missing roots where roots are required,
missing partials and any worker exception are raised.  Nothing is silently
skipped or replaced by a fallback.

Environment note
----------------
Run inside the repository environment, for example::

    PYTHONPATH="$PWD/src" uv run --no-sync python scripts/bdcc_internal_review_null_weak.py \
        --raw-dir data/raw/secom --workers 12

On the remote host the SECOM raw directory is passed explicitly with
``--raw-dir``; it is opened read-only and never written to.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import hashlib
from itertools import product
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import bdcc_pack3_diagnostics_and_bsweep as pack3  # noqa: E402
from shap_diff_analysis.experiment_runner import git_commit  # noqa: E402
from shap_diff_analysis.experiment_types import DatasetBundle  # noqa: E402
from shap_diff_analysis.gads import (  # noqa: E402
    DEFAULT_GADS_BINS,
    DEFAULT_GADS_EPSILON,
    rank_gads_features,
)
from shap_diff_analysis.secom import verify_secom_files  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/bdcc_internal_review/null_weak"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw/secom"

S0_CLEAN_SEEDS: tuple[int, ...] = tuple(range(200))
WEAK_SEEDS: tuple[int, ...] = tuple(range(20))
NO_FAULT_LEVEL = 0.2
WEAK_LEVELS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
WEAK_METRIC_LEVELS: tuple[float, ...] = tuple(level for level in WEAK_LEVELS if level != NO_FAULT_LEVEL)
S0_VIEWS: tuple[str, ...] = ("standard", "centered")
DATASETS: tuple[str, ...] = ("a", "b")
WEAK_METRIC_NAMES: tuple[str, ...] = ("mrr", "hr_at_1", "hr_at_3", "hr_at_5")
S0_DISTRIBUTION_METRICS: tuple[str, ...] = ("observed_max", "q95_max", "q99_max", "p_value_max")
N_FEATURES_B = 590
RANDOM_MRR_EXPECTATION = float(sum(1.0 / rank for rank in range(1, N_FEATURES_B + 1)) / N_FEATURES_B)
BCA_TIE_CORRECTION = "midrank:(count(boot<estimate)+0.5*count(boot==estimate))/n_boot"
BCA_SINGULAR_POLICY = "raise: no clipping of bias-correction proportion or adjusted quantile"

S0_CELL_COLUMNS: tuple[str, ...] = (
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
    "n_roots",
    "n_roots_detected_fwer95",
    "n_roots_detected_tau95",
    "any_root_detected_fwer95",
    "all_roots_detected_fwer95",
    "min_gads_rank_of_roots",
    "top1_feature",
    "top1_is_root",
    "top1_is_harmless",
    "kpi_auc",
)


@dataclass(frozen=True)
class S0Task:
    """One S0-clean cell: dataset x scenario seed, both views."""

    dataset: str
    scenario_seed: int
    task_id: str


@dataclass(frozen=True)
class WeakMetricTask:
    """One weak-intensity Dataset B cell with an injected mechanism fault."""

    level: float
    scenario_seed: int
    task_id: str


@dataclass(frozen=True)
class WeakEndpointTask:
    """One no-fault endpoint construction check (no fit, no ranking metrics)."""

    level: float
    scenario_seed: int
    task_id: str


Task = S0Task | WeakMetricTask | WeakEndpointTask


@dataclass
class TaskResult:
    """Result or captured error for one task."""

    task_id: str
    kind: str
    status: str
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error: str | None = None


_TASK_SETTINGS: dict[str, Any] = {}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _base_settings(n_permutations: int) -> dict[str, Any]:
    return {
        "n_permutations": int(n_permutations),
        "model_seed": int(pack3.MODEL_SEED),
        "cv_splits": int(pack3.CV_SPLITS),
        "gads_bins": int(DEFAULT_GADS_BINS),
        "gads_epsilon": float(DEFAULT_GADS_EPSILON),
        "script_sha256": _code_sha256(),
    }


def _jsonable(value: Any) -> Any:
    return pack3._jsonable(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    working = frame.copy()
    for column in columns:
        if column not in working.columns:
            working[column] = pd.NA
    return working.loc[:, list(columns)]


def _parse_seed_range(text: str) -> tuple[int, ...]:
    try:
        start_text, end_text = text.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise ValueError(f"seed range must look like START:END, got {text!r}") from error
    if start < 0 or end <= start:
        raise ValueError(f"seed range must satisfy 0 <= START < END, got {text!r}")
    return tuple(range(start, end))


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def _s0_clean_bundle(dataset: str, scenario_seed: int) -> DatasetBundle:
    """Load the pack3 S0-clean bundle (empty root manifest, no harmless features)."""
    bundle = pack3._load_bundle(
        dataset,
        "S0-clean",
        int(scenario_seed),
        float(pack3.DATASET_B_PARAMS["physical_shift"]),
        float(pack3.DATASET_B_PARAMS["mechanism_weight"]),
        float(pack3.DATASET_B_PARAMS["harmless_shift"]),
    )
    if any(bundle.root_causes.values()):
        raise RuntimeError(
            f"S0-clean bundle for dataset {dataset!r} seed {scenario_seed} has a non-empty root manifest"
        )
    if bundle.harmless_features:
        raise RuntimeError(
            f"S0-clean bundle for dataset {dataset!r} seed {scenario_seed} has non-empty harmless features"
        )
    return bundle


def _weak_bundle(level: float, scenario_seed: int) -> DatasetBundle:
    """Load the Dataset B S2 bundle at one mechanism local weight."""
    if not math.isclose(level, NO_FAULT_LEVEL) and not any(
        math.isclose(level, candidate) for candidate in WEAK_METRIC_LEVELS
    ):
        raise ValueError(f"unsupported weak-intensity level {level!r}; expected one of {WEAK_LEVELS}")
    bundle = pack3._load_bundle(
        "b",
        "S2",
        int(scenario_seed),
        float(pack3.DATASET_B_PARAMS["physical_shift"]),
        float(level),
        float(pack3.DATASET_B_PARAMS["harmless_shift"]),
    )
    if not bundle.root_causes:
        raise RuntimeError(f"weak-intensity level {level} seed {scenario_seed} unexpectedly has an empty root manifest")
    return bundle


def _no_fault_endpoint_bundle(scenario_seed: int) -> tuple[DatasetBundle, dict[str, Any]]:
    """Build and validate the Dataset B weight-0.2 no-fault endpoint.

    The generator always annotates EQP_B as the mechanism device for S2, even
    when the local coefficient equals the normal-pool coefficient.  The endpoint
    is therefore the same generated data with an explicitly emptied root
    manifest; the mechanism effect must be exactly zero.
    """
    bundle = pack3._load_bundle(
        "b",
        "S2",
        int(scenario_seed),
        float(pack3.DATASET_B_PARAMS["physical_shift"]),
        float(NO_FAULT_LEVEL),
        float(pack3.DATASET_B_PARAMS["harmless_shift"]),
    )
    device = bundle.abnormal_devices[0]
    mechanism = bundle.metadata.get("glm", {}).get("mechanism", {})
    if not isinstance(mechanism, dict) or device not in mechanism:
        raise RuntimeError(f"no-fault endpoint seed {scenario_seed} is missing the mechanism record")
    record = mechanism[device]
    local_weight = float(record["local_weight"])
    global_weight = float(record["global_weight"])
    effect = local_weight - global_weight
    if not math.isclose(effect, 0.0, abs_tol=1e-12):
        raise RuntimeError(f"no-fault endpoint seed {scenario_seed} has non-zero mechanism effect {effect}")

    endpoint_bundle = replace(
        bundle,
        scenario="S0-clean",
        root_causes={},
        harmless_features=(),
        metadata={
            **bundle.metadata,
            "null_weak_endpoint": {
                "level": NO_FAULT_LEVEL,
                "no_fault_endpoint": True,
                "root_manifest_empty": True,
                "construction": "mechanism local weight equals the normal-pool global weight",
                "abnormal_device": device,
            },
        },
    )
    row = {
        "dataset": "Dataset B",
        "scenario": "S0-clean",
        "scenario_seed": int(scenario_seed),
        "level": NO_FAULT_LEVEL,
        "no_fault_endpoint": True,
        "root_manifest_empty": True,
        "n_root_causes": 0,
        "mechanism_device": device,
        "mechanism_local_weight": local_weight,
        "mechanism_global_weight": global_weight,
        "mechanism_effect": effect,
        "n_process_features": len(endpoint_bundle.process_features),
        "n_abnormal_rows": int((endpoint_bundle.data[endpoint_bundle.group_column].astype(str) == device).sum()),
        "n_normal_rows": int(
            endpoint_bundle.data[endpoint_bundle.group_column].astype(str).isin(endpoint_bundle.normal_devices).sum()
        ),
    }
    return endpoint_bundle, row


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------


def _validate_s0_bundle(bundle: DatasetBundle) -> None:
    if bundle.abnormal_devices != ("EQP_B",):
        raise RuntimeError(f"unexpected S0-clean abnormal devices: {bundle.abnormal_devices}")
    if bundle.root_causes.get("EQP_B", ()):
        raise RuntimeError("S0-clean root manifest must be empty")


def _run_s0_task(task: S0Task) -> TaskResult:
    bundle = _s0_clean_bundle(task.dataset, task.scenario_seed)
    _validate_s0_bundle(bundle)
    X, y, groups = pack3._feature_matrix(bundle)
    dataset_label = pack3.DATASET_LABELS[task.dataset]
    global_kpi = pack3._global_kpi_features(bundle, task.dataset)
    rows: list[dict[str, Any]] = []

    for view in S0_VIEWS:
        cross = pack3._cross_fit(X, y, dataset=task.dataset, center_by_group=view == "centered")
        ranking = rank_gads_features(
            cross.shap_values,
            groups,
            bundle.abnormal_devices,
            bundle.normal_devices,
            process_features=bundle.process_features,
            view="process",
            distance="wasserstein",
            bins=DEFAULT_GADS_BINS,
            epsilon=DEFAULT_GADS_EPSILON,
        )
        cells, _feature_rows = pack3._permutation_null(
            cross.shap_values,
            groups,
            bundle,
            dataset_label=dataset_label,
            scenario_seed=task.scenario_seed,
            view=view,
            wasserstein_ranking=ranking,
            n_permutations=int(_TASK_SETTINGS["n_permutations"]),
            global_kpi=global_kpi,
        )
        for cell in cells:
            cell["kpi_auc"] = float(cross.mean_auc)
        rows.extend(cells)

    return TaskResult(task_id=task.task_id, kind="s0_clean", status="ok", tables={"s0_cells": rows})


def _run_weak_metric_task(task: WeakMetricTask) -> TaskResult:
    bundle = _weak_bundle(task.level, task.scenario_seed)
    X, y, groups = pack3._feature_matrix(bundle)
    cross = pack3._cross_fit(X, y, dataset="b", center_by_group=False)
    rankings = pack3._rank_all_distances(cross.shap_values, groups, bundle)
    rows = pack3._metric_rows_from_rankings(
        rankings,
        bundle=bundle,
        dataset_label=pack3.DATASET_LABELS["b"],
        scenario_seed=task.scenario_seed,
        view=f"weak_w{task.level:g}",
        kpi_auc=cross.mean_auc,
    )
    for row in rows:
        row["sweep_name"] = "weak_b"
        row["level"] = float(task.level)
        row["reused_base"] = False
    return TaskResult(task_id=task.task_id, kind="weak_b", status="ok", tables={"weak_metrics": rows})


def _run_weak_endpoint_task(task: WeakEndpointTask) -> TaskResult:
    _bundle, row = _no_fault_endpoint_bundle(task.scenario_seed)
    return TaskResult(task_id=task.task_id, kind="weak_b_endpoint", status="ok", tables={"weak_endpoint": [row]})


def _run_task_worker(task: Task) -> TaskResult:
    try:
        if isinstance(task, S0Task):
            return _run_s0_task(task)
        if isinstance(task, WeakMetricTask):
            return _run_weak_metric_task(task)
        if isinstance(task, WeakEndpointTask):
            return _run_weak_endpoint_task(task)
        raise TypeError(f"unsupported task type: {type(task).__name__}")
    except Exception:
        return TaskResult(
            task_id=task.task_id,
            kind="s0_clean" if isinstance(task, S0Task) else "weak_b",
            status="error",
            error=traceback.format_exc(),
        )


def _init_worker(raw_dir: str, n_permutations: int) -> None:
    global _TASK_SETTINGS
    _TASK_SETTINGS = {
        **_base_settings(n_permutations),
        "raw_dir": str(raw_dir),
    }
    pack3.RAW_DIR = Path(raw_dir)
    pack3._load_bundle.cache_clear()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    threadpool_limits(limits=1)


# ---------------------------------------------------------------------------
# Task grid
# ---------------------------------------------------------------------------


def build_tasks(
    *,
    part: str,
    datasets: Sequence[str],
    s0_seeds: Sequence[int],
    weak_seeds: Sequence[int],
) -> tuple[list[S0Task], list[WeakMetricTask], list[WeakEndpointTask]]:
    """Expand the S0-clean and weak-intensity grids without duplicate cells."""
    if part not in {"both", "s0", "weak"}:
        raise ValueError(f"unsupported part {part!r}")
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; expected a subset of {DATASETS}")
    if len(set(s0_seeds)) != len(s0_seeds) or len(set(weak_seeds)) != len(weak_seeds):
        raise ValueError("scenario seeds must be unique")

    s0_tasks: list[S0Task] = []
    weak_metric_tasks: list[WeakMetricTask] = []
    endpoint_tasks: list[WeakEndpointTask] = []
    if part in {"both", "s0"}:
        s0_tasks.extend(
            S0Task(
                dataset=dataset,
                scenario_seed=int(seed),
                task_id=f"s0_clean_{dataset}_seed{seed}",
            )
            for dataset in datasets
            for seed in s0_seeds
        )
    if part in {"both", "weak"}:
        endpoint_tasks.extend(
            WeakEndpointTask(
                level=NO_FAULT_LEVEL,
                scenario_seed=int(seed),
                task_id=f"weak_b_endpoint_seed{seed}",
            )
            for seed in weak_seeds
        )
        weak_metric_tasks.extend(
            WeakMetricTask(
                level=float(level),
                scenario_seed=int(seed),
                task_id=f"weak_b_w{level:g}_seed{seed}",
            )
            for level in WEAK_METRIC_LEVELS
            for seed in weak_seeds
        )
    task_ids = [task.task_id for task in (*s0_tasks, *weak_metric_tasks, *endpoint_tasks)]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("task grid contains duplicate task ids")
    return s0_tasks, weak_metric_tasks, endpoint_tasks


# ---------------------------------------------------------------------------
# Partial persistence and pool
# ---------------------------------------------------------------------------


def _partial_path(partials_dir: Path, task_id: str) -> Path:
    return partials_dir / f"{task_id}.json"


def _partial_is_compatible(payload: dict[str, Any]) -> bool:
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return False
    expected = _TASK_SETTINGS
    keys = (
        "n_permutations",
        "model_seed",
        "cv_splits",
        "gads_bins",
        "gads_epsilon",
        "script_sha256",
    )
    return all(str(settings.get(key)) == str(expected.get(key)) for key in keys)


def _write_partial(partials_dir: Path, result: TaskResult) -> None:
    payload = {
        "task_id": result.task_id,
        "kind": result.kind,
        "status": result.status,
        "error": result.error,
        "settings": dict(_TASK_SETTINGS),
        "tables": {name: [_jsonable(row) for row in rows] for name, rows in result.tables.items()},
    }
    _write_json(_partial_path(partials_dir, result.task_id), payload)


def _load_partial(partials_dir: Path, task_id: str) -> dict[str, Any]:
    path = _partial_path(partials_dir, task_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing task partial: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != task_id:
        raise ValueError(f"partial {path} contains task_id {payload.get('task_id')!r}, expected {task_id!r}")
    return payload


def _load_all_partials(
    partials_dir: Path,
    tasks: Sequence[Task],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load every partial; return successes and failures without hiding either."""
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in tasks:
        path = _partial_path(partials_dir, task.task_id)
        if not path.is_file():
            missing.append(task.task_id)
            continue
        payload = _load_partial(partials_dir, task.task_id)
        if not _partial_is_compatible(payload):
            raise RuntimeError(f"partial {task.task_id} was produced with different settings; rerun with --force")
        if payload.get("status") != "ok":
            failures.append({"task_id": task.task_id, "error": str(payload.get("error"))})
            continue
        payloads.append(payload)
    if missing:
        raise FileNotFoundError(f"{len(missing)} task partials are missing: {missing[:10]}")
    return payloads, failures


def _load_partials_for_aggregation(
    partials_dir: Path,
    tasks: Sequence[Task],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load training partials without requiring the aggregation script hash.

    The partials record the script hash that produced the fitted numbers.  Pure
    re-aggregation intentionally uses a corrected summarizer, so only the
    training-affecting settings are re-checked here.  The returned hash is the
    single training-script hash shared by all partials.
    """
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[str] = []
    training_hashes: set[str] = set()
    training_keys = ("n_permutations", "model_seed", "cv_splits", "gads_bins", "gads_epsilon")
    for task in tasks:
        path = _partial_path(partials_dir, task.task_id)
        if not path.is_file():
            missing.append(task.task_id)
            continue
        payload = _load_partial(partials_dir, task.task_id)
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise RuntimeError(f"partial {task.task_id} has no settings block")
        for key in training_keys:
            if str(settings.get(key)) != str(_TASK_SETTINGS.get(key)):
                raise RuntimeError(
                    f"partial {task.task_id} training setting {key!r} is "
                    f"{settings.get(key)!r}, expected {_TASK_SETTINGS.get(key)!r}"
                )
        training_hashes.add(str(settings.get("script_sha256")))
        if payload.get("status") != "ok":
            failures.append({"task_id": task.task_id, "error": str(payload.get("error"))})
            continue
        payloads.append(payload)
    if missing:
        raise FileNotFoundError(f"{len(missing)} task partials are missing: {missing[:10]}")
    if len(training_hashes) != 1:
        raise RuntimeError(f"training partials disagree on script_sha256: {sorted(training_hashes)}")
    return payloads, failures, training_hashes.pop()


def _run_pool(
    tasks: Sequence[Task],
    *,
    workers: int,
    partials_dir: Path,
    status_path: Path,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    """Run pending tasks with spawn multiprocessing, persisting each result."""
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    context = mp.get_context("spawn")
    started = _utc_now()
    start = time.monotonic()
    failures = 0
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(raw_dir), int(_TASK_SETTINGS["n_permutations"])),
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(_run_task_worker, list(tasks), chunksize=1), start=1):
            _write_partial(partials_dir, result)
            if result.status != "ok":
                failures += 1
                print(f"[{index}/{len(tasks)}] FAILED {result.task_id}", flush=True)
            elapsed = time.monotonic() - start
            rate = index / elapsed if elapsed > 0.0 else 0.0
            remaining = (len(tasks) - index) / rate if rate > 0.0 else 0.0
            _write_json(
                status_path,
                {
                    "status": "running",
                    "started_at_utc": started,
                    "updated_at_utc": _utc_now(),
                    "completed": index,
                    "total": len(tasks),
                    "failed": failures,
                    "last_task_id": result.task_id,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": remaining,
                },
            )
            if index == 1 or index % max(1, len(tasks) // 25) == 0 or index == len(tasks):
                print(
                    f"[{index}/{len(tasks)}] elapsed={elapsed / 60.0:.1f} min "
                    f"eta={remaining / 60.0:.1f} min failed={failures}",
                    flush=True,
                )
    return [{"task_id": task.task_id} for task in tasks]


def _resolve_pending(tasks: Sequence[Task], partials_dir: Path, *, force: bool) -> list[Task]:
    pending: list[Task] = []
    for task in tasks:
        path = _partial_path(partials_dir, task.task_id)
        if force or not path.is_file():
            pending.append(task)
            continue
        payload = _load_partial(partials_dir, task.task_id)
        if not _partial_is_compatible(payload):
            raise RuntimeError(f"existing partial {task.task_id} has incompatible settings; rerun with --force")
        if payload.get("status") != "ok":
            print(f"retrying previously failed task {task.task_id}", flush=True)
            pending.append(task)
    return pending


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _midrank_proportion(bootstrap: np.ndarray, estimate: float) -> tuple[float, float, float]:
    """Return (below, equal, mid-rank proportion) for the BCa bias correction."""
    below = float(np.count_nonzero(bootstrap < estimate)) / float(bootstrap.size)
    equal = float(np.count_nonzero(bootstrap == estimate)) / float(bootstrap.size)
    return below, equal, float(below + 0.5 * equal)


def _bca_interval(
    values: Iterable[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int,
) -> dict[str, Any]:
    """Return the BCa bootstrap interval for the sample mean.

    Ties in the bootstrap distribution use the mid-rank correction
    ``(count(< estimate) + 0.5 * count(== estimate)) / n_boot``.  A constant
    sample is an exact degenerate case: every bootstrap and jackknife replicate
    equals the point estimate, so the interval is the point itself.  This is
    reported explicitly with ``bca_degenerate=True``.

    A non-constant sample with a singular bias correction (proportion 0 or 1),
    a non-finite acceleration, or an adjusted quantile outside (0, 1) is an
    error.  Nothing is clipped, and no fallback interval is produced.
    """
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("BCa requires at least two one-dimensional observations")
    if not np.isfinite(array).all():
        raise ValueError("BCa observations must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    estimate = float(array.mean())
    if float(array.max() - array.min()) == 0.0:
        return {
            "mean": estimate,
            "bca_low": estimate,
            "bca_high": estimate,
            "bca_degenerate": True,
            "n_boot": int(n_boot),
            "bca_z0": None,
            "bca_acceleration": None,
            "bca_proportion_below": None,
            "bca_proportion_equal": None,
            "bca_tie_correction": BCA_TIE_CORRECTION,
            "bca_singular_policy": BCA_SINGULAR_POLICY,
        }

    rng = np.random.default_rng(seed)
    n = array.size
    indices = rng.integers(0, n, size=(int(n_boot), n))
    bootstrap = array[indices].mean(axis=1)
    if not np.isfinite(bootstrap).all():
        raise RuntimeError("BCa bootstrap replicates contain non-finite values")
    below, equal, proportion = _midrank_proportion(bootstrap, estimate)
    if not 0.0 < proportion < 1.0:
        raise RuntimeError(
            "BCa bias correction is singular for a non-constant sample: "
            f"mid-rank proportion={proportion!r} (below={below!r}, equal={equal!r})"
        )
    z0 = float(norm.ppf(proportion))
    if not np.isfinite(z0):
        raise RuntimeError(f"BCa bias-correction z0 is non-finite: {z0!r}")

    jackknife = np.array([np.delete(array, index).mean() for index in range(n)], dtype=float)
    jack_mean = float(jackknife.mean())
    deviations = jack_mean - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    if denominator == 0.0:
        raise RuntimeError("BCa acceleration is undefined despite non-constant observations")
    acceleration = float(np.sum(deviations**3) / denominator)
    if not np.isfinite(acceleration):
        raise RuntimeError(f"BCa acceleration is non-finite: {acceleration!r}")

    z_low = float(norm.ppf(alpha / 2.0))
    z_high = float(norm.ppf(1.0 - alpha / 2.0))
    adjusted: list[float] = []
    for z_value in (z_low, z_high):
        shifted = z0 + z_value
        denominator_term = 1.0 - acceleration * shifted
        if denominator_term == 0.0 or not np.isfinite(denominator_term):
            raise RuntimeError("BCa adjusted percentile is undefined")
        inner = z0 + shifted / denominator_term
        if not np.isfinite(inner):
            raise RuntimeError(f"BCa adjusted normal argument is non-finite: {inner!r}")
        level = float(norm.cdf(inner))
        if not 0.0 < level < 1.0:
            raise RuntimeError(
                f"BCa adjusted percentile is outside (0, 1): {level!r}; refusing to clip it to the boundary"
            )
        adjusted.append(level)
    low_quantile, high_quantile = adjusted
    low = float(np.quantile(bootstrap, low_quantile, method="linear"))
    high = float(np.quantile(bootstrap, high_quantile, method="linear"))
    if not math.isfinite(low) or not math.isfinite(high):
        raise RuntimeError(f"BCa interval endpoints are non-finite: low={low!r}, high={high!r}")
    return {
        "mean": estimate,
        "bca_low": low,
        "bca_high": high,
        "bca_degenerate": False,
        "n_boot": int(n_boot),
        "bca_z0": z0,
        "bca_acceleration": acceleration,
        "bca_proportion_below": below,
        "bca_proportion_equal": equal,
        "bca_tie_correction": BCA_TIE_CORRECTION,
        "bca_singular_policy": BCA_SINGULAR_POLICY,
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    return pack3._wilson_interval(int(successes), int(trials))


def _require_finite_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    """Raise unless every value in the requested numeric columns is finite."""
    for column in columns:
        values = cast(pd.Series, pd.to_numeric(frame[column], errors="raise")).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad = int(np.count_nonzero(~np.isfinite(values)))
            raise RuntimeError(f"{context} column {column!r} has {bad} non-finite values")


def _require_exact_seed_grid(
    frame: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    context: str,
    duplicate_keys: Sequence[str],
) -> None:
    """Require exactly the expected scenario seeds, with no duplicate cells."""
    expected = {int(seed) for seed in expected_seeds}
    duplicates = frame.duplicated(list(duplicate_keys))
    if bool(duplicates.any()):
        first = frame.loc[duplicates, list(duplicate_keys)].head(5).to_dict(orient="records")
        raise RuntimeError(f"{context} has duplicate cells: {first}")
    observed = {int(seed) for seed in frame["scenario_seed"].tolist()}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"{context} seed grid mismatch; missing={missing[:10]}, extra={extra[:10]}")


def validate_s0_cell_grid(
    cells: pd.DataFrame,
    *,
    datasets: Sequence[str],
    expected_seeds: Sequence[int],
) -> None:
    """Validate the S0-clean output grid before any aggregation.

    Each dataset x view must contain exactly the requested seeds (200 by default)
    with one row per seed/device, and every continuous result must be finite.
    """
    numeric = ("observed_max", "q95_max", "q99_max", "p_value_max", "kpi_auc", "n_features")
    for dataset, view in product(datasets, S0_VIEWS):
        label = pack3.DATASET_LABELS[dataset]
        subset = cells.loc[(cells["dataset"] == label) & (cells["view"] == view)]
        context = f"S0-clean {label} {view}"
        if subset.empty:
            raise RuntimeError(f"{context} has no cells")
        _require_exact_seed_grid(
            subset,
            expected_seeds=expected_seeds,
            context=context,
            duplicate_keys=("scenario_seed", "device"),
        )
        _require_finite_columns(subset, numeric, context)
        n_features = cast(pd.Series, pd.to_numeric(subset["n_features"], errors="raise"))
        if bool((n_features <= 0).any()):
            raise RuntimeError(f"{context} has a non-positive feature count")


def validate_weak_metric_grid(
    metrics: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
    levels: Sequence[float] = WEAK_METRIC_LEVELS,
    distances: Sequence[str] = pack3.DISTANCES,
) -> None:
    """Validate every weak level x distance cell before aggregation."""
    present = {
        (round(float(level), 10), str(distance))
        for level, distance in zip(metrics["level"], metrics["distance"], strict=True)
    }
    expected = {(round(float(level), 10), str(distance)) for level, distance in product(levels, distances)}
    if present != expected:
        raise RuntimeError(
            f"weak-intensity level/distance grid mismatch; missing={sorted(expected - present)}, "
            f"extra={sorted(present - expected)}"
        )
    numeric = (*WEAK_METRIC_NAMES, "kpi_auc")
    for level, distance in product(levels, distances):
        subset = metrics.loc[
            np.isclose(metrics["level"].astype(float), float(level)) & (metrics["distance"] == distance)
        ]
        context = f"weak-intensity level {level:g} distance {distance}"
        if subset.empty:
            raise RuntimeError(f"{context} has no rows")
        _require_exact_seed_grid(
            subset,
            expected_seeds=expected_seeds,
            context=context,
            duplicate_keys=("scenario_seed",),
        )
        _require_finite_columns(subset, numeric, context)


def summarize_s0_cells(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the S0-clean exceedance, cutoff-distribution and per-feature tables."""
    if cells.empty:
        raise ValueError("cannot summarize an empty S0-clean cell table")
    missing = set(S0_CELL_COLUMNS) - set(cells.columns)
    if missing:
        raise ValueError(f"S0-clean cells are missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    per_feature_rows: list[dict[str, Any]] = []
    for group_key, group in cells.groupby(["dataset", "view"], observed=True):
        dataset, view = cast(tuple[str, str], group_key)
        trials = len(group)
        successes_95 = int(group["fwer_reject_95"].sum())
        successes_99 = int(group["fwer_reject_99"].sum())
        low_95, high_95 = _wilson(successes_95, trials)
        low_99, high_99 = _wilson(successes_99, trials)
        per_feature_rate = group["n_features_exceed_tau95"].to_numpy(dtype=float) / group["n_features"].to_numpy(
            dtype=float
        )
        rows.append(
            {
                "dataset": dataset,
                "view": view,
                "n_tests": trials,
                "n_scenario_seeds": int(group["scenario_seed"].nunique()),
                "n_device_tests": int(group["device"].nunique() * group["scenario_seed"].nunique()),
                "n_exceed_95": successes_95,
                "exceedance_rate_95": successes_95 / trials,
                "wilson_low_95": low_95,
                "wilson_high_95": high_95,
                "expected_rate_95": 0.05,
                "n_exceed_99": successes_99,
                "exceedance_rate_99": successes_99 / trials,
                "wilson_low_99": low_99,
                "wilson_high_99": high_99,
                "expected_rate_99": 0.01,
                "per_feature_fpr_95_mean": float(per_feature_rate.mean()),
                "per_feature_fpr_95_std": float(per_feature_rate.std(ddof=1)) if trials > 1 else 0.0,
                "mean_features_exceed_tau95": float(group["n_features_exceed_tau95"].mean()),
                "mean_features_exceed_tau99": float(group["n_features_exceed_tau99"].mean()),
                "mean_n_features": float(group["n_features"].mean()),
                "mean_kpi_auc": float(group["kpi_auc"].mean()),
                "mean_observed_max": float(group["observed_max"].mean()),
                "mean_q95_max": float(group["q95_max"].mean()),
                "mean_q99_max": float(group["q99_max"].mean()),
            }
        )
        for metric in S0_DISTRIBUTION_METRICS:
            values = group[metric].to_numpy(dtype=float)
            distribution_rows.append(
                {
                    "dataset": dataset,
                    "view": view,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "q05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                    "max": float(values.max()),
                }
            )
        per_feature_rows.append(
            {
                "dataset": dataset,
                "view": view,
                "n_tests": trials,
                "mean_per_feature_fpr_95": float(per_feature_rate.mean()),
                "std_per_feature_fpr_95": float(per_feature_rate.std(ddof=1)) if trials > 1 else 0.0,
                "min_per_feature_fpr_95": float(per_feature_rate.min()),
                "max_per_feature_fpr_95": float(per_feature_rate.max()),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["dataset", "view"], kind="stable"),
        pd.DataFrame(distribution_rows).sort_values(["dataset", "view", "metric"], kind="stable"),
        pd.DataFrame(per_feature_rows).sort_values(["dataset", "view"], kind="stable"),
    )


def _weak_metric_summary_row(
    metrics: pd.DataFrame,
    *,
    level: float,
    distance: str,
    metric: str,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize one level x distance x metric cell over scenario seeds."""
    subset = metrics.loc[np.isclose(metrics["level"].astype(float), float(level)) & (metrics["distance"] == distance)]
    if subset.empty:
        raise RuntimeError(f"weak-intensity level {level:g} distance {distance} has no rows")
    if float(level) == NO_FAULT_LEVEL:
        raise RuntimeError("the no-fault endpoint must not contribute MRR/HR metric rows")
    seeds = subset["scenario_seed"].astype(int).to_numpy()
    if len(set(seeds.tolist())) != len(seeds):
        raise RuntimeError(f"level {level} distance {distance} has duplicate scenario seeds")
    values = cast(pd.Series, pd.to_numeric(subset[metric], errors="raise")).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"level {level} distance {distance} metric {metric} contains non-finite values")
    interval = _bca_interval(
        values,
        n_boot=n_boot,
        seed=pack3._stable_seed("bca", level, distance, metric, bootstrap_seed),
    )
    return {
        "dataset": "Dataset B",
        "level": float(level),
        "distance": distance,
        "metric": metric,
        "n_scenario_seeds": int(values.size),
        "mean": interval["mean"],
        "std": float(values.std(ddof=1)),
        "bca_low_95": interval["bca_low"],
        "bca_high_95": interval["bca_high"],
        "bca_degenerate": interval["bca_degenerate"],
        "bca_z0": interval["bca_z0"],
        "bca_acceleration": interval["bca_acceleration"],
        "bca_proportion_below": interval["bca_proportion_below"],
        "bca_proportion_equal": interval["bca_proportion_equal"],
        "bca_tie_correction": interval["bca_tie_correction"],
        "bca_singular_policy": interval["bca_singular_policy"],
        "n_boot": interval["n_boot"],
        "random_mrr_expectation": RANDOM_MRR_EXPECTATION if metric == "mrr" else None,
    }


def summarize_weak_metrics(
    metrics: pd.DataFrame,
    *,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate MRR/HR over scenario seeds with mean, SD and 95% BCa intervals."""
    columns = [
        "dataset",
        "level",
        "distance",
        "metric",
        "n_scenario_seeds",
        "mean",
        "std",
        "bca_low_95",
        "bca_high_95",
        "bca_degenerate",
        "bca_z0",
        "bca_acceleration",
        "bca_proportion_below",
        "bca_proportion_equal",
        "bca_tie_correction",
        "bca_singular_policy",
        "n_boot",
        "random_mrr_expectation",
    ]
    if metrics.empty:
        raise ValueError("cannot summarize an empty weak-intensity metric table")
    if bool(np.isclose(metrics["level"].astype(float), NO_FAULT_LEVEL).any()):
        raise RuntimeError("the no-fault endpoint must not contribute MRR/HR metric rows")
    rows = [
        _weak_metric_summary_row(
            metrics,
            level=level,
            distance=str(distance),
            metric=metric,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
        )
        for level, distance, metric in product(WEAK_METRIC_LEVELS, pack3.DISTANCES, WEAK_METRIC_NAMES)
    ]
    summary = pd.DataFrame(rows)
    summary["distance"] = pd.Categorical(
        summary["distance"], categories=["wasserstein", "mean", "kl", "js"], ordered=True
    )
    summary = summary.sort_values(["level", "distance", "metric"], kind="stable").reset_index(drop=True)
    per_seed = metrics.sort_values(["level", "distance", "scenario_seed"], kind="stable").reset_index(drop=True)
    return _ensure_columns(summary, columns), per_seed


# ---------------------------------------------------------------------------
# Compatibility and environment manifests
# ---------------------------------------------------------------------------


def compatibility_with_pack3(
    cells: pd.DataFrame,
    pack3_partials_dir: Path,
    *,
    datasets: Sequence[str],
    seeds: Sequence[int] = tuple(range(20)),
) -> pd.DataFrame:
    """Compare the new S0-clean standard view with the existing pack3 partials.

    The comparison is read-only and exact.  It establishes whether the current
    code reproduces seeds 0..19 of the old 20-seed cohort; it does not make the
    new 200-seed cohort exchangeable with the old one by construction.
    """
    fields = (
        "n_permutations",
        "observed_max",
        "q95_max",
        "q99_max",
        "p_value_max",
        "fwer_reject_95",
        "fwer_reject_99",
        "n_features_exceed_tau95",
        "n_features_exceed_tau99",
        "top1_feature",
        "observed_argmax_feature",
    )
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            task_id = f"base_{dataset}_S0-clean_seed{seed}"
            path = pack3_partials_dir / f"{task_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"pack3 compatibility partial is missing: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok":
                raise RuntimeError(f"pack3 compatibility partial {task_id} is not ok")
            reference = [cell for cell in payload["tables"]["perm_null_cells"] if str(cell.get("view")) == "standard"]
            if len(reference) != 1:
                raise RuntimeError(f"pack3 partial {task_id} has {len(reference)} standard S0-clean cells")
            ours = cells.loc[
                (cells["dataset"] == pack3.DATASET_LABELS[dataset])
                & (cells["scenario"] == "S0-clean")
                & (cells["scenario_seed"] == seed)
                & (cells["view"] == "standard")
            ]
            if len(ours) != 1:
                raise RuntimeError(f"new S0-clean results have {len(ours)} standard cells for {dataset} seed {seed}")
            reference_cell = reference[0]
            our_cell = ours.iloc[0]
            for field_name in fields:
                ours_value = our_cell[field_name]
                reference_value = reference_cell[field_name]
                if isinstance(ours_value, (float, np.floating)) or isinstance(reference_value, (float, np.floating)):
                    matches = bool(np.isclose(float(ours_value), float(reference_value), rtol=0.0, atol=0.0))
                else:
                    matches = str(ours_value) == str(reference_value)
                rows.append(
                    {
                        "dataset": pack3.DATASET_LABELS[dataset],
                        "scenario_seed": int(seed),
                        "view": "standard",
                        "field": field_name,
                        "new_value": ours_value,
                        "pack3_value": reference_value,
                        "exact_match": matches,
                    }
                )
    result = pd.DataFrame(rows)
    mismatches = result.loc[~result["exact_match"].astype(bool)]
    if not mismatches.empty:
        details = mismatches.head(20).to_dict(orient="records")
        raise RuntimeError(
            f"new S0-clean results do not exactly reproduce pack3 seeds 0..19; first mismatches: {details}"
        )
    return result


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    detected = platform.processor().strip()
    return detected or platform.machine() or "unknown"


def _memory_bytes() -> tuple[int | None, int | None]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", maxsplit=1)
            parts = raw.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
        return values.get("MemTotal"), values.get("MemAvailable")
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        total = None
    return total, None


def _package_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": pack3.scipy.__version__,
        "scikit_learn": pack3.sklearn.__version__,
        "shap": pack3.shap.__version__,
        "xgboost": pack3.xgboost.__version__,
    }


def environment_manifest(*, workers: int, raw_dir: Path) -> dict[str, Any]:
    """Capture the environment of *this* run, not a retroactive pack3 runtime."""
    total_memory, available_memory = _memory_bytes()
    return {
        "recorded_at_utc": _utc_now(),
        "recorded_for": "bdcc_internal_review_null_weak",
        "note": "Environment captured for this new experiment; it is not a retroactive pack3 runtime record.",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_logical_count": os.cpu_count(),
        "memory_total_bytes": total_memory,
        "memory_available_bytes": available_memory,
        "workers": int(workers),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "raw_dir": str(raw_dir.resolve()),
        "raw_dir_sha256": {
            name: _sha256(raw_dir / name) for name in ("secom.data", "secom_labels.data", "secom.names")
        },
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def protocol_manifest(
    *,
    s0_seeds: Sequence[int],
    weak_seeds: Sequence[int],
    datasets: Sequence[str],
    n_permutations: int,
    n_boot: int,
) -> dict[str, Any]:
    """Build the protocol manifest for both experiment blocks."""
    return {
        "s0_clean": {
            "datasets": list(datasets),
            "scenario_seeds": list(s0_seeds),
            "views": list(S0_VIEWS),
            "n_permutations": int(n_permutations),
            "event": "observed_max > q95_max (process-max permutation cutoff)",
            "cutoff_definition": "q95_max = 95th percentile of the permutation maximum over all process features",
        },
        "weak_b": {
            "dataset": "Dataset B",
            "scenario": "S2",
            "levels": list(WEAK_LEVELS),
            "metric_levels": list(WEAK_METRIC_LEVELS),
            "no_fault_endpoint_level": NO_FAULT_LEVEL,
            "no_fault_endpoint": (
                "mechanism local weight equals the normal-pool global weight; root manifest is empty; "
                "no MRR/HR is computed"
            ),
            "scenario_seeds": list(weak_seeds),
            "views": ["standard"],
            "distances": ["wasserstein", "mean", "kl", "js"],
            "metrics": ["mrr", "hr_at_1", "hr_at_3", "hr_at_5"],
            "n_boot": int(n_boot),
            "bootstrap_interval": "BCa for the sample mean over scenario seeds",
            "bca_tie_correction": BCA_TIE_CORRECTION,
            "bca_singular_policy": BCA_SINGULAR_POLICY,
            "bca_constant_sample": "explicit degenerate interval at the constant value",
            "statistical_unit": "scenario seed",
            "random_mrr_expectation_theoretical_only": RANDOM_MRR_EXPECTATION,
            "random_mrr_expectation_formula": "H_590 / 590",
        },
        "shared": {
            "model_seed": int(pack3.MODEL_SEED),
            "cv_splits": int(pack3.CV_SPLITS),
            "model_params": pack3.MODEL_PARAMS,
            "dataset_a_params": pack3.DATASET_A_PARAMS,
            "dataset_b_params": pack3.DATASET_B_PARAMS,
            "gads_distances": list(pack3.DISTANCES),
            "gads_bins": int(DEFAULT_GADS_BINS),
            "gads_epsilon": float(DEFAULT_GADS_EPSILON),
            "implementation": "scripts.bdcc_pack3_diagnostics_and_bsweep",
        },
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _merge(payloads: Sequence[dict[str, Any]], name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("tables", {}).get(name, []))
    return pd.DataFrame(rows)


def _assemble_s0(
    payloads: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    environment: dict[str, Any],
    protocol: dict[str, Any],
    pack3_partials_dir: Path | None,
    datasets: Sequence[str],
) -> dict[str, Any]:
    s0_payloads = [payload for payload in payloads if payload.get("kind") == "s0_clean"]
    cells = _ensure_columns(_merge(s0_payloads, "s0_cells"), S0_CELL_COLUMNS)
    if cells.empty:
        raise RuntimeError("no S0-clean cells were produced")
    expected_seeds = protocol["s0_clean"]["scenario_seeds"]
    validate_s0_cell_grid(cells, datasets=datasets, expected_seeds=expected_seeds)
    exceedance, distribution, per_feature = summarize_s0_cells(cells)
    compatibility = pd.DataFrame()
    compatibility_status = "not_checked"
    if pack3_partials_dir is not None:
        compatibility = compatibility_with_pack3(cells, pack3_partials_dir, datasets=datasets)
        compatibility_status = "exact_match_0_19"
    _write_csv(cells.sort_values(["dataset", "view", "scenario_seed"], kind="stable"), output_dir / "cells.csv")
    _write_csv(exceedance, output_dir / "exceedance_summary.csv")
    _write_csv(distribution, output_dir / "threshold_distribution.csv")
    _write_csv(per_feature, output_dir / "per_feature_fpr_summary.csv")
    if not compatibility.empty:
        _write_csv(compatibility, output_dir / "compatibility_pack3_seeds_0_19.csv")
    manifest = {
        "part": "s0_clean",
        "status": "completed",
        "environment": environment,
        "source": {
            "git_commit": git_commit(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _code_sha256(),
            "pack3_script_sha256": _sha256(PROJECT_ROOT / "scripts/bdcc_pack3_diagnostics_and_bsweep.py"),
        },
        "protocol": protocol["s0_clean"],
        "compatibility_with_pack3_0_19": {
            "status": compatibility_status,
            "note": (
                "Seeds 0..19 are only a compatibility check against the old 20-seed cohort. "
                "The new cohort adds seeds 20..199, so exchangeability with the old cohort is not "
                "guaranteed merely by reusing seeds 0..19."
            ),
        },
        "outputs": [
            "cells.csv",
            "exceedance_summary.csv",
            "threshold_distribution.csv",
            "per_feature_fpr_summary.csv",
            "compatibility_pack3_seeds_0_19.csv",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _assemble_weak(
    payloads: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    environment: dict[str, Any],
    protocol: dict[str, Any],
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    endpoint_payloads = [payload for payload in payloads if payload.get("kind") == "weak_b_endpoint"]
    metric_payloads = [payload for payload in payloads if payload.get("kind") == "weak_b"]
    endpoint = _merge(endpoint_payloads, "weak_endpoint")
    if endpoint.empty:
        raise RuntimeError("no no-fault endpoint rows were produced")
    if not bool(endpoint["root_manifest_empty"].all()):
        raise RuntimeError("the no-fault endpoint has a non-empty root manifest")
    if not bool((endpoint["mechanism_effect"].astype(float).abs() < 1e-12).all()):
        raise RuntimeError("the no-fault endpoint has a non-zero mechanism effect")
    metrics = _ensure_columns(_merge(metric_payloads, "weak_metrics"), pack3.SWEEP_METRIC_COLUMNS)
    if metrics.empty:
        raise RuntimeError("no weak-intensity metric rows were produced")
    present_levels = set(metrics["level"].astype(float).round(10).tolist())
    if round(NO_FAULT_LEVEL, 10) in present_levels:
        raise RuntimeError("the no-fault endpoint must not produce MRR/HR rows")
    validate_weak_metric_grid(
        metrics,
        expected_seeds=protocol["weak_b"]["scenario_seeds"],
        levels=WEAK_METRIC_LEVELS,
        distances=pack3.DISTANCES,
    )
    summary, per_seed = summarize_weak_metrics(metrics, n_boot=n_boot, bootstrap_seed=bootstrap_seed)
    _write_csv(endpoint.sort_values("scenario_seed", kind="stable"), output_dir / "no_fault_endpoint.csv")
    _write_csv(per_seed, output_dir / "metrics_per_seed.csv")
    _write_csv(summary, output_dir / "metric_summary.csv")
    manifest = {
        "part": "weak_b",
        "status": "completed",
        "environment": environment,
        "source": {
            "git_commit": git_commit(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _code_sha256(),
            "pack3_script_sha256": _sha256(PROJECT_ROOT / "scripts/bdcc_pack3_diagnostics_and_bsweep.py"),
        },
        "protocol": protocol["weak_b"],
        "no_fault_endpoint_validation": {
            "rows": len(endpoint),
            "root_manifest_empty": True,
            "mechanism_effect_zero": True,
            "no_mrr_hr_rows": True,
        },
        "outputs": [
            "no_fault_endpoint.csv",
            "metrics_per_seed.csv",
            "metric_summary.csv",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--part", choices=("both", "s0", "weak"), default="both")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--s0-seeds", default="0:200", help="S0-clean seed range START:END (default 0:200)")
    parser.add_argument("--weak-seeds", default="0:20", help="weak-intensity seed range START:END (default 0:20)")
    parser.add_argument("--n-permutations", type=int, default=int(pack3.DEFAULT_N_PERMUTATIONS))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pack3-partials", type=Path, default=None, help="read-only pack3 partial directory")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip fitting and recompute summaries from existing training partials",
    )
    parser.add_argument(
        "--partials-root",
        type=Path,
        default=None,
        help="root containing s0_clean/partials and weak_b/partials (default: --output-dir)",
    )
    parser.add_argument("--force", action="store_true", help="rerun every task, ignoring existing partials")
    parser.add_argument("--dry-run", action="store_true", help="print the task grid and exit")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], Path]:
    if args.n_permutations < 100:
        raise ValueError("--n-permutations must be at least 100")
    if args.n_boot < 100:
        raise ValueError("--n-boot must be at least 100")
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    if args.aggregate_only and args.force:
        raise ValueError("--aggregate-only cannot be combined with --force because it never fits")
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"SECOM raw directory does not exist: {raw_dir}")
    verify_secom_files(raw_dir)
    s0_seeds = _parse_seed_range(args.s0_seeds)
    weak_seeds = _parse_seed_range(args.weak_seeds)
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets contains duplicates")
    if args.pack3_partials is not None and not args.pack3_partials.is_dir():
        raise FileNotFoundError(f"pack3 partial directory does not exist: {args.pack3_partials}")
    return s0_seeds, weak_seeds, raw_dir


def _print_grid(
    s0_tasks: Sequence[S0Task],
    weak_metric_tasks: Sequence[WeakMetricTask],
    endpoint_tasks: Sequence[WeakEndpointTask],
) -> None:
    print(f"S0-clean tasks: {len(s0_tasks)}")
    print(f"weak-intensity metric tasks: {len(weak_metric_tasks)}")
    print(f"weak-intensity endpoint checks (no fit): {len(endpoint_tasks)}")
    for task in (*s0_tasks, *endpoint_tasks, *weak_metric_tasks):
        print(f"  {task.task_id}")


def _run_aggregation(
    *,
    args: argparse.Namespace,
    s0_tasks: Sequence[S0Task],
    weak_metric_tasks: Sequence[WeakMetricTask],
    endpoint_tasks: Sequence[WeakEndpointTask],
    output_dir: Path,
    environment: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Recompute summaries from existing partials without fitting anything."""
    partials_root = (args.partials_root or output_dir).expanduser().resolve()
    aggregation_hash = _code_sha256()
    common = {
        "mode": "aggregate_only",
        "partials_root": str(partials_root),
        "aggregation_script_sha256": aggregation_hash,
        "bca_tie_correction": BCA_TIE_CORRECTION,
        "bca_singular_policy": BCA_SINGULAR_POLICY,
        "recorded_at_utc": _utc_now(),
    }
    if args.part in ("both", "s0"):
        s0_partials = partials_root / "s0_clean" / "partials"
        payloads, failures, training_hash = _load_partials_for_aggregation(s0_partials, s0_tasks)
        if failures:
            raise RuntimeError(f"S0-clean aggregation found failed tasks: {failures[:3]}")
        _assemble_s0(
            payloads,
            output_dir=output_dir / "s0_clean",
            environment=environment,
            protocol=protocol,
            pack3_partials_dir=args.pack3_partials.resolve() if args.pack3_partials is not None else None,
            datasets=tuple(args.datasets),
        )
        _write_json(
            output_dir / "s0_clean" / "reaggregation_manifest.json",
            {**common, "part": "s0_clean", "training_script_sha256": training_hash},
        )
    if args.part in ("both", "weak"):
        weak_partials = partials_root / "weak_b" / "partials"
        payloads, failures, training_hash = _load_partials_for_aggregation(
            weak_partials, [*endpoint_tasks, *weak_metric_tasks]
        )
        if failures:
            raise RuntimeError(f"weak-intensity aggregation found failed tasks: {failures[:3]}")
        _assemble_weak(
            payloads,
            output_dir=output_dir / "weak_b",
            environment=environment,
            protocol=protocol,
            n_boot=args.n_boot,
            bootstrap_seed=args.bootstrap_seed,
        )
        _write_json(
            output_dir / "weak_b" / "reaggregation_manifest.json",
            {**common, "part": "weak_b", "training_script_sha256": training_hash},
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the S0-clean and weak-intensity experiments end to end."""
    args = _parse_args(argv)
    s0_seeds, weak_seeds, raw_dir = _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    s0_dir = output_dir / "s0_clean"
    weak_dir = output_dir / "weak_b"
    s0_partials = s0_dir / "partials"
    weak_partials = weak_dir / "partials"
    s0_dir.mkdir(parents=True, exist_ok=True)
    weak_dir.mkdir(parents=True, exist_ok=True)
    s0_partials.mkdir(parents=True, exist_ok=True)
    weak_partials.mkdir(parents=True, exist_ok=True)

    s0_tasks, weak_metric_tasks, endpoint_tasks = build_tasks(
        part=args.part,
        datasets=args.datasets,
        s0_seeds=s0_seeds,
        weak_seeds=weak_seeds,
    )
    if args.dry_run:
        _print_grid(s0_tasks, weak_metric_tasks, endpoint_tasks)
        return

    global _TASK_SETTINGS
    _TASK_SETTINGS = {**_base_settings(args.n_permutations), "raw_dir": str(raw_dir)}
    environment = environment_manifest(workers=args.workers, raw_dir=raw_dir)
    protocol = protocol_manifest(
        s0_seeds=s0_seeds,
        weak_seeds=weak_seeds,
        datasets=tuple(args.datasets),
        n_permutations=args.n_permutations,
        n_boot=args.n_boot,
    )
    if args.aggregate_only:
        _run_aggregation(
            args=args,
            s0_tasks=s0_tasks,
            weak_metric_tasks=weak_metric_tasks,
            endpoint_tasks=endpoint_tasks,
            output_dir=output_dir,
            environment=environment,
            protocol=protocol,
        )
        return
    _write_json(
        output_dir / "run_manifest.json",
        {
            "status": "starting",
            "started_at_utc": _utc_now(),
            "environment": environment,
            "source": {
                "git_commit": git_commit(),
                "script": str(Path(__file__).resolve()),
                "script_sha256": _code_sha256(),
                "pack3_script_sha256": _sha256(PROJECT_ROOT / "scripts/bdcc_pack3_diagnostics_and_bsweep.py"),
            },
            "protocol": protocol,
        },
    )

    started = time.monotonic()
    if s0_tasks:
        pending = _resolve_pending(s0_tasks, s0_partials, force=args.force)
        print(f"S0-clean: {len(pending)} pending of {len(s0_tasks)}", flush=True)
        _run_pool(
            pending,
            workers=args.workers,
            partials_dir=s0_partials,
            status_path=s0_dir / "run_status.json",
            raw_dir=raw_dir,
        )
    if weak_metric_tasks or endpoint_tasks:
        pending_weak = [
            *_resolve_pending(endpoint_tasks, weak_partials, force=args.force),
            *_resolve_pending(weak_metric_tasks, weak_partials, force=args.force),
        ]
        print(
            f"weak-intensity: {len(pending_weak)} pending of {len(endpoint_tasks) + len(weak_metric_tasks)}",
            flush=True,
        )
        _run_pool(
            pending_weak,
            workers=args.workers,
            partials_dir=weak_partials,
            status_path=weak_dir / "run_status.json",
            raw_dir=raw_dir,
        )

    all_payloads: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    if s0_tasks:
        s0_payloads, s0_failures = _load_all_partials(s0_partials, s0_tasks)
        all_payloads.extend(s0_payloads)
        all_failures.extend(s0_failures)
    if weak_metric_tasks or endpoint_tasks:
        weak_payloads, weak_failures = _load_all_partials(weak_partials, [*endpoint_tasks, *weak_metric_tasks])
        all_payloads.extend(weak_payloads)
        all_failures.extend(weak_failures)

    if all_failures:
        _write_json(
            output_dir / "failures.json",
            {"failures": all_failures, "recorded_at_utc": _utc_now()},
        )
        first = "\n".join(f"{item['task_id']}: {item['error']}" for item in all_failures[:3])
        raise RuntimeError(f"{len(all_failures)} tasks failed; see failures.json. First failures:\n{first}")

    s0_manifest: dict[str, Any] | None = None
    weak_manifest: dict[str, Any] | None = None
    if s0_tasks:
        s0_manifest = _assemble_s0(
            all_payloads,
            output_dir=s0_dir,
            environment=environment,
            protocol=protocol,
            pack3_partials_dir=args.pack3_partials.resolve() if args.pack3_partials is not None else None,
            datasets=tuple(args.datasets),
        )
    if weak_metric_tasks or endpoint_tasks:
        weak_manifest = _assemble_weak(
            all_payloads,
            output_dir=weak_dir,
            environment=environment,
            protocol=protocol,
            n_boot=args.n_boot,
            bootstrap_seed=args.bootstrap_seed,
        )

    runtime = time.monotonic() - started
    _write_json(
        output_dir / "run_manifest.json",
        {
            "status": "completed",
            "started_at_utc": environment["recorded_at_utc"],
            "completed_at_utc": _utc_now(),
            "runtime_seconds": runtime,
            "environment": environment,
            "source": {
                "git_commit": git_commit(),
                "script": str(Path(__file__).resolve()),
                "script_sha256": _code_sha256(),
                "pack3_script_sha256": _sha256(PROJECT_ROOT / "scripts/bdcc_pack3_diagnostics_and_bsweep.py"),
            },
            "protocol": protocol,
            "parts": {
                "s0_clean": s0_manifest["status"] if s0_manifest is not None else "not_run",
                "weak_b": weak_manifest["status"] if weak_manifest is not None else "not_run",
            },
            "task_counts": {
                "s0_clean": len(s0_tasks),
                "weak_metric": len(weak_metric_tasks),
                "weak_endpoint": len(endpoint_tasks),
            },
        },
    )
    print(f"completed in {runtime / 60.0:.1f} min; outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()

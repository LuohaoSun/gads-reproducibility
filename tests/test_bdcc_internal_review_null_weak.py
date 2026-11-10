"""Focused tests for the internal-review S0-clean and weak-intensity script.

These tests deliberately avoid fitting models.  They verify the experiment
design invariants that would silently invalidate the run if broken:

* the weight-0.2 Dataset B endpoint really has an empty root manifest and a
  zero mechanism effect;
* the endpoint never enters the MRR/HR metric grid;
* the S0-clean grid covers exactly the requested 200 seeds and both views;
* the run manifest records this run's CPU/RAM provenance and source hash.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

null_weak: ModuleType = importlib.import_module("scripts.bdcc_internal_review_null_weak")
RAW_DIR = ROOT / "data/raw/secom"


@pytest.fixture()
def raw_dir() -> Path:
    """Return the SECOM raw directory, skipping when the official files are absent."""
    if not (RAW_DIR / "secom.data").is_file():
        pytest.skip(f"SECOM raw files not found in {RAW_DIR}; run scripts/download_secom.py")
    null_weak.pack3.RAW_DIR = RAW_DIR
    null_weak.pack3._load_bundle.cache_clear()
    return RAW_DIR


def test_weak_grid_excludes_the_no_fault_endpoint_from_metrics() -> None:
    """The 0.2 endpoint is a manifest check only; 0.4--1.5 carry MRR/HR."""
    _s0, metric_tasks, endpoint_tasks = null_weak.build_tasks(
        part="weak",
        datasets=("a", "b"),
        s0_seeds=(),
        weak_seeds=tuple(range(20)),
    )
    assert len(endpoint_tasks) == 20
    assert {task.level for task in endpoint_tasks} == {null_weak.NO_FAULT_LEVEL}
    assert {task.scenario_seed for task in endpoint_tasks} == set(range(20))

    expected_levels = {0.4, 0.6, 0.8, 1.0, 1.2, 1.5}
    assert {task.level for task in metric_tasks} == expected_levels
    assert null_weak.NO_FAULT_LEVEL not in {task.level for task in metric_tasks}
    assert len(metric_tasks) == 6 * 20
    for level in expected_levels:
        seeds = {task.scenario_seed for task in metric_tasks if task.level == level}
        assert seeds == set(range(20))


def test_s0_grid_covers_200_seeds_and_both_views() -> None:
    """S0-clean runs a fresh 200-seed cohort for both datasets and both views."""
    s0_tasks, metric_tasks, endpoint_tasks = null_weak.build_tasks(
        part="s0",
        datasets=("a", "b"),
        s0_seeds=tuple(range(200)),
        weak_seeds=tuple(range(20)),
    )
    assert not metric_tasks and not endpoint_tasks
    assert len(s0_tasks) == 2 * 200
    assert {task.dataset for task in s0_tasks} == {"a", "b"}
    assert {task.scenario_seed for task in s0_tasks} == set(range(200))
    assert null_weak.S0_VIEWS == ("standard", "centered")


def test_no_fault_endpoint_has_empty_root_manifest_and_zero_effect(raw_dir: Path) -> None:
    """The weight-0.2 endpoint must not claim a fault or a mechanism shift."""
    bundle, row = null_weak._no_fault_endpoint_bundle(0)
    assert bundle.root_causes == {}
    assert bundle.harmless_features == ()
    assert row["no_fault_endpoint"] is True
    assert row["root_manifest_empty"] is True
    assert row["n_root_causes"] == 0
    assert row["mechanism_local_weight"] == pytest.approx(null_weak.NO_FAULT_LEVEL)
    assert row["mechanism_local_weight"] == pytest.approx(row["mechanism_global_weight"])
    assert row["mechanism_effect"] == pytest.approx(0.0, abs=1e-12)
    assert bundle.metadata["null_weak_endpoint"]["root_manifest_empty"] is True
    assert raw_dir.is_dir()


def test_weak_metric_summary_rejects_no_fault_endpoint_rows() -> None:
    """A 0.2 row in the metric table is a hard design violation, not a fallback."""
    metrics = pd.DataFrame(
        {
            "level": [null_weak.NO_FAULT_LEVEL],
            "distance": ["wasserstein"],
            "scenario_seed": [0],
            "mrr": [1.0],
            "hr_at_1": [1.0],
            "hr_at_3": [1.0],
            "hr_at_5": [1.0],
        }
    )
    with pytest.raises(RuntimeError, match="no-fault endpoint"):
        null_weak.summarize_weak_metrics(metrics, n_boot=100, bootstrap_seed=42)


def test_environment_manifest_records_current_run_provenance(raw_dir: Path) -> None:
    """The manifest describes this run's host, not a retroactive pack3 runtime."""
    manifest = null_weak.environment_manifest(workers=12, raw_dir=raw_dir)
    assert manifest["recorded_for"] == "bdcc_internal_review_null_weak"
    assert "not a retroactive pack3 runtime" in manifest["note"]
    assert manifest["workers"] == 12
    assert manifest["cpu_model"]
    assert manifest["cpu_logical_count"] and manifest["cpu_logical_count"] >= 1
    assert manifest["memory_total_bytes"] and manifest["memory_total_bytes"] > 0
    assert manifest["python_executable"]
    assert manifest["raw_dir"] == str(raw_dir.resolve())
    for digest in manifest["raw_dir_sha256"].values():
        assert len(digest) == 64
        int(digest, 16)


def test_random_mrr_reference_matches_harmonic_formula() -> None:
    """The random MRR reference is H_590/590 and is never a hypothesis test."""
    assert pytest.approx(0.01179353461022657, rel=0, abs=1e-15) == null_weak.RANDOM_MRR_EXPECTATION
    assert np.isfinite(null_weak.RANDOM_MRR_EXPECTATION)


def test_bca_midrank_tie_correction_is_exact() -> None:
    """The bias-correction proportion is (below + 0.5 * equal) / B, not below / B."""
    below, equal, proportion = null_weak._midrank_proportion(np.array([0.0, 0.0, 1.0, 1.0]), 0.5)
    assert (below, equal, proportion) == (0.5, 0.0, 0.5)

    below, equal, proportion = null_weak._midrank_proportion(np.array([0.0, 1.0, 1.0]), 1.0)
    assert below == pytest.approx(1.0 / 3.0)
    assert equal == pytest.approx(2.0 / 3.0)
    assert proportion == pytest.approx(2.0 / 3.0)


def test_bca_discrete_hr_ties_are_finite_and_use_midrank() -> None:
    """Discrete HR values produce many bootstrap ties; the interval stays valid."""
    values = [1.0, 1.0, 1.0, 1.0, 0.0]
    result = null_weak._bca_interval(values, n_boot=2_000, seed=42)
    assert np.isfinite(result["bca_low"]) and np.isfinite(result["bca_high"])
    assert 0.0 <= result["bca_low"] <= result["bca_high"] <= 1.0
    assert result["bca_degenerate"] is False
    assert result["bca_tie_correction"] == null_weak.BCA_TIE_CORRECTION
    assert result["bca_singular_policy"] == null_weak.BCA_SINGULAR_POLICY
    assert result["bca_proportion_equal"] > 0.0


def test_bca_constant_sample_is_explicitly_degenerate() -> None:
    """A constant HR/FAR sample yields a documented point interval, not a fake CI."""
    result = null_weak._bca_interval([1.0, 1.0, 1.0], n_boot=500, seed=7)
    assert result["bca_degenerate"] is True
    assert result["bca_low"] == result["bca_high"] == 1.0
    assert result["bca_tie_correction"] == null_weak.BCA_TIE_CORRECTION


def test_bca_rejects_nonfinite_and_singular_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-finite values and a singular bias correction are hard errors."""
    with pytest.raises(ValueError, match="finite"):
        null_weak._bca_interval([0.0, np.nan, 1.0], n_boot=500, seed=1)

    monkeypatch.setattr(
        null_weak,
        "_midrank_proportion",
        lambda bootstrap, estimate: (0.0, 0.0, 1.0),
    )
    with pytest.raises(RuntimeError, match="singular"):
        null_weak._bca_interval([0.0, 1.0], n_boot=500, seed=1)


def _s0_cell_frame(
    seeds: list[int],
    *,
    dataset: str = "Dataset A",
    views: tuple[str, ...] = ("standard", "centered"),
) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "view": view,
            "scenario_seed": seed,
            "device": "EQP_B",
            "observed_max": 0.1,
            "q95_max": 0.2,
            "q99_max": 0.3,
            "p_value_max": 0.5,
            "kpi_auc": 0.8,
            "n_features": 590,
        }
        for view in views
        for seed in seeds
    ]
    return pd.DataFrame(rows)


def test_s0_grid_validation_rejects_missing_and_duplicate_seeds() -> None:
    """S0 output validation is exact per dataset/view, not merely non-empty."""
    null_weak.validate_s0_cell_grid(_s0_cell_frame([0, 1]), datasets=("a",), expected_seeds=(0, 1))

    with pytest.raises(RuntimeError, match="seed grid mismatch"):
        null_weak.validate_s0_cell_grid(_s0_cell_frame([0]), datasets=("a",), expected_seeds=(0, 1))

    with pytest.raises(RuntimeError, match="duplicate"):
        null_weak.validate_s0_cell_grid(_s0_cell_frame([0, 0]), datasets=("a",), expected_seeds=(0,))


def _weak_cell_frame(
    *,
    levels: tuple[float, ...],
    distances: tuple[str, ...],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = [
        {
            "level": level,
            "distance": distance,
            "scenario_seed": seed,
            "mrr": 1.0,
            "hr_at_1": 1.0,
            "hr_at_3": 1.0,
            "hr_at_5": 1.0,
            "kpi_auc": 0.8,
        }
        for level in levels
        for distance in distances
        for seed in seeds
    ]
    return pd.DataFrame(rows)


def test_weak_grid_validation_rejects_missing_distance_and_duplicate_seed() -> None:
    """Every weak level x distance must have exactly the expected unique seeds."""
    complete = _weak_cell_frame(levels=(0.4, 0.6), distances=("wasserstein", "mean"), seeds=(0, 1))
    null_weak.validate_weak_metric_grid(
        complete,
        expected_seeds=(0, 1),
        levels=(0.4, 0.6),
        distances=("wasserstein", "mean"),
    )

    missing_distance = complete.loc[complete["distance"] == "wasserstein"]
    with pytest.raises(RuntimeError, match="grid mismatch"):
        null_weak.validate_weak_metric_grid(
            missing_distance,
            expected_seeds=(0, 1),
            levels=(0.4, 0.6),
            distances=("wasserstein", "mean"),
        )

    duplicated = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        null_weak.validate_weak_metric_grid(
            duplicated,
            expected_seeds=(0, 1),
            levels=(0.4, 0.6),
            distances=("wasserstein", "mean"),
        )

    nonfinite = complete.copy()
    nonfinite.loc[0, "mrr"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        null_weak.validate_weak_metric_grid(
            nonfinite,
            expected_seeds=(0, 1),
            levels=(0.4, 0.6),
            distances=("wasserstein", "mean"),
        )

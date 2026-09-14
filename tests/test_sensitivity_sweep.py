"""Tests for the sensitivity sweep grid assembly."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sweep: ModuleType = importlib.import_module("scripts.run_sensitivity_sweep")


def test_assemble_sweep_runs_overrides_only_the_swept_factor() -> None:
    """Each cell sets the swept factor to its level and keeps other params at base."""
    specs = sweep.assemble_sweep_runs(
        factors=("noise_level",),
        levels_by_factor={"noise_level": (0.1, 0.4)},
        scenarios=("S1", "S2"),
        seeds=(0, 1),
        base_generation_params=dict(sweep.BASE_GENERATION_PARAMS),
    )
    assert len(specs) == 8
    for spec in specs:
        assert spec.factor == "noise_level"
        assert spec.generation_kwargs["noise_level"] in (0.1, 0.4)
        assert spec.generation_kwargs["n_samples"] == sweep.BASE_GENERATION_PARAMS["n_samples"]
        assert spec.generation_kwargs["distribution_shift"] == sweep.BASE_GENERATION_PARAMS["distribution_shift"]
        assert spec.generation_kwargs["scenario"] in ("S1", "S2")
        assert spec.generation_kwargs["seed"] in (0, 1)
        assert spec.generation_kwargs["scenario"] == spec.scenario
        assert spec.generation_kwargs["seed"] == spec.seed
        assert spec.run_dir == sweep._sweep_run_dir(spec.factor, spec.level, spec.scenario, spec.seed)


def test_assemble_sweep_runs_expands_the_full_grid() -> None:
    specs = sweep.assemble_sweep_runs(
        factors=("n_samples", "n_devices"),
        levels_by_factor={"n_samples": (100, 200), "n_devices": (10,)},
        scenarios=("S1",),
        seeds=(0,),
        base_generation_params=dict(sweep.BASE_GENERATION_PARAMS),
    )
    assert len(specs) == 3
    assert [spec.level for spec in specs] == [100, 200, 10]


def test_assemble_sweep_runs_deduplicates_factors() -> None:
    specs = sweep.assemble_sweep_runs(
        factors=("n_devices", "n_devices"),
        levels_by_factor={"n_devices": (10, 15)},
        scenarios=("S1",),
        seeds=(0,),
        base_generation_params=dict(sweep.BASE_GENERATION_PARAMS),
    )
    assert len(specs) == 2


def test_assemble_sweep_runs_rejects_unknown_factor() -> None:
    with pytest.raises(ValueError, match="unknown sweep factors"):
        sweep.assemble_sweep_runs(
            factors=("n_samples",),
            levels_by_factor={"noise_level": (0.1,)},
            scenarios=("S1",),
            seeds=(0,),
            base_generation_params=dict(sweep.BASE_GENERATION_PARAMS),
        )


def test_assemble_sweep_runs_rejects_empty_levels() -> None:
    with pytest.raises(ValueError, match="has no levels"):
        sweep.assemble_sweep_runs(
            factors=("noise_level",),
            levels_by_factor={"noise_level": ()},
            scenarios=("S1",),
            seeds=(0,),
            base_generation_params=dict(sweep.BASE_GENERATION_PARAMS),
        )


def test_assemble_sweep_runs_rejects_uncovered_factor() -> None:
    with pytest.raises(ValueError, match="must cover every swept factor"):
        sweep.assemble_sweep_runs(
            factors=("noise_level",),
            levels_by_factor={"noise_level": (0.1,)},
            scenarios=("S1",),
            seeds=(0,),
            base_generation_params={"n_samples": 500},
        )


def test_assemble_sweep_runs_rejects_empty_inputs() -> None:
    base = dict(sweep.BASE_GENERATION_PARAMS)
    levels = {"noise_level": (0.1,)}
    with pytest.raises(ValueError, match="at least one sweep factor"):
        sweep.assemble_sweep_runs(
            factors=(), levels_by_factor=levels, scenarios=("S1",), seeds=(0,), base_generation_params=base
        )
    with pytest.raises(ValueError, match="at least one scenario"):
        sweep.assemble_sweep_runs(
            factors=("noise_level",), levels_by_factor=levels, scenarios=(), seeds=(0,), base_generation_params=base
        )
    with pytest.raises(ValueError, match="at least one seed"):
        sweep.assemble_sweep_runs(
            factors=("noise_level",), levels_by_factor=levels, scenarios=("S1",), seeds=(), base_generation_params=base
        )


def test_parse_level_override_types() -> None:
    factor, levels = sweep.parse_level_override("n_samples=100, 200")
    assert factor == "n_samples"
    assert levels == (100, 200)
    assert all(isinstance(level, int) for level in levels)

    factor, levels = sweep.parse_level_override("noise_level=0.1,0.4")
    assert factor == "noise_level"
    assert levels == (0.1, 0.4)
    assert all(isinstance(level, float) for level in levels)


@pytest.mark.parametrize(
    "raw",
    ["noise_level", "=1.0", "noise_level=", "unknown_factor=1.0", "noise_level=,,", "n_samples=1.5"],
)
def test_parse_level_override_rejects_malformed_input(raw: str) -> None:
    with pytest.raises(ValueError):
        sweep.parse_level_override(raw)


def test_default_grids_match_factor_types() -> None:
    assert set(sweep.DEFAULT_SWEEP_LEVELS) == set(sweep.FACTOR_VALUE_TYPES)
    for factor, levels in sweep.DEFAULT_SWEEP_LEVELS.items():
        expected_type = sweep.FACTOR_VALUE_TYPES[factor]
        assert all(isinstance(level, expected_type) for level in levels), factor
        assert levels, factor

"""Scenario contracts for the synthetic Dataset A benchmark.

The experiment runner uses these declarations as the source of truth for
which equipment is abnormal in each independent scenario and which process
features are injected as root causes.  Keeping the declarations separate from
the random data generation makes it harder for an experiment to accidentally
reuse an anomaly from another scenario.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """Ground-truth specification for one Dataset A scenario."""

    name: str
    abnormal_devices: tuple[str, ...]
    root_causes: dict[str, tuple[str, ...]]
    distribution_shift_features: dict[str, tuple[str, ...]]
    mechanism_shift_features: dict[str, tuple[str, ...]]
    harmless_features: tuple[str, ...] = ()
    description: str = ""


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        name="S1",
        abnormal_devices=("EQP_A",),
        root_causes={"EQP_A": ("feat_special_dist",)},
        distribution_shift_features={"EQP_A": ("feat_special_dist",)},
        mechanism_shift_features={},
        description="Parameter/distribution drift on EQP_A only.",
    ),
    "S2": ScenarioSpec(
        name="S2",
        abnormal_devices=("EQP_B",),
        root_causes={"EQP_B": ("feat_special_imp",)},
        distribution_shift_features={},
        mechanism_shift_features={"EQP_B": ("feat_special_imp",)},
        description="Mechanism/conditional-importance shift on EQP_B only.",
    ),
    "S3": ScenarioSpec(
        name="S3",
        abnormal_devices=("EQP_A", "EQP_B", "EQP_C"),
        root_causes={
            "EQP_A": ("feat_special_dist",),
            "EQP_B": ("feat_special_imp",),
            "EQP_C": ("feat_special_mix",),
        },
        distribution_shift_features={
            "EQP_A": ("feat_special_dist",),
            "EQP_C": ("feat_special_mix",),
        },
        mechanism_shift_features={
            "EQP_B": ("feat_special_imp",),
            "EQP_C": ("feat_special_mix",),
        },
        description="Concurrent independent faults on EQP_A, EQP_B and EQP_C.",
    ),
    "S4-main": ScenarioSpec(
        name="S4-main",
        abnormal_devices=("EQP_D",),
        root_causes={"EQP_D": ("feat_special_imp",)},
        distribution_shift_features={},
        mechanism_shift_features={"EQP_D": ("feat_special_imp",)},
        harmless_features=tuple(f"feat_useless_{index}" for index in range(1, 11)),
        description="A real mechanism shift is masked by harmless physical drift.",
    ),
    "S4-null": ScenarioSpec(
        name="S4-null",
        abnormal_devices=("EQP_E",),
        root_causes={"EQP_E": ()},
        distribution_shift_features={},
        mechanism_shift_features={},
        harmless_features=tuple(f"feat_useless_{index}" for index in range(1, 11)),
        description="Harmless physical drift without a KPI root cause (FAR only).",
    ),
}


SCENARIO_ALIASES = {"S4_main": "S4-main", "S4_null": "S4-null"}


def normalize_scenario(scenario: str) -> str:
    """Validate and return the canonical scenario name.

    Args:
        scenario: One of ``S1``, ``S2``, ``S3``, ``S4-main`` or ``S4-null``.

    Raises:
        TypeError: If ``scenario`` is not a string.
        ValueError: If the scenario is unknown.
    """
    if not isinstance(scenario, str):
        raise TypeError(f"scenario must be a string, got {type(scenario).__name__}")
    canonical = SCENARIO_ALIASES.get(scenario, scenario)
    if canonical not in SCENARIO_SPECS:
        valid = ", ".join(SCENARIO_SPECS)
        raise ValueError(f"unknown Dataset A scenario {scenario!r}; expected one of: {valid}")
    return canonical


def get_scenario_spec(scenario: str) -> ScenarioSpec:
    """Return the immutable ground-truth specification for ``scenario``."""
    return SCENARIO_SPECS[normalize_scenario(scenario)]

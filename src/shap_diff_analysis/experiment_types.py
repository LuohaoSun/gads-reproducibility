"""Shared data contracts for reproducible paper experiments."""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

EqpMode = Literal["contextual_process", "no_eqp", "all_features"]
EQP_MODES: tuple[EqpMode, ...] = ("contextual_process", "no_eqp", "all_features")
DEFAULT_EQP_MODE: EqpMode = "contextual_process"


@dataclass(frozen=True)
class DatasetBundle:
    """A generated benchmark dataset together with its evaluation manifest."""

    data: pd.DataFrame
    scenario: str
    seed: int
    group_column: str
    target_column: str
    process_features: tuple[str, ...]
    abnormal_devices: tuple[str, ...]
    normal_devices: tuple[str, ...]
    root_causes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    harmless_features: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation manifest."""
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "group_column": self.group_column,
            "target_column": self.target_column,
            "process_features": list(self.process_features),
            "abnormal_devices": list(self.abnormal_devices),
            "normal_devices": list(self.normal_devices),
            "root_causes": {device: list(features) for device, features in self.root_causes.items()},
            "harmless_features": list(self.harmless_features),
            "metadata": self.metadata,
        }

"""Focused tests for the independent Dataset A scenario generator."""

from typing import Any, cast

import pandas as pd
import pytest

from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.generate_data import generate_data, generate_dataset_bundle


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3", "S4-main", "S4-null"])
def test_scenario_returns_complete_bundle(scenario: str) -> None:
    """Each scenario exposes data and a complete group-level manifest."""
    bundle = generate_dataset_bundle(scenario=scenario, n_samples=300, seed=7)

    assert isinstance(bundle, DatasetBundle)
    assert bundle.scenario == scenario
    assert bundle.data.shape == (300, 102)
    assert bundle.group_column == "EQP"
    assert bundle.target_column == "target"
    assert "EQP" not in bundle.process_features
    assert set(bundle.process_features).issubset(bundle.data.columns)
    assert set(bundle.abnormal_devices).isdisjoint(bundle.normal_devices)
    assert set(bundle.abnormal_devices) | set(bundle.normal_devices) == set(bundle.data["EQP"])
    assert bundle.manifest()["scenario"] == scenario


def test_scenario_ground_truth_isolated() -> None:
    """S1--S3 activate only the declared device-specific root causes."""
    expected = {
        "S1": (("EQP_A",), {"EQP_A": ("feat_special_dist",)}, ()),
        "S2": (("EQP_B",), {"EQP_B": ("feat_special_imp",)}, ()),
        "S3": (
            ("EQP_A", "EQP_B", "EQP_C"),
            {
                "EQP_A": ("feat_special_dist",),
                "EQP_B": ("feat_special_imp",),
                "EQP_C": ("feat_special_mix",),
            },
            (),
        ),
    }
    for scenario, (devices, roots, harmless) in expected.items():
        bundle = generate_dataset_bundle(scenario=scenario, n_samples=300, seed=10)
        assert bundle.abnormal_devices == devices
        assert bundle.root_causes == roots
        assert bundle.harmless_features == harmless


def test_s4_main_and_null_have_independent_negative_control_protocols() -> None:
    """S4's main and null variants both inject measurable harmless drift."""
    main = generate_dataset_bundle(scenario="S4-main", n_samples=1_000, seed=3)
    null = generate_dataset_bundle(scenario="S4-null", n_samples=1_000, seed=3)

    assert main.abnormal_devices == ("EQP_D",)
    assert main.root_causes == {"EQP_D": ("feat_special_imp",)}
    assert null.abnormal_devices == ("EQP_E",)
    assert null.root_causes == {"EQP_E": ()}
    assert len(main.harmless_features) == 10
    assert main.harmless_features == null.harmless_features

    for bundle in (main, null):
        abnormal = bundle.data[bundle.data["EQP"].eq(bundle.abnormal_devices[0])]
        normal = bundle.data[bundle.data["EQP"].isin(bundle.normal_devices)]
        mean_diffs = abnormal.loc[:, bundle.harmless_features].mean() - normal.loc[:, bundle.harmless_features].mean()
        assert (mean_diffs > 2.0).all()


def test_generation_is_reproducible_and_scenarios_do_not_reuse_rows() -> None:
    """Same seed is reproducible, while a different scenario has a new stream."""
    first = generate_dataset_bundle(scenario="S1", n_samples=250, seed=19)
    second = generate_dataset_bundle(scenario="S1", n_samples=250, seed=19)
    assert first.data.equals(second.data)
    assert first.manifest() == second.manifest()

    other = generate_dataset_bundle(scenario="S2", n_samples=250, seed=19)
    assert not first.data.equals(other.data)


def test_legacy_dataframe_wrapper_is_explicit() -> None:
    """The legacy no-scenario call remains a DataFrame API."""
    frame = generate_data(n_samples=100, seed=2)
    bundle = generate_data(n_samples=100, seed=2, scenario="S3")
    explicit_frame = generate_data(n_samples=100, seed=2, scenario="S3", return_bundle=False)

    assert isinstance(frame, pd.DataFrame)
    assert isinstance(bundle, DatasetBundle)
    assert isinstance(explicit_frame, pd.DataFrame)
    assert frame.equals(explicit_frame)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"scenario": "S5"}, ValueError),
        ({"scenario": 1}, TypeError),
        ({"n_samples": 0}, ValueError),
        ({"n_samples": 9}, ValueError),
        ({"n_devices": 9}, ValueError),
        ({"noise_level": -0.1}, ValueError),
        ({"seed": -1}, ValueError),
    ],
)
def test_invalid_generation_parameters_raise(kwargs: dict[str, object], error: type[Exception]) -> None:
    """Invalid scenario and generation settings fail loudly."""
    with pytest.raises(error):
        generate_dataset_bundle(**cast(Any, kwargs))

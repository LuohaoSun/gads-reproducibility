"""Focused tests for the modern baselines M2OE-Group and XPE."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import TypedDict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from shap_diff_analysis.baselines import MODERN_BASELINE_METHODS, rank_baseline_features, rank_m2oe_group
from shap_diff_analysis.experiment_runner import ExperimentSettings, run_experiment_bundle
from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.modeling import CrossFittedAttributions
from shap_diff_analysis.vendor.m2oe.tabular_group_explainer import (
    _normal_dispersion,
    _SharedChoiceMaskingModel,
)
from shap_diff_analysis.xpe import _emd_coupling, _stratified_subsample

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

run_matrix_script: ModuleType = importlib.import_module("scripts.run_experiment_matrix")


class _M2oeFastSettings(TypedDict):
    m2oe_epochs: int
    m2oe_n_neighbors: int
    m2oe_max_group_rows: int
    m2oe_batch_size: int


class _M2oeGroupRankKwargs(TypedDict):
    epochs: int
    n_neighbors: int
    batch_size: int
    max_group_rows: int
    seed: int


M2OE_FAST_SETTINGS: _M2oeFastSettings = {
    "m2oe_epochs": 5,
    "m2oe_n_neighbors": 8,
    "m2oe_max_group_rows": 16,
    "m2oe_batch_size": 16,
}
MEDIAN_FEATURE = "feat_useless_45"  # middle of the 100 Dataset A process features
TOP_QUARTILE = 25  # of the 100 Dataset A process features


def _modern_settings(methods: tuple[str, ...]) -> ExperimentSettings:
    """Fast ExperimentSettings for the modern baselines on small bundles."""
    return ExperimentSettings(methods=methods, cv_splits=3, xpe_permutations=4, **M2OE_FAST_SETTINGS)


def test_modern_baselines_are_opt_in_and_dispatchable() -> None:
    """New methods are supported but excluded from the default method tuple."""
    assert set(MODERN_BASELINE_METHODS) == {"M2OE-Group", "XPE"}
    assert not set(MODERN_BASELINE_METHODS) & set(ExperimentSettings().methods)
    groups = pd.Series(["A", "A", "C", "C"], name="EQP")
    X = pd.DataFrame({"root_a": [1.0, 1.1, 0.0, 0.1], "EQP": [1.0, 1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="unknown ranking method"):
        rank_baseline_features(
            "NotAMethod",
            X=X,
            groups=groups,
            abnormal_devices=("A",),
            normal_devices=("C",),
        )


def test_xpe_requires_target_and_attribution() -> None:
    """XPE fails explicitly without the KPI target and the cross-fitted model."""
    groups = pd.Series(["A", "A", "C", "C"], name="EQP")
    X = pd.DataFrame({"root_a": [1.0, 1.1, 0.0, 0.1], "EQP": [1.0, 1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="XPE requires"):
        rank_baseline_features(
            "XPE",
            X=X,
            groups=groups,
            abnormal_devices=("A",),
            normal_devices=("C",),
            process_features=("root_a",),
        )


def test_m2oe_vendor_dispersion_matches_upstream_formula() -> None:
    """The closed-form normal dispersion equals the upstream O(N^2 D) double sum."""
    rng = np.random.default_rng(0)
    normal_data = rng.normal(size=(30, 5))
    differences = (-normal_data[:, np.newaxis, :] + normal_data[np.newaxis, :, :]) ** 2
    upstream = (differences.sum(axis=1) / (differences.shape[1] - 1)).mean(axis=0)
    np.testing.assert_allclose(_normal_dispersion(normal_data), upstream, rtol=1e-10, atol=1e-12)


def test_m2oe_vendor_gradients_match_finite_differences() -> None:
    """The hand-written backpropagation matches central finite differences."""
    rng = np.random.default_rng(0)
    model = _SharedChoiceMaskingModel(
        n_features=4,
        normal_dist=_normal_dispersion(rng.normal(size=(25, 4))),
        loss_weights=(1.0, 1.0, 0.5),
        seed=3,
    )
    outliers = rng.normal(size=(7, 4)) + 0.7
    references = rng.normal(size=(7, 4))
    _loss, grads = model.loss_and_grads(outliers, references)
    epsilon = 1e-6
    worst = 0.0
    for parameter_index, parameter in enumerate(model.parameters()):
        iterator = np.nditer(parameter, flags=["multi_index"])
        for _ in iterator:
            position = iterator.multi_index
            original = parameter[position]
            parameter[position] = original + epsilon
            loss_plus, _ = model.loss_and_grads(outliers, references)
            parameter[position] = original - epsilon
            loss_minus, _ = model.loss_and_grads(outliers, references)
            parameter[position] = original
            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic = grads[parameter_index][position]
            worst = max(worst, abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8))
    assert worst < 1e-5


def test_m2oe_group_is_deterministic() -> None:
    """Repeated calls with the same seed return identical ranking scores."""
    rng = np.random.default_rng(11)
    columns = pd.Index([f"f{i}" for i in range(12)])
    frame = pd.DataFrame(rng.normal(size=(120, 12)), columns=columns)
    groups = pd.Series(["AB"] * 12 + ["NO"] * 108)
    kwargs: _M2oeGroupRankKwargs = {"epochs": 3, "n_neighbors": 5, "batch_size": 16, "max_group_rows": 8, "seed": 7}
    first = rank_m2oe_group(frame, groups, ("AB",), ("NO",), **kwargs)
    second = rank_m2oe_group(frame, groups, ("AB",), ("NO",), **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_m2oe_group_contract_and_drift_ranking_on_s1() -> None:
    """M2OE-Group returns protocol-valid rankings and detects the S1 drift."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=240, seed=12)
    settings = _modern_settings(("M2OE-Group",))
    result = run_experiment_bundle(bundle, settings=settings)
    m2oe = result.rankings.loc[result.rankings["method"] == "M2OE-Group"]
    assert set(m2oe["view"]) == {"process"}
    assert "EQP" not in m2oe["feature"].tolist()
    assert set(m2oe["equipment"]) == {"EQP_A"}
    assert m2oe["rank"].min() == 1
    assert m2oe.groupby("equipment")["rank"].max().tolist() == [len(bundle.process_features)]
    assert set(result.metrics["method"]) == {"M2OE-Group"}
    ranks = m2oe.set_index("feature")["rank"]
    assert ranks["feat_special_dist"] < ranks[MEDIAN_FEATURE]
    assert m2oe.loc[m2oe["feature"] == "feat_special_dist", "score"].iloc[0] > 0.5


def test_xpe_contract_and_drift_ranking_on_s1() -> None:
    """XPE returns protocol-valid rankings and attributes the S1 drift highly."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=240, seed=12)
    settings = _modern_settings(("XPE",))
    result = run_experiment_bundle(bundle, settings=settings)
    xpe = result.rankings.loc[result.rankings["method"] == "XPE"]
    assert set(xpe["view"]) == {"process"}
    assert "EQP" not in xpe["feature"].tolist()
    assert set(xpe["equipment"]) == {"EQP_A"}
    assert xpe["rank"].min() == 1
    assert xpe.groupby("equipment")["rank"].max().tolist() == [len(bundle.process_features)]
    assert set(result.metrics["method"]) == {"XPE"}
    root_rank = int(xpe.loc[xpe["feature"] == "feat_special_dist", "rank"].iloc[0])
    assert root_rank <= TOP_QUARTILE


def test_modern_baselines_one_vs_rest_on_s3() -> None:
    """Multi-abnormal scenarios produce independent per-equipment rankings."""
    bundle = generate_dataset_bundle(scenario="S3", n_samples=300, seed=5)
    settings = _modern_settings(("M2OE-Group", "XPE"))
    result = run_experiment_bundle(bundle, settings=settings)
    for method in ("M2OE-Group", "XPE"):
        method_rankings = result.rankings.loc[result.rankings["method"] == method]
        assert set(method_rankings["equipment"]) == {"EQP_A", "EQP_B", "EQP_C"}
        per_equipment = method_rankings.groupby("equipment")["rank"].agg(["min", "max", "count"])
        assert (per_equipment["min"] == 1).all()
        assert (per_equipment["max"] == len(bundle.process_features)).all()
        assert (per_equipment["count"] == len(bundle.process_features)).all()
    m2oe_scores = result.rankings.loc[result.rankings["method"] == "M2OE-Group"].pivot_table(
        index="feature", columns="equipment", values="score"
    )
    # Equipment-specific faults must yield equipment-specific scores.
    assert (m2oe_scores["EQP_A"] - m2oe_scores["EQP_B"]).abs().max() > 0


def test_xpe_stratified_subsample_is_deterministic_and_label_preserving() -> None:
    """The row-cap subsample keeps label proportions, stays sorted and deterministic."""
    rng = np.random.default_rng(3)
    positions = np.arange(100)
    labels = np.zeros(100, dtype=int)
    labels[rng.choice(100, size=30, replace=False)] = 1
    cap = 25
    first = _stratified_subsample(positions, labels, cap=cap, rng=np.random.default_rng(7))
    second = _stratified_subsample(positions, labels, cap=cap, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert len(first) == cap
    assert (np.diff(first) > 0).all()
    selected_positive_share = labels[first].mean()
    expected_share = labels.mean()
    assert abs(selected_positive_share - expected_share) <= 1 / cap
    # Non-binding caps return the input untouched.
    np.testing.assert_array_equal(
        _stratified_subsample(positions, labels, cap=100, rng=np.random.default_rng(7)), positions
    )
    with pytest.raises(ValueError, match="row cap must be at least 1"):
        _stratified_subsample(positions, labels, cap=0, rng=np.random.default_rng(7))


def test_xpe_row_caps_deterministic_and_recorded_in_manifest() -> None:
    """Capped XPE runs are deterministic and the cap is recorded in the manifest."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=240, seed=12)
    settings = ExperimentSettings(methods=("XPE",), cv_splits=3, xpe_permutations=2, xpe_max_rows=(10, 50))
    first = run_experiment_bundle(bundle, settings=settings)
    second = run_experiment_bundle(bundle, settings=settings)
    pd.testing.assert_frame_equal(
        first.rankings.loc[first.rankings["method"] == "XPE"].reset_index(drop=True),
        second.rankings.loc[second.rankings["method"] == "XPE"].reset_index(drop=True),
    )
    assert first.manifest["settings"]["xpe_max_rows"] == (10, 50)
    assert first.manifest["settings"]["xpe_seed"] == 42
    # The durable JSON manifest records the cap as a list.
    assert json.loads(json.dumps(first.manifest["settings"]))["xpe_max_rows"] == [10, 50]


def test_xpe_non_binding_row_caps_match_uncapped_exactly() -> None:
    """A cap larger than the available row count changes nothing."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=200, seed=4)
    uncapped = run_experiment_bundle(
        bundle, settings=ExperimentSettings(methods=("XPE",), cv_splits=3, xpe_permutations=2)
    )
    capped = run_experiment_bundle(
        bundle,
        settings=ExperimentSettings(methods=("XPE",), cv_splits=3, xpe_permutations=2, xpe_max_rows=(1000, 1000)),
    )
    pd.testing.assert_frame_equal(
        uncapped.rankings.loc[uncapped.rankings["method"] == "XPE"].reset_index(drop=True),
        capped.rankings.loc[capped.rankings["method"] == "XPE"].reset_index(drop=True),
    )


def test_xpe_row_caps_bound_the_transport_problem_size() -> None:
    """Caps reduce the EMD cost-matrix size to the capped target dimension."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=240, seed=9)
    seen_shapes: list[tuple[int, int]] = []

    def _spy(cost: np.ndarray) -> np.ndarray:
        seen_shapes.append(cost.shape)
        return _emd_coupling(cost)

    settings = ExperimentSettings(methods=("XPE",), cv_splits=3, xpe_permutations=2, xpe_max_rows=(10, 50))
    with patch("shap_diff_analysis.xpe._emd_coupling", _spy):
        result = run_experiment_bundle(bundle, settings=settings)
    assert seen_shapes == [(10, 10)]
    xpe = result.rankings.loc[result.rankings["method"] == "XPE"]
    assert len(xpe) == len(bundle.process_features)


def test_cli_maps_modern_baseline_row_caps() -> None:
    """CLI flags map onto the new ExperimentSettings fields."""
    args = run_matrix_script.build_parser().parse_args(
        [
            "--output",
            "/tmp/unused",
            "--method",
            "XPE",
            "--xpe-max-rows",
            "200",
            "2000",
            "--m2oe-max-group-rows",
            "64",
        ]
    )
    settings = run_matrix_script.build_settings(args)
    assert settings.xpe_max_rows == (200, 2000)
    assert settings.m2oe_max_group_rows == 64
    assert settings.methods == ("XPE",)


def _sparse_sensor_bundle(seed: int) -> DatasetBundle:
    """Small bundle whose ``sparse_sensor`` is exactly constant in the normal pool."""
    rng = np.random.default_rng(seed)
    rows = [
        pd.DataFrame(
            {
                "root_a": rng.normal(size=30),
                "noise": rng.normal(size=30),
                "sparse_sensor": np.zeros(30),
                "EQP": device,
            }
        )
        for device in ("EQP_A", "EQP_B", "EQP_C")
    ]
    data = pd.concat(rows, ignore_index=True)
    abnormal = data["EQP"] == "EQP_A"
    data.loc[abnormal, "root_a"] += 3.0
    data.loc[abnormal, "sparse_sensor"] = 1.0
    logit = -1.0 + 1.5 * data["root_a"]
    data["target"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return DatasetBundle(
        data=data,
        scenario="S1",
        seed=seed,
        group_column="EQP",
        target_column="target",
        process_features=("root_a", "noise", "sparse_sensor"),
        abnormal_devices=("EQP_A",),
        normal_devices=("EQP_B", "EQP_C"),
        root_causes={"EQP_A": ("root_a",)},
    )


def test_xpe_degenerate_reference_feature_ranks_bottom_and_keeps_full_length() -> None:
    """A feature constant in the reference still occupies the bottom ranking slot."""
    bundle = _sparse_sensor_bundle(seed=2)
    settings = ExperimentSettings(methods=("XPE",), cv_splits=3)
    result = run_experiment_bundle(bundle, settings=settings)
    xpe = result.rankings.loc[result.rankings["method"] == "XPE"]
    assert len(xpe) == len(bundle.process_features)
    assert xpe["rank"].min() == 1
    assert xpe.groupby("equipment")["rank"].max().tolist() == [len(bundle.process_features)]
    sparse = xpe.loc[xpe["feature"] == "sparse_sensor"].iloc[0]
    assert int(sparse["rank"]) == len(bundle.process_features)
    assert sparse["score"] == float("-inf")
    # The ranking machinery accepts the full-length frame.
    assert set(result.metrics["method"]) == {"XPE"}
    assert np.isfinite(result.metrics["mrr"].iloc[0])


def test_xpe_all_degenerate_reference_still_raises() -> None:
    """A reference in which every feature is constant remains an explicit error."""
    groups = pd.Series(["A"] * 6 + ["C"] * 6)
    X = pd.DataFrame(
        {
            "const_a": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "const_b": [4.0, 5.0, 6.0, 4.5, 5.5, 6.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        }
    )
    y = pd.Series([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    model = Pipeline([("estimator", LogisticRegression())]).fit(X, y)
    attribution = CrossFittedAttributions(
        shap_values=pd.DataFrame(np.zeros((12, 2)), columns=X.columns),
        oof_predictions=pd.Series(y, dtype=float),
        fold_metrics=pd.DataFrame({"fold": [0], "auc": [0.5]}),
        model=model,
        feature_names=tuple(X.columns),
        encoded_feature_names=tuple(X.columns),
        metadata={"mean_auc": 0.5},
    )
    with pytest.raises(ValueError, match="positive reference variance"):
        rank_baseline_features(
            "XPE",
            X=X,
            groups=groups,
            abnormal_devices=("A",),
            normal_devices=("C",),
            y=y,
            attribution=attribution,
            process_features=("const_a", "const_b"),
        )

"""Tests for the nonlinear-label internal-review benchmark."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.scenario import get_scenario_spec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

nonlinear: ModuleType = importlib.import_module("scripts.bdcc_internal_review_nonlinear")
GADS_METHOD = nonlinear.GADS_METHOD
GAM_METHOD = nonlinear.GAM_METHOD
GLM_METHOD = nonlinear.GLM_METHOD
METHODS = nonlinear.METHODS
SCENARIOS = nonlinear.SCENARIOS
SPECIAL_FEATURES = nonlinear.SPECIAL_FEATURES
USEFUL_FEATURES = nonlinear.USEFUL_FEATURES
_paired_rows = nonlinear._paired_rows
_run_single = nonlinear._run_single
_summary_rows = nonlinear._summary_rows
bca_mean_ci = nonlinear.bca_mean_ci
generate_nonlinear_bundle = nonlinear.generate_nonlinear_bundle
nonlinear_probabilities = nonlinear.nonlinear_probabilities


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_nonlinear_bundle_keeps_original_manifest(scenario: str) -> None:
    """The nonlinear overlay must preserve the borrowed scenario contract."""
    bundle = generate_nonlinear_bundle(scenario=scenario, n_samples=600, seed=3)
    spec = get_scenario_spec(scenario)

    assert bundle.scenario == scenario
    assert bundle.abnormal_devices == spec.abnormal_devices
    assert bundle.root_causes == spec.root_causes
    assert bundle.harmless_features == spec.harmless_features
    assert set(bundle.abnormal_devices).isdisjoint(bundle.normal_devices)
    assert set(bundle.abnormal_devices) | set(bundle.normal_devices) == set(bundle.data["EQP"])
    assert len(bundle.process_features) == 100
    assert set(bundle.data["target"].unique()) <= {0, 1}
    assert bundle.metadata["source_dataset"] == "Dataset A"
    assert bundle.metadata["label_generator"] == "bdcc_internal_review_nonlinear_rsm_v2"


def test_s2_raw_marginal_is_unchanged_but_conditional_changes() -> None:
    """S2 changes only P(Y|EQP); the special_imp marginal is still the same sample."""
    bundle = generate_nonlinear_bundle(scenario="S2", n_samples=5_000, seed=0)
    groups = bundle.data["EQP"].astype(str).to_numpy()
    feature = "feat_special_imp"
    abnormal = groups == "EQP_B"
    normal = np.isin(groups, list(bundle.normal_devices))

    mean_difference = abs(float(bundle.data.loc[abnormal, feature].mean() - bundle.data.loc[normal, feature].mean()))
    assert mean_difference < 0.3, "S2 must not inject a raw location shift"

    _, probability_on, overrides = nonlinear_probabilities(bundle, apply_mechanism=True)
    _, probability_off, _ = nonlinear_probabilities(bundle, apply_mechanism=False)
    assert overrides == {"EQP_B": (feature,)}
    assert np.mean(np.abs(probability_on[abnormal] - probability_off[abnormal])) > 0.02
    assert np.array_equal(probability_on[~abnormal], probability_off[~abnormal])


@pytest.mark.parametrize("scenario", ["S4-main", "S4-null"])
def test_harmless_features_never_enter_the_label(scenario: str) -> None:
    """Changing harmless drift columns must leave the true logit bit-identical."""
    bundle = generate_nonlinear_bundle(scenario=scenario, n_samples=800, seed=5)
    eta_before, probability_before, _ = nonlinear_probabilities(bundle)
    perturbed = bundle.data.copy(deep=True)
    perturbed.loc[:, list(bundle.harmless_features)] += 100.0
    perturbed_bundle = DatasetBundle(
        data=perturbed,
        scenario=bundle.scenario,
        seed=bundle.seed,
        group_column=bundle.group_column,
        target_column=bundle.target_column,
        process_features=bundle.process_features,
        abnormal_devices=bundle.abnormal_devices,
        normal_devices=bundle.normal_devices,
        root_causes=bundle.root_causes,
        harmless_features=bundle.harmless_features,
        metadata=bundle.metadata,
    )
    eta_after, probability_after, _ = nonlinear_probabilities(perturbed_bundle)
    assert np.array_equal(eta_before, eta_after)
    assert np.array_equal(probability_before, probability_after)


def test_bilinear_mechanism_delta_has_zero_population_linear_slope() -> None:
    """The RSM coupling term is uncorrelated with x_s but still load-dependent."""
    rng = np.random.default_rng(2026)
    n = 200_000
    values = rng.normal(0.0, 1.0, size=n)
    loads = rng.normal(0.0, 1.0, size=n)
    delta = nonlinear._special_mechanism_delta(values, loads, sigma=1.0)

    linear_slope = float(np.cov(values, delta, ddof=0)[0, 1] / np.var(values))
    assert abs(linear_slope) < 0.02, "bilinear coupling must have ~zero population linear slope in x_s"
    assert abs(float(np.corrcoef(values, delta)[0, 1])) < 0.02
    assert float(delta.std()) > 0.5, "the coupling term must still vary across rows"

    load_dependent = nonlinear._special_mechanism_delta(values, 2.0 * loads, sigma=1.0)
    assert np.allclose(load_dependent, 2.0 * delta)


def test_bilinear_delta_changes_conditional_probability_on_abnormal_devices() -> None:
    """The fault changes P(Y|EQP) without a raw marginal shift of the special feature."""
    bundle = generate_nonlinear_bundle(scenario="S2", n_samples=5_000, seed=0)
    groups = bundle.data["EQP"].astype(str).to_numpy()
    abnormal = groups == "EQP_B"
    feature = "feat_special_imp"

    values = bundle.data.loc[abnormal, feature].to_numpy(dtype=float)
    loads = bundle.data.loc[abnormal, nonlinear.COUPLED_LOAD_FEATURE].to_numpy(dtype=float)
    expected_delta = nonlinear.FAULT_COUPLING_WEIGHT * loads * (values / nonlinear.SPECIAL_SIGMA[feature])
    delta = nonlinear._special_mechanism_delta(values, loads, sigma=nonlinear.SPECIAL_SIGMA[feature])
    assert np.array_equal(delta, expected_delta)

    _, probability_on, _ = nonlinear_probabilities(bundle, apply_mechanism=True)
    _, probability_off, _ = nonlinear_probabilities(bundle, apply_mechanism=False)
    change = probability_on[abnormal] - probability_off[abnormal]
    assert float(np.mean(np.abs(change))) > 0.02
    assert float(np.std(change)) > 0.0, "the fault must change the conditional probability shape, not only its level"
    assert np.array_equal(probability_on[~abnormal], probability_off[~abnormal])


def test_s3_uses_only_the_designated_normal_pool() -> None:
    """S3 faults are device-local and other abnormal devices are never references."""
    bundle = generate_nonlinear_bundle(scenario="S3", n_samples=800, seed=7)
    groups = bundle.data["EQP"].astype(str).to_numpy()
    assert set(bundle.normal_devices).isdisjoint({"EQP_A", "EQP_B", "EQP_C"})
    assert set(bundle.normal_devices) == set(groups) - {"EQP_A", "EQP_B", "EQP_C"}

    _, probability_on, overrides = nonlinear_probabilities(bundle, apply_mechanism=True)
    _, probability_off, _ = nonlinear_probabilities(bundle, apply_mechanism=False)
    assert overrides == {"EQP_B": ("feat_special_imp",), "EQP_C": ("feat_special_mix",)}
    changed = np.abs(probability_on - probability_off) > 0
    assert set(groups[changed]) == {"EQP_B", "EQP_C"}
    assert np.array_equal(probability_on[groups == "EQP_A"], probability_off[groups == "EQP_A"])

    for device in bundle.abnormal_devices:
        reference = bundle.normal_devices
        assert device not in reference
        assert set(reference) == set(bundle.normal_devices)


def test_nonlinear_logit_is_not_reproduced_by_a_linear_glm() -> None:
    """A best-fit linear projection leaves a large nonlinear residual."""
    bundle = generate_nonlinear_bundle(scenario="S1", n_samples=5_000, seed=11)
    eta, _, _ = nonlinear_probabilities(bundle)
    features = [*USEFUL_FEATURES, *SPECIAL_FEATURES]
    design = bundle.data.loc[:, features].to_numpy(dtype=float)
    model = LinearRegression().fit(design, eta)
    r_squared = float(model.score(design, eta))
    mean_abs_residual = float(np.mean(np.abs(eta - model.predict(design))))
    assert r_squared < 0.95
    assert mean_abs_residual > 0.2


def test_generation_is_reproducible() -> None:
    """The same scenario and seed must reproduce the exact label."""
    first = generate_nonlinear_bundle(scenario="S1", n_samples=700, seed=13)
    second = generate_nonlinear_bundle(scenario="S1", n_samples=700, seed=13)
    assert first.data.equals(second.data)
    assert first.manifest() == second.manifest()


def test_bca_mean_ci_reports_degenerate_and_empty_cases() -> None:
    """BCa must bound the mean and never turn an empty sample into zero."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = bca_mean_ci(values, n_resamples=500, seed=1)
    assert result["n"] == 5
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    constant = bca_mean_ci([0.7, 0.7, 0.7], n_resamples=500, seed=1)
    assert constant["ci_low"] == constant["ci_high"] == constant["mean"]
    assert constant["ci_low"] == pytest.approx(0.7)

    empty = bca_mean_ci([], n_resamples=500, seed=1)
    assert empty["n"] == 0
    assert all(np.isnan(empty[key]) for key in ("mean", "std", "ci_low", "ci_high"))


def test_bca_uses_midrank_ties_for_discrete_metrics() -> None:
    """The bias correction must count bootstrap ties at the mean at half weight."""
    values = np.array([0.0, 0.0, 1.0])
    n_resamples = 2_000
    result = bca_mean_ci(values, n_resamples=n_resamples, seed=7)
    rng = np.random.default_rng(7)
    boot = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    below = int(np.count_nonzero(boot < values.mean()))
    equal = int(np.count_nonzero(boot == values.mean()))
    expected_z0 = float(norm.ppf((below + 0.5 * equal) / n_resamples))
    assert equal > 0
    assert result["z0"] == pytest.approx(expected_z0)
    assert expected_z0 > float(norm.ppf(below / n_resamples))
    assert np.isfinite([result["ci_low"], result["ci_high"]]).all()


def _complete_metric_rows(seeds: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for seed in seeds:
                has_root = scenario != "S4-null"
                has_harmless = scenario in {"S4-main", "S4-null"}
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "mrr": 0.4 + 0.01 * seed if has_root else np.nan,
                        "hr_at_1": 1.0 if has_root else np.nan,
                        "hr_at_5": 1.0 if has_root else np.nan,
                        "far_at_1": 0.1 + 0.01 * seed if has_harmless else np.nan,
                        "far_at_5": 0.2 if has_harmless else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def test_complete_summary_rejects_missing_duplicate_and_nonfinite_cells() -> None:
    """Complete mode must fail loudly instead of silently dropping seeds or NaNs."""
    metrics = _complete_metric_rows()
    summary = _summary_rows(
        metrics,
        confidence=0.95,
        n_resamples=500,
        seed=42,
        expected_seeds=(0, 1),
        require_complete=True,
    )
    assert len(summary) == len(SCENARIOS) * len(METHODS) * len(("mrr", "hr_at_1", "hr_at_5", "far_at_1", "far_at_5"))
    paired = _paired_rows(
        metrics,
        confidence=0.95,
        n_resamples=500,
        seed=42,
        expected_seeds=(0, 1),
        require_complete=True,
    )
    assert bool(paired.loc[paired["applicable"], "n_pairs"].eq(2).all())

    drop_mask = (metrics["scenario"] == "S1") & (metrics["method"] == GADS_METHOD) & (metrics["seed"] == 0)
    missing = metrics.loc[~drop_mask].reset_index(drop=True)
    with pytest.raises(ValueError, match="seed set mismatch"):
        _summary_rows(
            missing,
            confidence=0.95,
            n_resamples=500,
            seed=42,
            expected_seeds=(0, 1),
            require_complete=True,
        )

    duplicated = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate seed"):
        _summary_rows(
            duplicated,
            confidence=0.95,
            n_resamples=500,
            seed=42,
            expected_seeds=(0, 1),
            require_complete=True,
        )

    nonfinite = metrics.copy(deep=True)
    nonfinite.loc[(nonfinite["scenario"] == "S2") & (nonfinite["seed"] == 0), "mrr"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        _summary_rows(
            nonfinite,
            confidence=0.95,
            n_resamples=500,
            seed=42,
            expected_seeds=(0, 1),
            require_complete=True,
        )


def test_summary_and_paired_rows_respect_undefined_metrics() -> None:
    """Empty manifests produce applicable=False and n=0, not zero performance."""
    rows = []
    for seed in range(4):
        for method in METHODS:
            rows.append(
                {
                    "scenario": "S1",
                    "seed": seed,
                    "method": method,
                    "mrr": 0.5 + 0.01 * seed,
                    "hr_at_1": 1.0,
                    "hr_at_5": 1.0,
                    "far_at_1": np.nan,
                    "far_at_5": np.nan,
                }
            )
            rows.append(
                {
                    "scenario": "S4-null",
                    "seed": seed,
                    "method": method,
                    "mrr": np.nan,
                    "hr_at_1": np.nan,
                    "hr_at_5": np.nan,
                    "far_at_1": 0.1 + 0.01 * seed,
                    "far_at_5": 0.2,
                }
            )
    metrics = pd.DataFrame(rows)
    summary = _summary_rows(metrics, confidence=0.95, n_resamples=500, seed=42)
    s4_mrr = summary.loc[(summary["scenario"] == "S4-null") & (summary["metric"] == "mrr")]
    assert bool(s4_mrr["applicable"].eq(False).all())
    assert bool(s4_mrr["n"].eq(0).all())
    assert bool(s4_mrr["mean"].isna().all())
    s1_far = summary.loc[(summary["scenario"] == "S1") & (summary["metric"] == "far_at_1")]
    assert bool(s1_far["applicable"].eq(False).all())
    assert bool(s1_far["n"].eq(0).all())

    paired = _paired_rows(metrics, confidence=0.95, n_resamples=500, seed=42)
    s4_mrr_pair = paired.loc[(paired["scenario"] == "S4-null") & (paired["metric"] == "mrr")]
    assert bool(s4_mrr_pair["n_pairs"].eq(0).all())
    assert bool(s4_mrr_pair["mean_difference"].isna().all())
    s4_far_pair = paired.loc[(paired["scenario"] == "S4-null") & (paired["metric"] == "far_at_1")]
    assert bool(s4_far_pair["applicable"].all())
    assert bool(s4_far_pair["n_pairs"].eq(4).all())


def _pooled_metric_rows(seeds: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    base = {GADS_METHOD: 0.80, GLM_METHOD: 0.40, GAM_METHOD: 0.20}
    rows = []
    for seed in seeds:
        for method in METHODS:
            for offset, scenario in enumerate(nonlinear.POOLED_SCENARIOS):
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "mrr": base[method] + 0.10 * offset + 0.01 * seed,
                        "hr_at_1": 0.40 + 0.05 * offset,
                        "hr_at_5": 0.80 + 0.05 * offset,
                    }
                )
    return pd.DataFrame(rows)


def test_pooled_s1_s4_main_uses_equal_weight_scenario_mean() -> None:
    """Pooled cells must average S1/S2/S3/S4-main equally inside each seed."""
    metrics = _pooled_metric_rows()
    pooled = nonlinear._pooled_per_seed(metrics, expected_seeds=(0, 1))
    assert len(pooled) == len(METHODS) * 2

    gads_seed0 = pooled.loc[(pooled["method"] == GADS_METHOD) & (pooled["seed"] == 0), "mrr"].item()
    assert gads_seed0 == pytest.approx(0.80 + 0.10 * 1.5)
    glm_seed1 = pooled.loc[(pooled["method"] == GLM_METHOD) & (pooled["seed"] == 1), "mrr"].item()
    assert glm_seed1 == pytest.approx(0.40 + 0.10 * 1.5 + 0.01)

    summary = nonlinear._pooled_summary_rows(pooled, confidence=0.95, n_resamples=500, seed=42, expected_seeds=(0, 1))
    assert set(summary["scenario_group"]) == {nonlinear.POOLED_LABEL}
    assert len(summary) == len(METHODS) * len(nonlinear.POOLED_METRICS)
    gads_mrr = summary.loc[(summary["method"] == GADS_METHOD) & (summary["metric"] == "mrr")].iloc[0]
    assert gads_mrr["n"] == 2
    assert gads_mrr["mean"] == pytest.approx(0.955)
    assert gads_mrr["applicable"]

    paired = nonlinear._pooled_paired_rows(pooled, confidence=0.95, n_resamples=500, seed=42, expected_seeds=(0, 1))
    glm_diff = paired.loc[(paired["metric"] == "mrr") & (paired["competitor"] == GLM_METHOD)].iloc[0]
    assert glm_diff["n_pairs"] == 2
    assert glm_diff["mean_difference"] == pytest.approx(0.40)
    assert glm_diff["ci_low"] <= glm_diff["mean_difference"] <= glm_diff["ci_high"]


def test_pooled_rejects_partial_scenario_grid() -> None:
    """Silently dropping one pooled scenario would reweight the average."""
    metrics = _pooled_metric_rows()
    drop = (metrics["scenario"] == "S3") & (metrics["seed"] == 0) & (metrics["method"] == GADS_METHOD)
    with pytest.raises(ValueError, match="must contain one row per pooled scenario"):
        nonlinear._pooled_per_seed(metrics.loc[~drop], expected_seeds=(0, 1))


def test_one_run_executes_all_methods(tmp_path: Path) -> None:
    """A small end-to-end run must produce complete rankings for all three methods."""
    settings = {
        "output": str(tmp_path),
        "n_samples": 800,
        "n_devices": 10,
        "noise_level": 0.2,
        "distribution_shift": 3.0,
        "mechanism_scale": 1.0,
        "harmless_drift": 3.5,
        "model_seed": 42,
        "cv_seed": 42,
        "cv_splits": 5,
        "overwrite": True,
        "save_rankings": True,
    }
    result = _run_single("S2", 0, settings)
    assert result["status"] == "ok"
    run_dir = Path(result["run_dir"])
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert set(metrics["method"]) == set(METHODS)
    assert len(metrics) == len(METHODS)
    assert cast(bool, metrics["mrr"].notna().all())
    assert metrics["hr_at_1"].between(0.0, 1.0).all()
    rankings = pd.read_parquet(run_dir / "rankings.parquet")
    for method in METHODS:
        subset = rankings.loc[(rankings["method"] == method) & (rankings["equipment"] == "EQP_B")]
        assert len(subset) == 100
        assert sorted(subset["rank"].tolist()) == list(range(1, 101))
        assert np.isfinite(subset["score"].to_numpy(dtype=float)).all()
    assert {GADS_METHOD, GAM_METHOD, GLM_METHOD} == set(METHODS)

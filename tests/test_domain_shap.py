"""Focused tests for Domain-SHAP baseline protocol."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from shap_diff_analysis.baselines import rank_domain_shap
from shap_diff_analysis.experiment_runner import ExperimentSettings, run_experiment_bundle
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.modeling import CrossFittedAttributions


def _fake_attribution(columns: list[str]) -> CrossFittedAttributions:
    shap_values = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0] for name in columns})
    return CrossFittedAttributions(
        shap_values=shap_values,
        oof_predictions=pd.Series([0, 1, 0, 1]),
        fold_metrics=pd.DataFrame({"fold": [0], "auc": [1.0]}),
        model=object(),  # type: ignore[arg-type]
        feature_names=tuple(columns),
        encoded_feature_names=tuple(columns),
        metadata={"mean_auc": 1.0},
    )


def test_domain_shap_classifier_excludes_eqp_by_default() -> None:
    """Primary protocol must not train the domain classifier on EQP."""
    groups = pd.Series(["A", "A", "B", "B", "C", "C"], name="EQP")
    X = pd.DataFrame(
        {
            "root_a": [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
            "noise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "EQP": [3.0, 3.0, 4.0, 4.0, 0.0, 0.0],
        }
    )
    seen_columns: list[tuple[str, ...]] = []

    def _capture_fit(X_domain: pd.DataFrame, y_domain: pd.Series, **kwargs: object) -> CrossFittedAttributions:
        seen_columns.append(tuple(X_domain.columns))
        return _fake_attribution(list(X_domain.columns))

    with patch("shap_diff_analysis.baselines.fit_cross_fitted_model", side_effect=_capture_fit):
        rankings = rank_domain_shap(
            X,
            groups,
            abnormal_devices=("A", "B"),
            normal_devices=("C",),
            process_features=("root_a", "noise"),
            view="process",
            cv_splits=2,
        )

    assert seen_columns
    assert all("EQP" not in columns for columns in seen_columns)
    assert set(rankings["equipment"]) == {"A", "B"}
    assert set(rankings["feature"]) == {"root_a", "noise"}
    assert set(rankings["view"]) == {"process"}
    assert set(rankings["method"]) == {"Domain-SHAP"}
    assert rankings["rank"].min() == 1
    assert rankings.groupby("equipment")["rank"].max().tolist() == [2, 2]


def test_domain_shap_leakage_control_includes_eqp_when_requested() -> None:
    """Optional leakage mode may include EQP in the domain classifier."""
    groups = pd.Series(["A", "A", "C", "C"], name="EQP")
    X = pd.DataFrame(
        {
            "root_a": [1.0, 1.0, 0.0, 0.0],
            "EQP": [3.0, 3.0, 0.0, 0.0],
        }
    )
    seen_columns: list[tuple[str, ...]] = []

    def _capture_fit(X_domain: pd.DataFrame, y_domain: pd.Series, **kwargs: object) -> CrossFittedAttributions:
        seen_columns.append(tuple(X_domain.columns))
        return _fake_attribution(list(X_domain.columns))

    with patch("shap_diff_analysis.baselines.fit_cross_fitted_model", side_effect=_capture_fit):
        rankings = rank_domain_shap(
            X,
            groups,
            abnormal_devices=("A",),
            normal_devices=("C",),
            process_features=("root_a",),
            view="all",
            include_eqp_in_classifier=True,
            cv_splits=2,
        )

    assert seen_columns == [("root_a", "EQP")]
    assert set(rankings["feature"]) == {"root_a", "EQP"}


def test_experiment_runner_domain_shap_keeps_process_ranking_without_eqp() -> None:
    """End-to-end default run keeps Domain-SHAP rankings on process features only."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=12)
    settings = ExperimentSettings(methods=("Domain-SHAP",), cv_splits=3)
    result = run_experiment_bundle(bundle, settings=settings)
    domain = result.rankings.loc[result.rankings["method"] == "Domain-SHAP"]
    assert not domain.empty
    assert set(domain["view"]) == {"process"}
    assert "EQP" not in domain["feature"].tolist()
    assert domain["rank"].min() == 1
    assert domain.groupby("equipment")["rank"].max().ge(1).all()


def test_experiment_runner_all_features_mode_allows_eqp_leakage_control() -> None:
    """All-features EQP mode enables the optional Domain-SHAP leakage control."""
    bundle = generate_dataset_bundle(scenario="S1", n_samples=160, seed=13)
    settings = ExperimentSettings(
        methods=("Domain-SHAP",),
        cv_splits=3,
        eqp_mode="all_features",
    )
    result = run_experiment_bundle(bundle, settings=settings)
    domain = result.rankings.loc[result.rankings["method"] == "Domain-SHAP"]
    assert set(domain["view"]) == {"all"}
    assert "EQP" in domain["feature"].tolist()


def test_domain_shap_rejects_all_view_without_eqp_in_classifier() -> None:
    """Ranking EQP requires the optional leakage-control classifier mode."""
    groups = pd.Series(["A", "A", "C", "C"], name="EQP")
    X = pd.DataFrame({"root_a": [1.0, 1.0, 0.0, 0.0], "EQP": [3.0, 3.0, 0.0, 0.0]})
    with (
        patch(
            "shap_diff_analysis.baselines.fit_cross_fitted_model",
            return_value=_fake_attribution(["root_a"]),
        ),
        pytest.raises(KeyError, match="EQP"),
    ):
        rank_domain_shap(
            X,
            groups,
            abnormal_devices=("A",),
            normal_devices=("C",),
            process_features=("root_a",),
            view="all",
            include_eqp_in_classifier=False,
            cv_splits=2,
        )

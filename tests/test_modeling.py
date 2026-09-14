import numpy as np
import pandas as pd
import pytest

from shap_diff_analysis.modeling import (
    _aggregate_shap,
    _build_encoded_to_original_mapping,
    _make_preprocessor,
    fit_cross_fitted_model,
)


def _fit_preprocessor(X: pd.DataFrame):
    preprocessor = _make_preprocessor(X)
    preprocessor.fit(X)
    return preprocessor


def test_aggregate_shap_keeps_missing_indicator_separate_from_base_feature() -> None:
    """feat_001 and feat_001_missing must not share aggregated SHAP mass."""
    X = pd.DataFrame(
        {
            "feat_001": [1.0, 2.0, 3.0],
            "feat_001_missing": [0.0, 1.0, 0.0],
        }
    )
    preprocessor = _fit_preprocessor(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    shap_encoded = np.zeros((len(X), len(encoded_names)))
    name_to_idx = {name: idx for idx, name in enumerate(encoded_names)}
    shap_encoded[:, name_to_idx["feat_001"]] = [10.0, 20.0, 30.0]
    shap_encoded[:, name_to_idx["feat_001_missing"]] = [100.0, 200.0, 300.0]

    result = _aggregate_shap(shap_encoded, preprocessor, list(X.columns))

    assert result["feat_001"].tolist() == [10.0, 20.0, 30.0]
    assert result["feat_001_missing"].tolist() == [100.0, 200.0, 300.0]


def test_aggregate_shap_sums_eqp_one_hot_columns() -> None:
    """One-hot EQP attributions aggregate back to a single EQP column."""
    X = pd.DataFrame({"process": [1.0, 2.0, 3.0], "EQP": ["A", "B", "A"]})
    preprocessor = _fit_preprocessor(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    shap_encoded = np.zeros((len(X), len(encoded_names)))
    name_to_idx = {name: idx for idx, name in enumerate(encoded_names)}
    shap_encoded[:, name_to_idx["process"]] = [0.1, 0.2, 0.3]
    shap_encoded[:, name_to_idx["EQP_A"]] = [1.5, 0.0, 2.5]
    shap_encoded[:, name_to_idx["EQP_B"]] = [0.0, 2.5, 0.0]

    result = _aggregate_shap(shap_encoded, preprocessor, list(X.columns))

    assert result["process"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert result["EQP"].tolist() == pytest.approx([1.5, 2.5, 2.5])


def test_build_encoded_mapping_covers_every_encoded_column_exactly_once() -> None:
    """Every encoded column maps once; no encoded columns are dropped or duplicated."""
    X = pd.DataFrame(
        {
            "feat_001": [1.0, 2.0],
            "feat_001_missing": [0.0, 1.0],
            "EQP": ["A", "B"],
        }
    )
    preprocessor = _fit_preprocessor(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    mapping = _build_encoded_to_original_mapping(preprocessor)

    assert set(mapping) == set(encoded_names)
    assert len(mapping) == len(encoded_names)
    assert mapping["feat_001"] == "feat_001"
    assert mapping["feat_001_missing"] == "feat_001_missing"
    assert mapping["EQP_A"] == "EQP"
    assert mapping["EQP_B"] == "EQP"


def test_aggregate_shap_raises_when_encoded_columns_map_outside_original_columns() -> None:
    """Encoded columns must not be silently dropped from aggregation."""
    X = pd.DataFrame({"feat_001": [1.0, 2.0], "feat_002": [3.0, 4.0]})
    preprocessor = _fit_preprocessor(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    shap_encoded = np.zeros((len(X), len(encoded_names)))

    with pytest.raises(ValueError, match="maps to unknown original column"):
        _aggregate_shap(shap_encoded, preprocessor, ["feat_001"])


def test_aggregate_shap_raises_when_original_column_lacks_encoded_features() -> None:
    """Every requested original column must have at least one encoded source."""
    X = pd.DataFrame({"feat_001": [1.0, 2.0]})
    preprocessor = _fit_preprocessor(X)
    encoded_names = list(preprocessor.get_feature_names_out())
    shap_encoded = np.zeros((len(X), len(encoded_names)))

    with pytest.raises(ValueError, match="original columns lack encoded features"):
        _aggregate_shap(shap_encoded, preprocessor, ["feat_001", "feat_002"])


def test_cross_fitted_model_preserves_eqp_as_aggregated_context() -> None:
    """OOF SHAP retains a single aggregated attribution column for EQP."""
    rng = np.random.default_rng(7)
    n = 60
    X = pd.DataFrame(
        {
            "process": rng.normal(size=n),
            "EQP": np.where(np.arange(n) % 2 == 0, "A", "B"),
        }
    )
    y = pd.Series((X["process"] + (X["EQP"] == "B") * 0.4 + rng.normal(scale=0.4, size=n) > 0).astype(int))
    result = fit_cross_fitted_model(
        X,
        y,
        cv_splits=3,
        model_seed=11,
        cv_seed=12,
        model_params={"n_estimators": 15, "max_depth": 2},
    )
    assert result.shap_values.columns.to_list() == ["process", "EQP"]
    assert not bool(result.shap_values.isna().to_numpy().any())
    assert result.metadata["model_seed"] == 11
    assert result.metadata["cv_seed"] == 12
    assert len(result.fold_metrics) == 3


def test_lightgbm_model_type_produces_cross_fitted_attributions() -> None:
    """The lightgbm branch works with the shared model_params hook."""
    rng = np.random.default_rng(5)
    n = 60
    X = pd.DataFrame({"process": rng.normal(size=n), "EQP": np.where(np.arange(n) % 2 == 0, "A", "B")})
    y = pd.Series((X["process"] + (X["EQP"] == "B") * 0.4 + rng.normal(scale=0.4, size=n) > 0).astype(int))
    result = fit_cross_fitted_model(
        X,
        y,
        cv_splits=3,
        model_type="lightgbm",
        model_params={"min_child_weight": 5.0, "max_depth": 3, "n_estimators": 20},
    )
    assert result.metadata["model_type"] == "lightgbm"
    assert result.shap_values.shape == (n, 2)
    assert not bool(result.shap_values.isna().to_numpy().any())
    assert 0.0 < result.mean_auc <= 1.0

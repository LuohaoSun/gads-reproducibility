"""Cross-fitted predictive models and out-of-fold SHAP attributions.

The paper experiments use one global KPI model per dataset/scenario.  The
equipment identifier is deliberately kept as a contextual covariate, because
the mechanism-shift scenario is conditional on equipment.  Attribution
consumers can still restrict their ranking to ``process_features``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

ModelType = Literal["xgboost", "lightgbm"]


@dataclass(frozen=True)
class CrossFittedAttributions:
    """OOF predictions, SHAP values and reproducibility metadata."""

    shap_values: pd.DataFrame
    oof_predictions: pd.Series
    fold_metrics: pd.DataFrame
    model: Pipeline
    feature_names: tuple[str, ...]
    encoded_feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def mean_auc(self) -> float:
        """Return the mean out-of-fold AUC."""
        return float(self.metadata["mean_auc"])


def _validate_xy(X: pd.DataFrame, y: pd.Series, cv_splits: int) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if X.empty:
        raise ValueError("X must contain at least one row")
    if X.columns.has_duplicates:
        raise ValueError("X columns must be unique")
    if y.nunique(dropna=False) != 2:
        raise ValueError("binary classification requires exactly two target classes")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    class_counts = y.value_counts()
    if int(class_counts.min()) < cv_splits:
        raise ValueError("each target class must contain at least cv_splits samples")


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if not transformers:
        raise ValueError("X must contain at least one usable feature")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False).set_output(
        transform="pandas"
    )


def _make_estimator(
    model_type: ModelType,
    *,
    random_state: int,
    scale_pos_weight: float,
    model_params: dict[str, Any] | None,
) -> Any:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    if model_params:
        params.update(model_params)
    if model_type == "xgboost":
        return XGBClassifier(**params)
    if model_type == "lightgbm":
        try:
            import importlib

            lightgbm = importlib.import_module("lightgbm")
            LGBMClassifier = lightgbm.LGBMClassifier
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("model_type='lightgbm' requires the optional lightgbm dependency") from exc
        params.pop("tree_method", None)
        params.pop("eval_metric", None)
        return LGBMClassifier(verbosity=-1, **params)
    raise ValueError(f"unsupported model_type: {model_type!r}")


def _class_weight(y: pd.Series) -> float:
    counts = y.value_counts()
    return float(counts.iloc[0] / counts.iloc[1]) if len(counts) == 2 else 1.0


def _extract_binary_shap(model: Any, X_transformed: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Extract class-1/tree SHAP values with a stable 2-D shape."""
    explanation = shap.TreeExplainer(model)(X_transformed, check_additivity=False)
    values = explanation.values  # noqa: PD011
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError("expected binary SHAP output")
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] != 2:
            raise ValueError(f"unexpected SHAP class dimension: {values.shape}")
        values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"unexpected SHAP shape: {values.shape}")
    return values.astype(float, copy=False)


def _register_encoded_mapping(mapping: dict[str, str], encoded: str, original: str) -> None:
    existing = mapping.get(encoded)
    if existing is not None and existing != original:
        msg = f"encoded feature {encoded!r} maps ambiguously to {existing!r} and {original!r}"
        raise ValueError(msg)
    mapping[encoded] = original


def _map_one_hot_encoder(
    mapping: dict[str, str],
    encoder: OneHotEncoder,
    input_columns: list[str],
) -> None:
    encoded_names = list(encoder.get_feature_names_out(input_columns))
    index = 0
    for column, categories in zip(input_columns, encoder.categories_, strict=True):
        for _category in categories:
            _register_encoded_mapping(mapping, encoded_names[index], column)
            index += 1
    if index != len(encoded_names):
        raise ValueError("OneHotEncoder produced unexpected feature name count")


def _build_encoded_to_original_mapping(preprocessor: ColumnTransformer) -> dict[str, str]:
    """Derive a bijection from encoded columns to original input columns."""
    check_is_fitted(preprocessor)
    mapping: dict[str, str] = {}

    for trans_name, transformer, columns in preprocessor.transformers_:
        if trans_name == "remainder":
            if transformer == "drop":
                continue
            raise ValueError(f"unsupported remainder transformer: {transformer!r}")

        if transformer == "drop":
            continue

        input_columns = list(columns)
        if transformer == "passthrough":
            for column in input_columns:
                _register_encoded_mapping(mapping, column, column)
            continue

        if isinstance(transformer, OneHotEncoder):
            _map_one_hot_encoder(mapping, transformer, input_columns)
            continue

        if hasattr(transformer, "get_feature_names_out"):
            encoded_names = list(transformer.get_feature_names_out(input_columns))
            if len(encoded_names) != len(input_columns):
                msg = (
                    f"transformer {trans_name!r} produced {len(encoded_names)} encoded "
                    f"features for {len(input_columns)} input columns; "
                    "explicit per-column mapping is required"
                )
                raise ValueError(msg)
            for encoded, column in zip(encoded_names, input_columns, strict=True):
                _register_encoded_mapping(mapping, encoded, column)
            continue

        raise ValueError(f"unsupported transformer {trans_name!r}: {transformer!r}")

    return mapping


def _aggregate_shap(
    shap_encoded: np.ndarray,
    preprocessor: ColumnTransformer,
    original_columns: list[str],
) -> pd.DataFrame:
    """Aggregate encoded attributions back to original input columns."""
    encoded_names = list(preprocessor.get_feature_names_out())
    if shap_encoded.shape[1] != len(encoded_names):
        msg = (
            f"SHAP matrix has {shap_encoded.shape[1]} columns but preprocessor "
            f"produced {len(encoded_names)} encoded features"
        )
        raise ValueError(msg)

    encoded_to_original = _build_encoded_to_original_mapping(preprocessor)
    unmapped = set(encoded_names) - set(encoded_to_original)
    if unmapped:
        msg = f"encoded features lack original-column mapping: {sorted(unmapped)}"
        raise ValueError(msg)
    extra = set(encoded_to_original) - set(encoded_names)
    if extra:
        msg = f"mapping contains unknown encoded features: {sorted(extra)}"
        raise ValueError(msg)

    original_to_indices: dict[str, list[int]] = {column: [] for column in original_columns}
    for encoded_idx, encoded_name in enumerate(encoded_names):
        original = encoded_to_original[encoded_name]
        if original not in original_to_indices:
            msg = f"encoded feature {encoded_name!r} maps to unknown original column {original!r}"
            raise ValueError(msg)
        original_to_indices[original].append(encoded_idx)

    missing_original = [column for column, indices in original_to_indices.items() if not indices]
    if missing_original:
        msg = f"original columns lack encoded features: {missing_original}"
        raise ValueError(msg)

    output = np.zeros((shap_encoded.shape[0], len(original_columns)), dtype=float)
    for output_idx, column in enumerate(original_columns):
        output[:, output_idx] = shap_encoded[:, original_to_indices[column]].sum(axis=1)
    return pd.DataFrame(output, columns=pd.Index(original_columns))


def fit_cross_fitted_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_type: ModelType = "xgboost",
    model_seed: int = 42,
    cv_seed: int | None = None,
    cv_splits: int = 5,
    model_params: dict[str, Any] | None = None,
) -> CrossFittedAttributions:
    """Fit one final model and collect cross-fitted OOF SHAP values.

    The preprocessor and estimator are fitted independently in each fold.  A
    final pipeline fitted on all observations is returned for predictions and
    model persistence; its use does not leak into the OOF attributions.
    """
    _validate_xy(X, y, cv_splits)
    cv_seed = model_seed if cv_seed is None else cv_seed
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    shap_oof = np.full((len(X), X.shape[1]), np.nan, dtype=float)
    predictions = np.full(len(X), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    encoded_feature_names: tuple[str, ...] | None = None

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        preprocessor = _make_preprocessor(X_train)
        X_train_t = preprocessor.fit_transform(X_train)
        X_valid_t = preprocessor.transform(X_valid)
        if encoded_feature_names is None:
            encoded_feature_names = tuple(preprocessor.get_feature_names_out())
        elif tuple(preprocessor.get_feature_names_out()) != encoded_feature_names:
            raise ValueError("fold preprocessors produced inconsistent feature names")
        estimator = _make_estimator(
            model_type,
            random_state=model_seed + fold,
            scale_pos_weight=_class_weight(y_train),
            model_params=model_params,
        )
        estimator.fit(X_train_t, y_train)
        predictions[valid_idx] = estimator.predict_proba(X_valid_t)[:, 1]
        shap_encoded = _extract_binary_shap(estimator, np.asarray(X_valid_t))
        shap_oof[valid_idx] = _aggregate_shap(shap_encoded, preprocessor, list(X.columns)).to_numpy()
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "auc": float(roc_auc_score(y_valid, predictions[valid_idx])),
            }
        )

    if np.isnan(predictions).any() or np.isnan(shap_oof).any():
        raise RuntimeError("cross-fitting did not produce complete OOF predictions/attributions")

    final_preprocessor = _make_preprocessor(X)
    X_all_t = final_preprocessor.fit_transform(X)
    final_estimator = _make_estimator(
        model_type,
        random_state=model_seed,
        scale_pos_weight=_class_weight(y),
        model_params=model_params,
    )
    final_estimator.fit(X_all_t, y)
    final_model = Pipeline([("preprocessor", final_preprocessor), ("estimator", final_estimator)])
    fold_metrics = pd.DataFrame(fold_rows)
    mean_auc = float(fold_metrics["auc"].mean())
    metadata = {
        "model_type": model_type,
        "model_seed": model_seed,
        "cv_seed": cv_seed,
        "cv_splits": cv_splits,
        "mean_auc": mean_auc,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "encoded_feature_names": list(encoded_feature_names or ()),
    }
    return CrossFittedAttributions(
        shap_values=pd.DataFrame(shap_oof, index=X.index, columns=X.columns),
        oof_predictions=pd.Series(predictions, index=X.index, name="oof_prediction"),
        fold_metrics=fold_metrics,
        model=final_model,
        feature_names=tuple(X.columns),
        encoded_feature_names=tuple(encoded_feature_names or ()),
        metadata=metadata,
    )


def create_training_pipeline(
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
    *,
    model_type: ModelType = "xgboost",
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create a predictive pipeline for compatibility with the patent demo."""
    # Feature-dependent preprocessing is constructed in ``fit_cross_fitted_model``;
    # this factory is retained for callers that need a sklearn estimator and
    # therefore intentionally starts with a dynamic all-column transformer.
    del scale_pos_weight
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    estimator = _make_estimator(
        model_type,
        random_state=random_state,
        scale_pos_weight=1.0,
        model_params=model_params,
    )
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_seed: int = 42,
    cv_seed: int | None = None,
    cv_splits: int = 5,
    model_type: ModelType = "xgboost",
    model_params: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Backward-compatible wrapper returning the final model and CV metadata."""
    result = fit_cross_fitted_model(
        X,
        y,
        model_type=model_type,
        model_seed=model_seed,
        cv_seed=cv_seed,
        cv_splits=cv_splits,
        model_params=model_params,
    )
    metrics: dict[str, Any] = {
        "n_samples": len(X),
        "auc": result.mean_auc,
        "mean_auc": result.mean_auc,
        "model_seed": model_seed,
        "cv_seed": model_seed if cv_seed is None else cv_seed,
        "cv_splits": cv_splits,
    }
    for fold_idx, auc_value in enumerate(result.fold_metrics["auc"].tolist()):
        metrics[f"fold_{fold_idx + 1}_auc"] = float(auc_value)
    # Expose the OOF matrix on the final estimator for the original patent demo.
    estimator = result.model.named_steps["estimator"]
    estimator.shap_values_oof_ = result.shap_values.to_numpy()
    estimator.feature_names_in_ = np.asarray(result.feature_names)
    estimator.get_feature_names_in = lambda: np.asarray(result.feature_names)
    return result.model, metrics

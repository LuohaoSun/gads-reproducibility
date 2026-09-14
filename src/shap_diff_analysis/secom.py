"""UCI SECOM semi-synthetic Dataset B generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit

from shap_diff_analysis.experiment_types import DatasetBundle

ScenarioName = Literal["S1", "S2", "S3", "S4-main", "S4-null"]
SCENARIO_NAMES: tuple[ScenarioName, ...] = ("S1", "S2", "S3", "S4-main", "S4-null")

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/179/secom"
N_FEATURES_DOCUMENTED = 591
N_FEATURES_OBSERVED = 590
N_GROUPS = 10
GROUP_PREFIX = "EQP_"
TARGET_COLUMN = "target"
GROUP_COLUMN = "EQP"

# UCI documents MatLab-style NaN; we also accept "?" for reproducible parsing rules.
MISSING_TOKENS: tuple[str, ...] = ("?", "NaN", "nan")

SECOM_FILE_SPECS: dict[str, dict[str, str]] = {
    "secom.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
        "sha256": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    },
    "secom_labels.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
        "sha256": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
    },
    "secom.names": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names",
        "sha256": "6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59",
    },
}

DEFAULT_RAW_DIR = Path("data/raw/secom")

# Semi-synthetic GLM coefficients (aligned with Dataset A magnitudes).
ROOT_CAUSE_GLOBAL_WEIGHT = 0.8
MECHANISM_ROOT_GLOBAL_WEIGHT = 0.2
MECHANISM_LOCAL_WEIGHT = 12.0
HARMLESS_GLOBAL_WEIGHT = 0.0

# Default injected-fault strengths (in train-fold standardized units).
PHYSICAL_SHIFT_MAGNITUDE = 3.0
HARMLESS_SHIFT_MAGNITUDE = 4.0


class SecoMDataError(Exception):
    """Raised when SECOM raw data is missing or fails integrity checks."""


@dataclass(frozen=True)
class SecoMSourceManifest:
    """Download provenance for the official UCI SECOM files."""

    dataset_page: str
    files: dict[str, dict[str, str]]
    missing_tokens: tuple[str, ...]
    n_features_documented: int
    n_features_observed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest dataclass to a plain dictionary."""
        return asdict(self)


def feature_column_names(n_features: int = N_FEATURES_OBSERVED) -> tuple[str, ...]:
    """Return zero-padded SECOM feature column names."""
    return tuple(f"feat_{index:03d}" for index in range(1, n_features + 1))


def sha256_file(path: Path) -> str:
    """Compute the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_secom_files(raw_dir: Path) -> dict[str, str]:
    """Verify that all official SECOM files exist and match expected SHA256 digests."""
    digests: dict[str, str] = {}
    for filename, spec in SECOM_FILE_SPECS.items():
        path = raw_dir / filename
        if not path.is_file():
            msg = f"Missing SECOM raw file: {path}. Run scripts/download_secom.py first."
            raise SecoMDataError(msg)
        digest = sha256_file(path)
        expected = spec["sha256"]
        if digest != expected:
            msg = f"SHA256 mismatch for {path.name}: expected {expected}, got {digest}"
            raise SecoMDataError(msg)
        digests[filename] = digest
    return digests


def source_manifest() -> SecoMSourceManifest:
    """Return static UCI source metadata used by download and generation scripts."""
    return SecoMSourceManifest(
        dataset_page=UCI_DATASET_PAGE,
        files={filename: {"url": spec["url"], "sha256": spec["sha256"]} for filename, spec in SECOM_FILE_SPECS.items()},
        missing_tokens=MISSING_TOKENS,
        n_features_documented=N_FEATURES_DOCUMENTED,
        n_features_observed=N_FEATURES_OBSERVED,
    )


def _parse_uci_labels(path: Path) -> pd.Series:
    """Parse UCI SECOM label file (-1 pass, 1 fail)."""
    parsed: list[int] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw_label = int(stripped.split(maxsplit=1)[0])
        parsed.append(0 if raw_label == -1 else 1)
    return pd.Series(parsed, name="uci_label", dtype=np.int8)


def load_raw_secom(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load official SECOM raw files from disk without network access."""
    directory = raw_dir or DEFAULT_RAW_DIR
    verify_secom_files(directory)

    feature_names = feature_column_names()
    features = pd.read_csv(
        directory / "secom.data",
        sep=r"\s+",
        header=None,
        names=list(feature_names),
        na_values=list(MISSING_TOKENS),
        engine="python",
    )
    uci_labels = _parse_uci_labels(directory / "secom_labels.data")

    if len(features) != len(uci_labels):
        msg = f"Row count mismatch: features={len(features)}, labels={len(uci_labels)}"
        raise SecoMDataError(msg)
    if len(features.columns) != N_FEATURES_OBSERVED:
        msg = f"Expected {N_FEATURES_OBSERVED} feature columns, got {len(features.columns)}"
        raise SecoMDataError(msg)

    frame = features.copy()
    frame["uci_label"] = uci_labels.to_numpy()
    return frame


def assign_pseudo_eqps(n_samples: int, group_seed: int, n_groups: int = N_GROUPS) -> pd.Series:
    """Deterministically partition rows into near-equal pseudo EQP groups."""
    if n_samples < n_groups:
        msg = f"Need at least {n_groups} samples for {n_groups} groups, got {n_samples}"
        raise ValueError(msg)

    rng = np.random.default_rng(group_seed)
    group_names = np.array([f"{GROUP_PREFIX}{chr(65 + index)}" for index in range(n_groups)], dtype=object)
    permuted = rng.permutation(n_samples)
    assignments = np.empty(n_samples, dtype=object)
    start = 0
    base, remainder = divmod(n_samples, n_groups)
    for group_index in range(n_groups):
        size = base + (1 if group_index < remainder else 0)
        assignments[permuted[start : start + size]] = group_names[group_index]
        start += size
    return pd.Series(assignments, name=GROUP_COLUMN)


def _train_indices(n_samples: int, seed: int, train_fraction: float = 0.8) -> np.ndarray:
    """Pick a deterministic train fold for imputation statistics only."""
    rng = np.random.default_rng(seed + 10_007)
    permuted = rng.permutation(n_samples)
    train_size = int(n_samples * train_fraction)
    return permuted[:train_size]


def preprocess_features(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...],
    train_idx: np.ndarray,
    *,
    include_missing_indicators: bool = True,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Apply train-fold-only median imputation and optional missing indicators."""
    raw = features.loc[:, list(feature_columns)]
    missing_mask = raw.isna()
    train_medians = raw.iloc[train_idx].median(numeric_only=True)
    imputed = raw.fillna(train_medians)
    if include_missing_indicators:
        indicator_columns = tuple(f"{column}_missing" for column in feature_columns)
        indicators = missing_mask.astype(np.float64)
        indicators.columns = list(indicator_columns)
        processed = pd.concat([imputed, indicators], axis=1)
        process_features = tuple(imputed.columns) + indicator_columns
    else:
        processed = imputed.copy()
        process_features = tuple(imputed.columns)
    metadata = {
        "imputation": "train_fold_median",
        "train_indices_count": len(train_idx),
        "train_medians": {key: float(value) for key, value in train_medians.items()},
        "missing_tokens": list(MISSING_TOKENS),
        "include_missing_indicators": include_missing_indicators,
    }
    return processed, process_features, metadata


def _non_constant_columns(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    train_idx: np.ndarray,
) -> tuple[str, ...]:
    """Return feature columns with non-zero train-fold spread after imputation."""
    train = frame.iloc[train_idx]
    varying = tuple(column for column in feature_columns if float(train[column].std()) > 0.0)
    minimum = 24  # 12 global + root + mechanism + 10 harmless
    if len(varying) < minimum:
        msg = f"Need at least {minimum} non-constant features, got {len(varying)}"
        raise SecoMDataError(msg)
    return varying


def _standardize_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit location/scale on imputed train rows and standardize every process feature.

    The protocol standardizes all process features (imputed sensors and, when
    kept, missing indicators), not only the role features, so scale-sensitive
    baselines compare like-for-like units across the whole feature space.
    """
    scaled = frame.copy()
    feature_stats: dict[str, dict[str, float]] = {}
    for feature in feature_columns:
        train_values = scaled.iloc[train_idx][feature]
        loc = float(train_values.mean())
        scale = float(train_values.std())
        if scale == 0.0 or not np.isfinite(scale):
            scale = 1.0
        scaled[feature] = (scaled[feature] - loc) / scale
        feature_stats[feature] = {"loc": loc, "scale": scale}
    metadata = {
        "method": "train_fold_location_scale_all_features",
        "train_indices_count": len(train_idx),
        "features": feature_stats,
    }
    return scaled, metadata


@dataclass(frozen=True)
class FeatureRoles:
    """Ground-truth feature roles selected deterministically from ``seed``."""

    global_signals: tuple[str, ...]
    root_cause: str
    mechanism_root: str
    harmless: tuple[str, ...]


def _select_feature_roles(feature_columns: tuple[str, ...], seed: int) -> FeatureRoles:
    rng = np.random.default_rng(seed)
    ordered = list(rng.permutation(feature_columns))
    global_signals = tuple(ordered[:12])
    root_cause = ordered[12]
    mechanism_root = ordered[13]
    harmless = tuple(ordered[14:24])
    return FeatureRoles(
        global_signals=global_signals,
        root_cause=root_cause,
        mechanism_root=mechanism_root,
        harmless=harmless,
    )


def _device_names() -> tuple[str, ...]:
    return tuple(f"{GROUP_PREFIX}{chr(65 + index)}" for index in range(N_GROUPS))


def _scenario_layout(
    scenario: ScenarioName,
    *,
    physical_shift_magnitude: float = PHYSICAL_SHIFT_MAGNITUDE,
    mechanism_local_weight: float = MECHANISM_LOCAL_WEIGHT,
    harmless_shift_magnitude: float = HARMLESS_SHIFT_MAGNITUDE,
) -> dict[str, Any]:
    devices = _device_names()
    drift_device = devices[0]  # EQP_A: distribution drift
    mechanism_device = devices[1]  # EQP_B: mechanism shift
    mixed_device = devices[2]  # EQP_C: mixed (physical drift + local coefficient)
    s4_main_device = devices[3]  # EQP_D
    s4_null_device = devices[4]  # EQP_E

    if scenario == "S1":
        return {
            "abnormal_devices": (drift_device,),
            "normal_devices": tuple(device for device in devices if device != drift_device),
            "physical_shifts": {drift_device: ("root_cause", physical_shift_magnitude)},
            "mechanism_devices": {},
            "root_causes": {drift_device: ("root_cause",)},
            "harmless_features": (),
        }
    if scenario == "S2":
        return {
            "abnormal_devices": (mechanism_device,),
            "normal_devices": tuple(device for device in devices if device != mechanism_device),
            "physical_shifts": {},
            "mechanism_devices": {mechanism_device: ("mechanism_root", mechanism_local_weight)},
            "root_causes": {mechanism_device: ("mechanism_root",)},
            "harmless_features": (),
        }
    if scenario == "S3":
        abnormal = {drift_device, mechanism_device, mixed_device}
        return {
            "abnormal_devices": (drift_device, mechanism_device, mixed_device),
            "normal_devices": tuple(device for device in devices if device not in abnormal),
            "physical_shifts": {
                drift_device: ("root_cause", physical_shift_magnitude),
                mixed_device: ("root_cause", physical_shift_magnitude),
            },
            "mechanism_devices": {
                mechanism_device: ("mechanism_root", mechanism_local_weight),
                mixed_device: ("root_cause", mechanism_local_weight),
            },
            "root_causes": {
                drift_device: ("root_cause",),
                mechanism_device: ("mechanism_root",),
                mixed_device: ("root_cause",),
            },
            "harmless_features": (),
        }
    if scenario == "S4-main":
        return {
            "abnormal_devices": (s4_main_device,),
            "normal_devices": tuple(device for device in devices if device != s4_main_device),
            "physical_shifts": {s4_main_device: ("harmless_all", harmless_shift_magnitude)},
            "mechanism_devices": {s4_main_device: ("mechanism_root", mechanism_local_weight)},
            "root_causes": {s4_main_device: ("mechanism_root",)},
            "harmless_features": ("harmless_all",),
        }
    if scenario == "S4-null":
        return {
            "abnormal_devices": (s4_null_device,),
            "normal_devices": tuple(device for device in devices if device != s4_null_device),
            "physical_shifts": {s4_null_device: ("harmless_all", harmless_shift_magnitude)},
            "mechanism_devices": {},
            "root_causes": {},
            "harmless_features": ("harmless_all",),
        }
    msg = f"Unsupported scenario: {scenario}"
    raise ValueError(msg)


def _resolve_feature(
    token: str,
    roles: FeatureRoles,
) -> str | tuple[str, ...]:
    if token == "root_cause":
        return roles.root_cause
    if token == "mechanism_root":
        return roles.mechanism_root
    if token == "harmless_all":
        return roles.harmless
    msg = f"Unknown feature token: {token}"
    raise ValueError(msg)


def _apply_physical_shifts(
    frame: pd.DataFrame,
    eqp: pd.Series,
    roles: FeatureRoles,
    physical_shifts: dict[str, tuple[str, float]],
) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for device, (token, magnitude) in physical_shifts.items():
        resolved = _resolve_feature(token, roles)
        feature_list = resolved if isinstance(resolved, tuple) else (resolved,)
        mask = eqp == device
        for feature in feature_list:
            frame.loc[mask, feature] = frame.loc[mask, feature] + magnitude
        applied[device] = {"features": list(feature_list), "magnitude": magnitude}
    return applied


def _compute_logit(
    frame: pd.DataFrame,
    eqp: pd.Series,
    roles: FeatureRoles,
    seed: int,
    mechanism_devices: dict[str, tuple[str, float]],
    intercept: float = -2.5,
    noise_level: float = 0.25,
    mechanism_local_weight: float = MECHANISM_LOCAL_WEIGHT,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_samples = len(frame)
    logit = np.full(n_samples, intercept, dtype=float)
    weights: dict[str, float] = {}

    for feature in roles.global_signals:
        weight = float(rng.uniform(0.8, 1.2))
        weights[feature] = weight
        logit += weight * frame[feature].to_numpy()

    weights[roles.root_cause] = ROOT_CAUSE_GLOBAL_WEIGHT
    logit += weights[roles.root_cause] * frame[roles.root_cause].to_numpy()

    weights[roles.mechanism_root] = MECHANISM_ROOT_GLOBAL_WEIGHT
    logit += weights[roles.mechanism_root] * frame[roles.mechanism_root].to_numpy()

    for feature in roles.harmless:
        weights[feature] = HARMLESS_GLOBAL_WEIGHT

    mechanism_meta: dict[str, Any] = {}
    for device, (token, local_weight) in mechanism_devices.items():
        feature = _resolve_feature(token, roles)
        if not isinstance(feature, str):
            msg = f"Mechanism shift expects one feature, got {feature}"
            raise ValueError(msg)
        mask = eqp == device
        global_weight = weights[feature]
        logit[mask.to_numpy()] += (local_weight - global_weight) * frame.loc[mask, feature].to_numpy()
        mechanism_meta[device] = {
            "feature": feature,
            "global_weight": global_weight,
            "local_weight": local_weight,
        }

    logit += rng.normal(0.0, noise_level, size=n_samples)
    coefficient_spec = {
        "root_cause_global": ROOT_CAUSE_GLOBAL_WEIGHT,
        "mechanism_root_global": MECHANISM_ROOT_GLOBAL_WEIGHT,
        "mechanism_local_weight": mechanism_local_weight,
        "harmless_global": HARMLESS_GLOBAL_WEIGHT,
    }
    return logit, {"weights": weights, "mechanism": mechanism_meta, "coefficient_spec": coefficient_spec}


def _sample_target(logit: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    probabilities = expit(logit)
    return rng.binomial(1, probabilities).astype(np.int8)


def _validate_fault_strength(name: str, value: float) -> None:
    """Validate one injected-fault strength without silently changing it."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise TypeError(f"{name} must be a finite real number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def generate_dataset_b(
    scenario: ScenarioName,
    seed: int = 42,
    group_seed: int = 2024,
    raw_dir: Path | None = None,
    intercept: float = -2.5,
    noise_level: float = 0.25,
    *,
    physical_shift_magnitude: float = PHYSICAL_SHIFT_MAGNITUDE,
    mechanism_local_weight: float = MECHANISM_LOCAL_WEIGHT,
    harmless_shift_magnitude: float = HARMLESS_SHIFT_MAGNITUDE,
    include_missing_indicators: bool = True,
) -> DatasetBundle:
    """Build a semi-synthetic SECOM Dataset B bundle for one scenario.

    Args:
        scenario: One of ``S1``--``S4-null``.
        seed: Label/role-selection seed.
        group_seed: Pseudo-EQP partition seed.
        raw_dir: Directory with the official SECOM raw files.
        intercept: GLM intercept.
        noise_level: Standard deviation of label-generation noise.
        physical_shift_magnitude: Mean shift (standardized units) for physical
            drift injections; default preserves the historical 3.0 protocol.
        mechanism_local_weight: Local GLM coefficient on mechanism-shifted
            devices; default preserves the historical 12.0 protocol.
        harmless_shift_magnitude: Mean shift (standardized units) applied to
            harmless features in S4; default preserves the historical 4.0.
        include_missing_indicators: When True (default) the ``*_missing``
            indicator columns are kept as process features; when False they
            are excluded from the bundle entirely.
    """
    if scenario not in SCENARIO_NAMES:
        msg = f"Unsupported scenario {scenario!r}; expected one of {SCENARIO_NAMES}"
        raise ValueError(msg)
    for name, value in (
        ("physical_shift_magnitude", physical_shift_magnitude),
        ("mechanism_local_weight", mechanism_local_weight),
        ("harmless_shift_magnitude", harmless_shift_magnitude),
    ):
        _validate_fault_strength(name, value)
    if not isinstance(include_missing_indicators, bool):
        raise TypeError("include_missing_indicators must be a boolean")

    raw_frame = load_raw_secom(raw_dir)
    feature_columns = feature_column_names()
    eqp = assign_pseudo_eqps(len(raw_frame), group_seed=group_seed)
    layout = _scenario_layout(
        scenario,
        physical_shift_magnitude=physical_shift_magnitude,
        mechanism_local_weight=mechanism_local_weight,
        harmless_shift_magnitude=harmless_shift_magnitude,
    )

    train_idx = _train_indices(len(raw_frame), seed=seed)
    processed, process_features, preprocess_meta = preprocess_features(
        raw_frame,
        feature_columns,
        train_idx,
        include_missing_indicators=include_missing_indicators,
    )

    varying_columns = _non_constant_columns(processed, feature_columns, train_idx)
    roles = _select_feature_roles(varying_columns, seed=seed)
    processed, scaling_meta = _standardize_features(processed, process_features, train_idx)

    physical_meta = _apply_physical_shifts(processed, eqp, roles, layout["physical_shifts"])
    logit, glm_meta = _compute_logit(
        processed,
        eqp,
        roles,
        seed=seed,
        mechanism_devices=layout["mechanism_devices"],
        intercept=intercept,
        noise_level=noise_level,
        mechanism_local_weight=mechanism_local_weight,
    )
    target = _sample_target(logit, seed=seed)

    resolved_root_causes: dict[str, tuple[str, ...]] = {}
    for device, tokens in layout["root_causes"].items():
        resolved: list[str] = []
        for token in tokens:
            feature = _resolve_feature(token, roles)
            if isinstance(feature, tuple):
                resolved.extend(feature)
            else:
                resolved.append(feature)
        resolved_root_causes[device] = tuple(resolved)

    harmless = layout["harmless_features"]
    resolved_harmless: tuple[str, ...]
    if harmless:
        harmless_resolved = _resolve_feature(harmless[0], roles)
        resolved_harmless = harmless_resolved if isinstance(harmless_resolved, tuple) else (harmless_resolved,)
    else:
        resolved_harmless = ()

    output = pd.concat(
        [
            eqp.reset_index(drop=True),
            pd.Series(target, name=TARGET_COLUMN),
            processed.reset_index(drop=True),
        ],
        axis=1,
    )

    metadata: dict[str, Any] = {
        "dataset": "B",
        "source": source_manifest().to_dict(),
        "raw_dir": str(raw_dir or DEFAULT_RAW_DIR),
        "group_seed": group_seed,
        "generation_params": {
            "scenario": scenario,
            "seed": seed,
            "group_seed": group_seed,
            "intercept": intercept,
            "noise_level": noise_level,
            "physical_shift_magnitude": physical_shift_magnitude,
            "mechanism_local_weight": mechanism_local_weight,
            "harmless_shift_magnitude": harmless_shift_magnitude,
            "include_missing_indicators": include_missing_indicators,
        },
        "feature_roles": {
            "global_signals": list(roles.global_signals),
            "root_cause": roles.root_cause,
            "mechanism_root": roles.mechanism_root,
            "harmless": list(roles.harmless),
        },
        "preprocess": preprocess_meta,
        "scaling": scaling_meta,
        "physical_shifts": physical_meta,
        "glm": glm_meta,
        "uci_label_positive_rate": float(raw_frame["uci_label"].mean()),
        "generated_target_rate": float(target.mean()),
        "integration": {
            "bundle_type": "DatasetBundle",
            "module": "shap_diff_analysis.experiment_types",
            "group_column": GROUP_COLUMN,
            "target_column": TARGET_COLUMN,
            "notes": "Semi-synthetic labels are generated locally; uci_label is not exported.",
        },
    }

    return DatasetBundle(
        data=output,
        scenario=scenario,
        seed=seed,
        group_column=GROUP_COLUMN,
        target_column=TARGET_COLUMN,
        process_features=process_features,
        abnormal_devices=tuple(layout["abnormal_devices"]),
        normal_devices=tuple(layout["normal_devices"]),
        root_causes=resolved_root_causes,
        harmless_features=resolved_harmless,
        metadata=metadata,
    )


def write_manifest(bundle: DatasetBundle, path: Path) -> None:
    """Write a JSON manifest for a generated Dataset B bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.manifest()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

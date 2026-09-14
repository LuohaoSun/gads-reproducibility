"""Generate independent, ground-truth controlled Dataset A scenarios.

Dataset A is deliberately synthetic: it gives the paper an exact root-cause
manifest while allowing distribution drift, mechanism shift and harmless
drift to be switched on independently.  ``EQP`` remains in the returned data
because it is a contextual covariate for the KPI model and the grouping key
for one-versus-rest diagnosis.  It is not included in ``process_features``.
"""

from __future__ import annotations

from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
from scipy.special import expit

from .experiment_types import DatasetBundle
from .scenario import ScenarioSpec, get_scenario_spec, normalize_scenario

ScenarioName = Literal["S1", "S2", "S3", "S4-main", "S4-null"]

N_DEVICES = 10
N_USEFUL_FEATURES = 10
N_USELESS_FEATURES = 87
SPECIAL_FEATURES = ("feat_special_dist", "feat_special_imp", "feat_special_mix")


def _validate_generation_parameters(
    *,
    n_samples: int,
    seed: int,
    noise_level: float,
    n_devices: int,
    distribution_shift: float,
    mechanism_scale: float,
    harmless_drift: float,
) -> None:
    """Validate public generator arguments without silently changing them."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be an integer")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if isinstance(n_devices, bool) or not isinstance(n_devices, (int, np.integer)):
        raise TypeError("n_devices must be an integer")
    if n_devices < N_DEVICES:
        raise ValueError(f"n_devices must be at least {N_DEVICES} so EQP_A--EQP_J exist")
    if n_devices > 26:
        raise ValueError("n_devices cannot exceed 26 (EQP_A--EQP_Z)")
    if n_samples < n_devices:
        raise ValueError("n_samples must be at least n_devices so every equipment has observations")
    for name, value in (
        ("noise_level", noise_level),
        ("distribution_shift", distribution_shift),
        ("mechanism_scale", mechanism_scale),
        ("harmless_drift", harmless_drift),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise TypeError(f"{name} must be a finite real number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _equipment_names(n_devices: int) -> tuple[str, ...]:
    """Return stable equipment labels in alphabetical order."""
    return tuple(f"EQP_{chr(65 + index)}" for index in range(n_devices))


def _balanced_equipment_assignment(
    *, rng: np.random.Generator, n_samples: int, equipment: tuple[str, ...]
) -> np.ndarray:
    """Assign each equipment at least one sample, then randomise row order."""
    repeated = np.resize(np.asarray(equipment, dtype=object), n_samples)
    rng.shuffle(repeated)
    return repeated


def _feature_names() -> tuple[str, ...]:
    useful = tuple(f"feat_useful_{index}" for index in range(1, N_USEFUL_FEATURES + 1))
    useless = tuple(f"feat_useless_{index}" for index in range(1, N_USELESS_FEATURES + 1))
    return useful + useless + SPECIAL_FEATURES


def _as_device_mask(equipment: pd.Series, device: str) -> np.ndarray:
    """Build a boolean mask while keeping NumPy's positional semantics."""
    return equipment.to_numpy(dtype=object) == device


def _inject_distribution_shift(
    values: np.ndarray,
    *,
    equipment: pd.Series,
    devices: tuple[str, ...],
    rng: np.random.Generator,
    amount: float,
) -> np.ndarray:
    """Apply a positive mean shift to a feature on selected equipment."""
    shifted = values.copy()
    for device in devices:
        mask = _as_device_mask(equipment, device)
        if mask.any():
            shifted[mask] += rng.normal(loc=amount, scale=amount * 0.15, size=int(mask.sum()))
    return shifted


def _scenario_coefficients(
    *,
    spec: ScenarioSpec,
    equipment: tuple[str, ...],
    mechanism_scale: float,
) -> dict[str, dict[str, float]]:
    """Create per-equipment coefficients for the three special features."""
    baseline = {
        "feat_special_dist": 0.8,
        "feat_special_imp": 0.1,
        "feat_special_mix": 0.3,
    }
    root_strength = {
        "feat_special_dist": 0.8,
        "feat_special_imp": 5.0,
        "feat_special_mix": 3.5,
    }
    coefficients = {feature: dict.fromkeys(equipment, value) for feature, value in baseline.items()}
    for device, features in spec.mechanism_shift_features.items():
        for feature in features:
            if feature not in coefficients:
                raise ValueError(f"scenario references unknown mechanism feature {feature!r}")
            # ``mechanism_scale`` is a multiplier for the change from the
            # normal coefficient.  A zero value intentionally disables the
            # injected mechanism shift.
            normal = baseline[feature]
            coefficients[feature][device] = normal + (root_strength[feature] - normal) * mechanism_scale
    return coefficients


def _build_metadata(
    *,
    spec: ScenarioSpec,
    seed: int,
    n_samples: int,
    noise_level: float,
    n_devices: int,
    distribution_shift: float,
    mechanism_scale: float,
    harmless_drift: float,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build a JSON-serialisable manifest metadata payload."""
    return {
        "dataset": "Dataset A",
        "dataset_kind": "pure_synthetic",
        "scenario_description": spec.description,
        "n_samples": n_samples,
        "noise_level": noise_level,
        "n_devices": n_devices,
        "distribution_shift": distribution_shift,
        "mechanism_scale": mechanism_scale,
        "harmless_drift": harmless_drift,
        "generation_params": {
            "scenario": spec.name,
            "seed": seed,
            "n_samples": n_samples,
            "noise_level": noise_level,
            "n_devices": n_devices,
            "distribution_shift": distribution_shift,
            "mechanism_scale": mechanism_scale,
            "harmless_drift": harmless_drift,
        },
        "all_process_features": list(feature_names),
        "distribution_shift_features": {
            device: list(features) for device, features in spec.distribution_shift_features.items()
        },
        "mechanism_shift_features": {
            device: list(features) for device, features in spec.mechanism_shift_features.items()
        },
        "seed": seed,
        # The null scene intentionally has no root-cause rank metrics.
        "metrics_scope": "far_only" if spec.name == "S4-null" else "mrr_hr_far",
    }


def generate_dataset_bundle(
    *,
    scenario: str = "S1",
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
) -> DatasetBundle:
    """Generate one independent Dataset A scenario.

    Every call creates a fresh dataset and a complete ground-truth manifest.
    Scenarios do not share rows or injected devices: an experiment should call
    this function once per ``(scenario, seed)`` pair.

    Args:
        scenario: ``S1``, ``S2``, ``S3``, ``S4-main`` or ``S4-null``.  The
            underscore spellings ``S4_main``/``S4_null`` are accepted aliases.
        n_samples: Number of rows; each equipment receives at least one row.
        seed: NumPy random seed.
        noise_level: Standard deviation of label-generation noise.
        n_devices: Number of equipment labels, with A--J always present.
        distribution_shift: Mean shift for injected physical drift.
        mechanism_scale: Multiplier for injected conditional-effect changes.
        harmless_drift: Mean shift applied to S4 harmless features.

    Returns:
        A :class:`~shap_diff_analysis.experiment_types.DatasetBundle` whose
        ``data`` contains ``EQP``, process features and ``target``.

    Raises:
        TypeError, ValueError: If the scenario or any generation parameter is
            invalid.  Invalid settings are never silently corrected.
    """
    canonical_scenario = normalize_scenario(scenario)
    _validate_generation_parameters(
        n_samples=n_samples,
        seed=seed,
        noise_level=noise_level,
        n_devices=n_devices,
        distribution_shift=distribution_shift,
        mechanism_scale=mechanism_scale,
        harmless_drift=harmless_drift,
    )
    equipment_names = _equipment_names(int(n_devices))
    spec = get_scenario_spec(canonical_scenario)
    unknown_devices = set(spec.abnormal_devices).difference(equipment_names)
    if unknown_devices:
        raise ValueError(f"scenario {canonical_scenario} requires unavailable equipment: {sorted(unknown_devices)}")

    # Mix the canonical scenario into the random stream.  Reusing a numeric
    # seed across S1--S4 therefore still creates independent observations.
    scenario_index = ("S1", "S2", "S3", "S4-main", "S4-null").index(canonical_scenario)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), scenario_index]))
    equipment = pd.Series(
        _balanced_equipment_assignment(rng=rng, n_samples=int(n_samples), equipment=equipment_names),
        name="EQP",
    )
    features: dict[str, np.ndarray] = {}
    feature_names = _feature_names()

    # Baseline process variables have realistic correlation opportunities only
    # through the label model; no device is anomalous until the scenario
    # injection below is applied.
    useful_features = tuple(f"feat_useful_{index}" for index in range(1, N_USEFUL_FEATURES + 1))
    useless_features = tuple(f"feat_useless_{index}" for index in range(1, N_USELESS_FEATURES + 1))
    for feature in useful_features:
        features[feature] = rng.normal(0.0, 1.0, size=n_samples)
    for feature in useless_features:
        features[feature] = rng.normal(0.0, 1.0, size=n_samples)
    features["feat_special_dist"] = rng.normal(0.0, 1.0, size=n_samples)
    features["feat_special_imp"] = rng.normal(0.0, 2.0, size=n_samples)
    features["feat_special_mix"] = rng.normal(0.0, 1.0, size=n_samples)

    # Inject physical/distribution shifts.  In S4 the harmless feature drift
    # is intentionally large but never enters the KPI equation.
    for device, shifted_features in spec.distribution_shift_features.items():
        for feature in shifted_features:
            if feature not in features:
                raise ValueError(f"scenario references unknown distribution feature {feature!r}")
            features[feature] = _inject_distribution_shift(
                features[feature],
                equipment=equipment,
                devices=(device,),
                rng=rng,
                amount=float(distribution_shift),
            )
    for feature in spec.harmless_features:
        if feature not in features:
            raise ValueError(f"scenario references unknown harmless feature {feature!r}")
        features[feature] = _inject_distribution_shift(
            features[feature],
            equipment=equipment,
            devices=spec.abnormal_devices,
            rng=rng,
            amount=float(harmless_drift),
        )

    coefficients = _scenario_coefficients(
        spec=spec,
        equipment=equipment_names,
        mechanism_scale=float(mechanism_scale),
    )
    # The KPI is generated from useful process variables and the three special
    # variables.  Useless variables (including S4 harmless drift) are absent by
    # construction, so the manifest has an exact negative-control set.
    logit = np.full(int(n_samples), -2.5, dtype=float)
    useful_coefficients = rng.uniform(0.8, 1.2, size=len(useful_features))
    for feature, coefficient in zip(useful_features, useful_coefficients, strict=True):
        logit += coefficient * features[feature]
    equipment_array = equipment.to_numpy(dtype=object)
    for feature in SPECIAL_FEATURES:
        coefficient_by_device = coefficients[feature]
        coefficient_array = np.asarray([coefficient_by_device[str(device)] for device in equipment_array])
        logit += coefficient_array * features[feature]
    logit += rng.normal(0.0, float(noise_level), size=int(n_samples))
    probabilities = expit(logit)
    target = rng.binomial(1, probabilities)

    data = pd.DataFrame(
        {"EQP": equipment.to_numpy(dtype=object), **features, "target": target.astype(np.int8)},
    )

    normal_devices = tuple(device for device in equipment_names if device not in spec.abnormal_devices)
    metadata = _build_metadata(
        spec=spec,
        seed=int(seed),
        n_samples=int(n_samples),
        noise_level=float(noise_level),
        n_devices=int(n_devices),
        distribution_shift=float(distribution_shift),
        mechanism_scale=float(mechanism_scale),
        harmless_drift=float(harmless_drift),
        feature_names=feature_names,
    )
    metadata["coefficient_by_device"] = {
        feature: {device: float(value) for device, value in values.items()} for feature, values in coefficients.items()
    }
    return DatasetBundle(
        data=data,
        scenario=canonical_scenario,
        seed=int(seed),
        group_column="EQP",
        target_column="target",
        process_features=feature_names,
        abnormal_devices=spec.abnormal_devices,
        normal_devices=normal_devices,
        root_causes={device: tuple(features_) for device, features_ in spec.root_causes.items()},
        harmless_features=spec.harmless_features,
        metadata=metadata,
    )


@overload
def generate_data(
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    *,
    scenario: str | None = None,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
    return_bundle: Literal[False],
) -> pd.DataFrame: ...


@overload
def generate_data(
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    *,
    scenario: str | None = None,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
    return_bundle: Literal[True],
) -> DatasetBundle: ...


@overload
def generate_data(
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    *,
    scenario: str | None = None,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
    return_bundle: None = None,
) -> DatasetBundle | pd.DataFrame: ...


def generate_data(
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    *,
    scenario: str | None = None,
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
    return_bundle: bool | None = None,
) -> DatasetBundle | pd.DataFrame:
    """Compatibility wrapper around :func:`generate_dataset_bundle`.

    Explicit scenario calls return a :class:`DatasetBundle`, which is the
    paper-experiment API.  Calls without ``scenario`` preserve the original
    patent-demo behavior and return the generated DataFrame for the combined
    S3 fault pattern.  Callers
    that need an unambiguous return type should use
    :func:`generate_dataset_bundle` or pass ``return_bundle`` explicitly.
    """
    explicit_scenario = scenario is not None
    # The old patent demo generated all three special faults in one frame;
    # keep that no-argument behavior while requiring paper experiments to
    # select an independent scenario explicitly.
    selected_scenario = "S3" if scenario is None else scenario
    bundle = generate_dataset_bundle(
        scenario=selected_scenario,
        n_samples=n_samples,
        seed=seed,
        noise_level=noise_level,
        n_devices=n_devices,
        distribution_shift=distribution_shift,
        mechanism_scale=mechanism_scale,
        harmless_drift=harmless_drift,
    )
    if return_bundle is None:
        return_bundle = explicit_scenario
    if not isinstance(return_bundle, bool):
        raise TypeError("return_bundle must be a boolean when provided")
    return bundle if return_bundle else bundle.data.copy()


def generate_data_frame(
    n_samples: int = 10_000,
    seed: int = 42,
    noise_level: float = 0.2,
    *,
    scenario: str = "S1",
    n_devices: int = N_DEVICES,
    distribution_shift: float = 3.0,
    mechanism_scale: float = 1.0,
    harmless_drift: float = 3.5,
) -> pd.DataFrame:
    """Return only the DataFrame for legacy demo/report code."""
    return cast(
        pd.DataFrame,
        generate_data(
            n_samples=n_samples,
            seed=seed,
            noise_level=noise_level,
            scenario=scenario,
            n_devices=n_devices,
            distribution_shift=distribution_shift,
            mechanism_scale=mechanism_scale,
            harmless_drift=harmless_drift,
            return_bundle=False,
        ),
    )


def generate_dataset_report(df_or_bundle: pd.DataFrame | DatasetBundle) -> str:
    """Generate a concise report for a DataFrame or Dataset A bundle."""
    df = df_or_bundle.data if isinstance(df_or_bundle, DatasetBundle) else df_or_bundle
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df_or_bundle must be a pandas DataFrame or DatasetBundle")
    required = {"EQP", "target"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    report = ["### Dataset A synthetic benchmark\n", "\n**Overview**:\n"]
    report.append(f"- Samples: {len(df)}\n")
    report.append(f"- Positive target rate: {df['target'].mean():.2%}\n")
    report.append(f"- Process features: {len(df.columns) - 2}\n")
    report.append(f"- Equipment: {df['EQP'].nunique()} ({', '.join(sorted(df['EQP'].unique()))})\n\n")
    report.append("**Equipment target rates**:\n\n")
    report.append("| EQP | Positive | Samples | Rate |\n| :--- | ---: | ---: | ---: |\n")
    stats = df.groupby("EQP", observed=True)["target"].agg(["sum", "count", "mean"]).sort_index()
    for eqp, row in stats.iterrows():
        report.append(f"| {eqp} | {int(row['sum'])} | {int(row['count'])} | {row['mean']:.2%} |\n")
    return "".join(report)


if __name__ == "__main__":
    print(generate_dataset_report(generate_data_frame()))

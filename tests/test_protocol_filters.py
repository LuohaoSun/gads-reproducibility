"""Tests for protocol-level normal-pool feature filters."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from shap_diff_analysis.experiment_runner import ExperimentSettings, run_experiment_bundle, run_experiment_matrix
from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.protocol_filters import (
    PoolKpiFilterSpec,
    apply_pool_kpi_filter_to_bundle,
    benjamini_hochberg,
    pool_kpi_association_filter,
    pool_shap_std_floor,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

run_matrix_script: ModuleType = importlib.import_module("scripts.run_experiment_matrix")


def test_benjamini_hochberg_matches_step_up_reference() -> None:
    """Adjusted p-values follow the BH step-up formula with monotonicity."""
    # Reference values verified against R p.adjust(method="BH") on the sorted input.
    adjusted = benjamini_hochberg(np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]))
    expected = np.array([0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.0800, 0.074 * 8 / 7, 0.2050])
    assert adjusted == pytest.approx(expected)
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_benjamini_hochberg_rejects_invalid_input() -> None:
    """Malformed p-value input must fail explicitly."""
    with pytest.raises(ValueError, match="non-empty 1-D array"):
        benjamini_hochberg(np.array([], dtype=float))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        benjamini_hochberg(np.array([0.5, 1.5]))


def test_pool_kpi_filter_spec_requires_exactly_one_mode() -> None:
    """Ambiguous or invalid filter specifications are rejected."""
    with pytest.raises(ValueError, match="exactly one"):
        PoolKpiFilterSpec()
    with pytest.raises(ValueError, match="exactly one"):
        PoolKpiFilterSpec(fdr_alpha=0.2, top_k=10)
    with pytest.raises(ValueError, match="fdr_alpha must lie in"):
        PoolKpiFilterSpec(fdr_alpha=0.0)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        PoolKpiFilterSpec(top_k=0)
    assert PoolKpiFilterSpec(top_k=7).mode() == "top_k"
    assert PoolKpiFilterSpec(fdr_alpha=0.2).mode() == "fdr"


def _association_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 120
    target = rng.integers(0, 2, size=n)
    strong = target.astype(float) + rng.normal(0, 0.2, size=n)
    weak = target.astype(float) + rng.normal(0, 3.0, size=n)
    noise_a = rng.normal(0, 1, size=n)
    noise_b = rng.normal(0, 1, size=n)
    return pd.DataFrame(
        {
            "EQP": ["EQP_A"] * n,
            "target": target,
            "f_strong": strong,
            "f_weak": weak,
            "f_noise_a": noise_a,
            "f_noise_b": noise_b,
        }
    )


def test_pool_kpi_association_filter_top_k_keeps_strongest_in_order() -> None:
    """top-k mode keeps the strongest associations in descending |corr| order."""
    frame = _association_frame()
    kept, record = pool_kpi_association_filter(
        frame,
        target_column="target",
        group_column="EQP",
        normal_devices=("EQP_A",),
        process_features=("f_noise_a", "f_weak", "f_strong", "f_noise_b"),
        spec=PoolKpiFilterSpec(top_k=2),
    )
    assert kept[0] == "f_strong"
    assert kept[1] == "f_weak"
    assert record["mode"] == "top_k"
    assert record["n_kept"] == 2
    assert set(record["statistics"]) == {"f_strong", "f_weak", "f_noise_a", "f_noise_b"}
    assert record["statistics"]["f_strong"]["adjusted_p"] < record["statistics"]["f_noise_a"]["adjusted_p"]


def test_pool_kpi_association_filter_fdr_keeps_strong_signals() -> None:
    """FDR mode keeps strong signals while dropping noise features."""
    frame = _association_frame()
    kept, _ = pool_kpi_association_filter(
        frame,
        target_column="target",
        group_column="EQP",
        normal_devices=("EQP_A",),
        process_features=("f_strong", "f_weak", "f_noise_a", "f_noise_b"),
        spec=PoolKpiFilterSpec(fdr_alpha=0.2),
    )
    assert "f_strong" in kept
    assert "f_noise_a" not in kept and "f_noise_b" not in kept


def test_pool_kpi_association_filter_ignores_abnormal_device_rows() -> None:
    """Association present only on the abnormal device must not drive the filter."""
    frame = pd.DataFrame(
        {
            "EQP": ["EQP_A"] * 6 + ["EQP_B"] * 6,
            "target": [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            "f_pool": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
            "f_abnormal_only": [1, 1, -1, -1, 1, 1, 1, 1, 0, 0, 1, 1],
        }
    )
    kept, _ = pool_kpi_association_filter(
        frame,
        target_column="target",
        group_column="EQP",
        normal_devices=("EQP_A",),
        process_features=("f_pool", "f_abnormal_only"),
        spec=PoolKpiFilterSpec(top_k=1),
    )
    assert kept == ("f_pool",)


def test_pool_kpi_association_filter_zero_survivors_raise() -> None:
    """A filter that keeps nothing must raise instead of returning an empty ranking."""
    frame = pd.DataFrame({"EQP": ["EQP_A"] * 8, "target": [0, 1, 0, 1, 0, 1, 0, 1], "f_const": [2.0] * 8})
    with pytest.raises(ValueError, match="kept no process features"):
        pool_kpi_association_filter(
            frame,
            target_column="target",
            group_column="EQP",
            normal_devices=("EQP_A",),
            process_features=("f_const",),
            spec=PoolKpiFilterSpec(fdr_alpha=0.01),
        )


def test_apply_pool_kpi_filter_to_bundle_preserves_annotations() -> None:
    """Bundle-level filtering preserves ground truth and records the filter without mutating the source bundle."""
    bundle = generate_dataset_bundle(scenario="S4-null", n_samples=200, seed=5)
    filtered = apply_pool_kpi_filter_to_bundle(bundle, PoolKpiFilterSpec(top_k=5))
    assert isinstance(filtered, DatasetBundle)
    assert len(filtered.process_features) == 5
    assert set(filtered.process_features) <= set(bundle.process_features)
    assert filtered.harmless_features == bundle.harmless_features
    assert filtered.root_causes == bundle.root_causes
    assert bundle.process_features != filtered.process_features
    record = filtered.metadata["protocol_filters"]["pool_kpi_association"]
    assert record["n_kept"] == 5
    assert record["kept"] == list(filtered.process_features)
    assert "protocol_filters" not in bundle.metadata


def test_pool_shap_std_floor_keeps_upper_tail() -> None:
    """The SHAP-std floor keeps features at or above the percentile threshold."""
    rng = np.random.default_rng(3)
    shap_values = pd.DataFrame(
        rng.normal(0, 1, size=(60, 4)) * np.array([3.0, 2.0, 1.0, 0.1]),
        columns=pd.Index(list("abcd")),
    )
    groups = pd.Series(["EQP_A"] * 40 + ["EQP_B"] * 20)
    kept, record = pool_shap_std_floor(shap_values, groups, ("EQP_A",), list("abcd"), floor_percentile=50.0)
    assert set(kept) == {"a", "b"}
    threshold = float(np.percentile(shap_values.loc[groups == "EQP_A"].std(ddof=1), 50.0))
    assert record["threshold"] == pytest.approx(threshold)
    assert record["n_kept"] == 2
    assert record["pool_shap_std"]["d"] < threshold


def test_pool_shap_std_floor_rejects_invalid_percentile() -> None:
    """Out-of-range floor percentiles are rejected."""
    shap_values = pd.DataFrame(np.ones((10, 2)), columns=pd.Index(list("ab")))
    groups = pd.Series(["EQP_A"] * 10)
    for bad in (100.0, -1.0, float("nan")):
        with pytest.raises((ValueError, TypeError)):
            pool_shap_std_floor(shap_values, groups, ("EQP_A",), list("ab"), bad)


def test_run_experiment_bundle_applies_and_records_shap_std_floor() -> None:
    """run_experiment_bundle applies the floor to every ranking and records it in the manifest."""
    bundle = generate_dataset_bundle(scenario="S4-null", n_samples=200, seed=9)
    settings = ExperimentSettings(
        methods=("GADS",),
        cv_splits=3,
        model_params={"n_estimators": 40, "max_depth": 3},
        shap_std_floor_percentile=30.0,
    )
    result = run_experiment_bundle(bundle, settings=settings)
    floor_record = result.manifest["protocol_filters"]["shap_std_floor"]
    assert floor_record["floor_percentile"] == 30.0
    assert floor_record["n_kept"] < len(bundle.process_features)
    assert result.manifest["settings"]["shap_std_floor_percentile"] == 30.0
    process_view = result.rankings.loc[(result.rankings["method"] == "GADS") & (result.rankings["view"] == "process")]
    ranked = set(process_view["feature"])
    assert ranked == set(floor_record["kept"])
    assert result.metrics.loc[result.metrics["method"] == "GADS", "n_ranked_features"].iloc[0] == floor_record["n_kept"]


def test_run_experiment_matrix_threads_pool_filter(tmp_path: Path) -> None:
    """The matrix runner threads the pool filter into persisted bundles and the summary."""
    summary = run_experiment_matrix(
        output_dir=tmp_path,
        scenarios=("S4-null",),
        seeds=(4,),
        settings=ExperimentSettings(methods=("GADS",), cv_splits=3, model_params={"n_estimators": 40, "max_depth": 3}),
        datasets=("dataset_a",),
        dataset_a_n_samples=200,
        pool_feature_filter=PoolKpiFilterSpec(top_k=6),
    )
    assert summary["pool_feature_filter"] == {"fdr_alpha": None, "top_k": 6}
    run_dir = tmp_path / "runs" / "dataset_a_S4-null_seed4"
    bundle_manifest = json.loads((run_dir / "bundle.manifest.json").read_text(encoding="utf-8"))
    assert len(bundle_manifest["process_features"]) == 6
    record = bundle_manifest["metadata"]["protocol_filters"]["pool_kpi_association"]
    assert record["n_kept"] == 6


def test_cli_parses_protocol_filter_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The matrix CLI translates filter flags into explicit specs and settings."""
    captured: dict[str, object] = {}

    def fake_run_experiment_matrix(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(run_matrix_script, "run_experiment_matrix", fake_run_experiment_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_matrix.py",
            "--output",
            str(tmp_path / "out"),
            "--scenarios",
            "S4-null",
            "--seeds",
            "0",
            "--pool-feature-filter",
            "top-k",
            "--pool-filter-top-k",
            "7",
            "--shap-std-floor-percentile",
            "25",
        ],
    )
    run_matrix_script.main()

    assert captured["pool_feature_filter"] == PoolKpiFilterSpec(top_k=7)
    settings = captured["settings"]
    assert isinstance(settings, ExperimentSettings)
    assert settings.shap_std_floor_percentile == 25.0


def test_cli_defaults_keep_filters_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without filter flags both protocol filters stay disabled."""
    captured: dict[str, object] = {}

    def fake_run_experiment_matrix(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(run_matrix_script, "run_experiment_matrix", fake_run_experiment_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment_matrix.py", "--output", str(tmp_path / "out"), "--scenarios", "S1", "--seeds", "0"],
    )
    run_matrix_script.main()

    assert captured["pool_feature_filter"] is None
    settings = captured["settings"]
    assert isinstance(settings, ExperimentSettings)
    assert settings.shap_std_floor_percentile is None

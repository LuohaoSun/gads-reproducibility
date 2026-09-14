"""Tests for UCI SECOM Dataset B generation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from shap_diff_analysis.secom import (
    DEFAULT_RAW_DIR,
    HARMLESS_GLOBAL_WEIGHT,
    MECHANISM_LOCAL_WEIGHT,
    MECHANISM_ROOT_GLOBAL_WEIGHT,
    MISSING_TOKENS,
    ROOT_CAUSE_GLOBAL_WEIGHT,
    SCENARIO_NAMES,
    SecoMDataError,
    _train_indices,
    assign_pseudo_eqps,
    feature_column_names,
    generate_dataset_b,
    preprocess_features,
    source_manifest,
    verify_secom_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / DEFAULT_RAW_DIR


def _require_raw_files() -> None:
    if not RAW_DIR.is_dir() or not (RAW_DIR / "secom.data").is_file():
        raise unittest.SkipTest(f"SECOM raw files not found in {RAW_DIR}; run scripts/download_secom.py")


class TestSecoMSourceMetadata(unittest.TestCase):
    def test_source_manifest_has_uci_urls_and_sha256(self) -> None:
        manifest = source_manifest()
        self.assertEqual(manifest.dataset_page, "https://archive.ics.uci.edu/dataset/179/secom")
        self.assertEqual(manifest.missing_tokens, MISSING_TOKENS)
        for filename, meta in manifest.files.items():
            self.assertIn("archive.ics.uci.edu", meta["url"])
            self.assertEqual(len(meta["sha256"]), 64, msg=filename)


class TestSecoMIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        _require_raw_files()

    def test_verify_secom_files_passes(self) -> None:
        digests = verify_secom_files(RAW_DIR)
        self.assertEqual(set(digests), {"secom.data", "secom_labels.data", "secom.names"})

    def test_missing_raw_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, self.assertRaises(SecoMDataError):
            verify_secom_files(Path(tmp_dir))


class TestSecoMPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        _require_raw_files()

    def test_missing_tokens_include_question_mark(self) -> None:
        self.assertIn("?", MISSING_TOKENS)

    def test_train_fold_median_does_not_use_holdout(self) -> None:
        feature_columns = feature_column_names()
        values = pd.DataFrame(
            {
                feature_columns[0]: [1.0, 2.0, 3.0, 100.0, 200.0],
                feature_columns[1]: [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        values.loc[3:, feature_columns[0]] = np.nan
        raw = pd.concat([values, pd.Series([0, 0, 0, 0, 0], name="uci_label")], axis=1)
        train_idx = np.array([0, 1, 2])
        processed, _, meta = preprocess_features(raw, feature_columns[:2], train_idx)
        imputed_value = processed.loc[3, feature_columns[0]]
        self.assertEqual(imputed_value, 2.0)
        self.assertEqual(meta["train_medians"][feature_columns[0]], 2.0)
        self.assertTrue(meta["include_missing_indicators"])

    def test_preprocess_without_missing_indicators(self) -> None:
        feature_columns = feature_column_names()[:2]
        values = pd.DataFrame(
            {
                feature_columns[0]: [1.0, 2.0, 3.0, np.nan],
                feature_columns[1]: [0.0, 0.5, np.nan, 1.0],
            }
        )
        raw = pd.concat([values, pd.Series([0, 0, 0, 0], name="uci_label")], axis=1)
        train_idx = np.array([0, 1, 2])
        processed, process_features, meta = preprocess_features(
            raw,
            feature_columns,
            train_idx,
            include_missing_indicators=False,
        )
        self.assertEqual(process_features, feature_columns)
        self.assertEqual(set(processed.columns), set(feature_columns))
        self.assertFalse(meta["include_missing_indicators"])


class TestPseudoEqpPartition(unittest.TestCase):
    def test_assignments_are_deterministic(self) -> None:
        first = assign_pseudo_eqps(1567, group_seed=99)
        second = assign_pseudo_eqps(1567, group_seed=99)
        pd.testing.assert_series_equal(first, second)

    def test_assignments_are_near_equal(self) -> None:
        groups = assign_pseudo_eqps(1567, group_seed=99)
        counts = groups.value_counts()
        self.assertEqual(len(counts), 10)
        self.assertGreaterEqual(counts.min(), 156)
        self.assertLessEqual(counts.max(), 157)

    def test_group_seed_is_independent_from_label_seed(self) -> None:
        _require_raw_files()
        bundle_a = generate_dataset_b("S1", seed=42, group_seed=100, raw_dir=RAW_DIR)
        bundle_b = generate_dataset_b("S1", seed=43, group_seed=100, raw_dir=RAW_DIR)
        pd.testing.assert_series_equal(bundle_a.data["EQP"], bundle_b.data["EQP"])


class TestDatasetBScenarios(unittest.TestCase):
    def setUp(self) -> None:
        _require_raw_files()

    def test_all_scenarios_generate_dataset_bundle(self) -> None:
        for scenario in SCENARIO_NAMES:
            bundle = generate_dataset_b(scenario, seed=7, group_seed=11, raw_dir=RAW_DIR)
            self.assertEqual(bundle.scenario, scenario)
            self.assertIn("EQP", bundle.data.columns)
            self.assertIn("target", bundle.data.columns)
            self.assertGreater(len(bundle.process_features), len(feature_column_names()))

    def test_s1_has_root_cause_and_abnormal_device(self) -> None:
        bundle = generate_dataset_b("S1", seed=7, group_seed=11, raw_dir=RAW_DIR)
        self.assertEqual(bundle.abnormal_devices, ("EQP_A",))
        self.assertEqual(len(bundle.root_causes), 1)
        self.assertIn("EQP_A", bundle.root_causes)

    def test_s2_has_mechanism_root_on_single_device(self) -> None:
        bundle = generate_dataset_b("S2", seed=7, group_seed=11, raw_dir=RAW_DIR)
        self.assertEqual(bundle.abnormal_devices, ("EQP_B",))
        roles = bundle.metadata["feature_roles"]
        self.assertEqual(bundle.root_causes, {"EQP_B": (roles["mechanism_root"],)})

    def test_s3_has_three_independent_fault_devices(self) -> None:
        bundle = generate_dataset_b("S3", seed=7, group_seed=11, raw_dir=RAW_DIR)
        roles = bundle.metadata["feature_roles"]
        root_cause = roles["root_cause"]
        mechanism_root = roles["mechanism_root"]
        self.assertEqual(bundle.abnormal_devices, ("EQP_A", "EQP_B", "EQP_C"))
        self.assertEqual(
            bundle.root_causes,
            {
                "EQP_A": (root_cause,),
                "EQP_B": (mechanism_root,),
                "EQP_C": (root_cause,),
            },
        )
        self.assertEqual(len(bundle.harmless_features), 0)
        self.assertTrue(set(bundle.abnormal_devices).isdisjoint(bundle.normal_devices))
        self.assertNotIn("EQP_A", bundle.normal_devices)
        self.assertNotIn("EQP_B", bundle.normal_devices)
        self.assertNotIn("EQP_C", bundle.normal_devices)
        physical = bundle.metadata["physical_shifts"]
        mechanism = bundle.metadata["glm"]["mechanism"]
        self.assertIn("EQP_A", physical)
        self.assertIn("EQP_C", physical)
        self.assertIn("EQP_B", mechanism)
        self.assertIn("EQP_C", mechanism)
        self.assertEqual(physical["EQP_C"]["features"], [root_cause])
        self.assertEqual(mechanism["EQP_C"]["feature"], root_cause)

    def test_s4_null_has_one_designated_device_and_no_root_causes(self) -> None:
        bundle = generate_dataset_b("S4-null", seed=7, group_seed=11, raw_dir=RAW_DIR)
        self.assertEqual(len(bundle.abnormal_devices), 1)
        self.assertEqual(bundle.root_causes, {})
        self.assertGreater(len(bundle.harmless_features), 0)
        self.assertTrue(set(bundle.abnormal_devices).isdisjoint(bundle.normal_devices))
        self.assertEqual(
            set(bundle.abnormal_devices) | set(bundle.normal_devices),
            set(bundle.data["EQP"]),
        )

    def test_s4_main_has_root_cause_and_harmless_features(self) -> None:
        bundle = generate_dataset_b("S4-main", seed=7, group_seed=11, raw_dir=RAW_DIR)
        roles = bundle.metadata["feature_roles"]
        self.assertEqual(bundle.abnormal_devices, ("EQP_D",))
        abnormal = bundle.abnormal_devices[0]
        self.assertIn(abnormal, bundle.root_causes)
        self.assertEqual(bundle.root_causes[abnormal], (roles["mechanism_root"],))
        self.assertEqual(len(bundle.harmless_features), 10)
        self.assertNotIn(bundle.root_causes[abnormal][0], bundle.harmless_features)
        self.assertIn("EQP_D", bundle.metadata["physical_shifts"])
        self.assertIn("EQP_D", bundle.metadata["glm"]["mechanism"])

    def test_s4_null_uses_eqp_e_without_mechanism_shift(self) -> None:
        bundle = generate_dataset_b("S4-null", seed=7, group_seed=11, raw_dir=RAW_DIR)
        self.assertEqual(bundle.abnormal_devices, ("EQP_E",))
        self.assertEqual(bundle.metadata["glm"]["mechanism"], {})

    def test_generated_target_rate_is_not_saturated(self) -> None:
        for scenario in SCENARIO_NAMES:
            bundle = generate_dataset_b(scenario, seed=7, group_seed=11, raw_dir=RAW_DIR)
            rate = bundle.metadata["generated_target_rate"]
            self.assertGreater(rate, 0.01, msg=scenario)
            self.assertLess(rate, 0.95, msg=scenario)

    def test_role_features_selected_from_non_constant_columns(self) -> None:
        bundle = generate_dataset_b("S1", seed=7, group_seed=11, raw_dir=RAW_DIR)
        roles = bundle.metadata["feature_roles"]
        selected = (
            set(roles["global_signals"]) | {roles["root_cause"], roles["mechanism_root"]} | set(roles["harmless"])
        )
        for feature in selected:
            values = bundle.data[feature]
            self.assertGreater(float(values.std()), 0.0, msg=feature)

    def test_scaling_metadata_recorded(self) -> None:
        bundle = generate_dataset_b("S2", seed=7, group_seed=11, raw_dir=RAW_DIR)
        scaling = bundle.metadata["scaling"]
        self.assertEqual(scaling["method"], "train_fold_location_scale_all_features")
        self.assertEqual(set(scaling["features"]), set(bundle.process_features))
        for stats in scaling["features"].values():
            self.assertIn("loc", stats)
            self.assertIn("scale", stats)
            self.assertGreater(stats["scale"], 0.0)

    def test_non_role_features_are_standardized_on_train_fold(self) -> None:
        """All imputed process features, not only role features, get train-fold scaling."""
        bundle = generate_dataset_b("S4-null", seed=7, group_seed=11, raw_dir=RAW_DIR)
        roles = bundle.metadata["feature_roles"]
        role_features = (
            set(roles["global_signals"]) | {roles["root_cause"], roles["mechanism_root"]} | set(roles["harmless"])
        )
        train_idx = _train_indices(len(bundle.data), seed=7)
        plain_columns = [
            column
            for column in bundle.process_features
            if not column.endswith("_missing") and column not in role_features
        ]
        self.assertGreater(len(plain_columns), 500)
        train_view = bundle.data.loc[train_idx, plain_columns]
        means = train_view.mean()
        stds = train_view.std()
        self.assertTrue(float(means.abs().max()) < 1e-9)
        varying = stds[stds > 0]
        self.assertTrue(bool(((varying - 1.0).abs() < 1e-9).all()))

    def test_missing_indicators_standardized_when_included(self) -> None:
        bundle = generate_dataset_b("S1", seed=7, group_seed=11, raw_dir=RAW_DIR)
        indicator_columns = [column for column in bundle.process_features if column.endswith("_missing")]
        self.assertEqual(len(indicator_columns), len(feature_column_names()))
        train_idx = _train_indices(len(bundle.data), seed=7)
        train_view = bundle.data.loc[train_idx, indicator_columns]
        non_constant = train_view.loc[:, train_view.std() > 0]
        self.assertGreater(len(non_constant.columns), 0)
        self.assertTrue(float(non_constant.mean().abs().max()) < 1e-9)
        self.assertTrue(float((non_constant.std() - 1.0).abs().max()) < 1e-9)

    def test_include_missing_indicators_false_removes_indicator_columns(self) -> None:
        bundle = generate_dataset_b(
            "S4-null",
            seed=7,
            group_seed=11,
            raw_dir=RAW_DIR,
            include_missing_indicators=False,
        )
        self.assertEqual(bundle.process_features, feature_column_names())
        self.assertFalse(any(column.endswith("_missing") for column in bundle.data.columns))
        self.assertFalse(bundle.metadata["preprocess"]["include_missing_indicators"])

    def test_fault_strength_parameters_propagate_to_metadata(self) -> None:
        s3 = generate_dataset_b(
            "S3",
            seed=7,
            group_seed=11,
            raw_dir=RAW_DIR,
            physical_shift_magnitude=5.5,
            mechanism_local_weight=20.0,
            harmless_shift_magnitude=6.5,
        )
        self.assertEqual(s3.metadata["physical_shifts"]["EQP_A"]["magnitude"], 5.5)
        self.assertEqual(s3.metadata["physical_shifts"]["EQP_C"]["magnitude"], 5.5)
        self.assertEqual(s3.metadata["glm"]["mechanism"]["EQP_B"]["local_weight"], 20.0)
        self.assertEqual(s3.metadata["glm"]["coefficient_spec"]["mechanism_local_weight"], 20.0)

        s4_main = generate_dataset_b(
            "S4-main",
            seed=7,
            group_seed=11,
            raw_dir=RAW_DIR,
            physical_shift_magnitude=5.5,
            mechanism_local_weight=20.0,
            harmless_shift_magnitude=6.5,
        )
        physical = s4_main.metadata["physical_shifts"]["EQP_D"]
        self.assertEqual(physical["magnitude"], 6.5)
        self.assertEqual(sorted(physical["features"]), sorted(s4_main.harmless_features))
        self.assertEqual(s4_main.metadata["glm"]["mechanism"]["EQP_D"]["local_weight"], 20.0)
        params = s4_main.metadata["generation_params"]
        self.assertEqual(params["physical_shift_magnitude"], 5.5)
        self.assertEqual(params["mechanism_local_weight"], 20.0)
        self.assertEqual(params["harmless_shift_magnitude"], 6.5)

    def test_generation_params_recorded_in_metadata(self) -> None:
        bundle = generate_dataset_b("S1", seed=9, group_seed=13, raw_dir=RAW_DIR)
        params = bundle.metadata["generation_params"]
        self.assertEqual(
            set(params),
            {
                "scenario",
                "seed",
                "group_seed",
                "intercept",
                "noise_level",
                "physical_shift_magnitude",
                "mechanism_local_weight",
                "harmless_shift_magnitude",
                "include_missing_indicators",
            },
        )
        self.assertEqual(params["seed"], 9)
        self.assertEqual(params["group_seed"], 13)
        self.assertEqual(params["physical_shift_magnitude"], 3.0)
        self.assertEqual(params["mechanism_local_weight"], MECHANISM_LOCAL_WEIGHT)
        self.assertTrue(params["include_missing_indicators"])

    def test_manifest_is_json_serializable(self) -> None:
        bundle = generate_dataset_b("S3", seed=7, group_seed=11, raw_dir=RAW_DIR)
        payload = bundle.manifest()
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["scenario"], "S3")

    def test_glm_coefficients_recorded_and_match_injection_spec(self) -> None:
        """Regression: generation coefficients are explicit and independent of model ranking."""
        s1 = generate_dataset_b("S1", seed=7, group_seed=11, raw_dir=RAW_DIR)
        roles = s1.metadata["feature_roles"]
        spec = s1.metadata["glm"]["coefficient_spec"]
        weights = s1.metadata["glm"]["weights"]

        self.assertEqual(spec["root_cause_global"], ROOT_CAUSE_GLOBAL_WEIGHT)
        self.assertEqual(spec["mechanism_root_global"], MECHANISM_ROOT_GLOBAL_WEIGHT)
        self.assertEqual(spec["mechanism_local_weight"], MECHANISM_LOCAL_WEIGHT)
        self.assertEqual(spec["harmless_global"], HARMLESS_GLOBAL_WEIGHT)

        self.assertEqual(weights[roles["root_cause"]], ROOT_CAUSE_GLOBAL_WEIGHT)
        self.assertGreater(weights[roles["root_cause"]], HARMLESS_GLOBAL_WEIGHT)
        for feature in roles["harmless"]:
            self.assertEqual(weights[feature], HARMLESS_GLOBAL_WEIGHT)

        for scenario in ("S2", "S3", "S4-main"):
            bundle = generate_dataset_b(scenario, seed=7, group_seed=11, raw_dir=RAW_DIR)
            mechanism = bundle.metadata["glm"]["mechanism"]
            self.assertGreater(len(mechanism), 0, msg=scenario)
            for device_meta in mechanism.values():
                self.assertEqual(device_meta["local_weight"], MECHANISM_LOCAL_WEIGHT, msg=scenario)

        s4_null = generate_dataset_b("S4-null", seed=7, group_seed=11, raw_dir=RAW_DIR)
        self.assertEqual(s4_null.root_causes, {})


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run the paper experiment matrix across scenarios, seeds and datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shap_diff_analysis.experiment_runner import (
    DEFAULT_MATRIX_SCENARIOS,
    ExperimentSettings,
    is_secom_raw_available,
    run_experiment_matrix,
)
from shap_diff_analysis.experiment_types import EQP_MODES
from shap_diff_analysis.gads import GADS_DISTANCES
from shap_diff_analysis.generate_data import N_DEVICES
from shap_diff_analysis.protocol_filters import PoolKpiFilterSpec
from shap_diff_analysis.secom import (
    DEFAULT_RAW_DIR,
    HARMLESS_SHIFT_MAGNITUDE,
    MECHANISM_LOCAL_WEIGHT,
    PHYSICAL_SHIFT_MAGNITUDE,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="root directory for runs and aggregate tables")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("dataset_a", "dataset_b"),
        default=("dataset_a",),
        help="datasets to include; dataset_b runs only when SECOM raw files are available",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_MATRIX_SCENARIOS),
        help="scenario names (default: S1 S2 S3 S4-main S4-null)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="random seeds for each scenario")
    parser.add_argument("--dataset-a-n-samples", type=int, default=1_000, help="rows for on-the-fly Dataset A bundles")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Dataset A label-generation noise standard deviation",
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        default=N_DEVICES,
        help="Dataset A equipment count (10-26, EQP_A--EQP_J always present)",
    )
    parser.add_argument(
        "--distribution-shift",
        type=float,
        default=3.0,
        help="Dataset A mean shift for injected physical drift",
    )
    parser.add_argument(
        "--mechanism-scale",
        type=float,
        default=1.0,
        help="Dataset A multiplier for injected conditional-effect changes",
    )
    parser.add_argument(
        "--harmless-drift",
        type=float,
        default=3.5,
        help="Dataset A mean shift applied to S4 harmless features",
    )
    parser.add_argument("--dataset-b-raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="SECOM raw directory")
    parser.add_argument(
        "--dataset-b-group-seed",
        type=int,
        default=2024,
        help="pseudo-EQP partition seed for Dataset B",
    )
    parser.add_argument(
        "--dataset-b-physical-shift",
        type=float,
        default=PHYSICAL_SHIFT_MAGNITUDE,
        help="Dataset B physical drift magnitude in standardized units (default: 3.0)",
    )
    parser.add_argument(
        "--dataset-b-mechanism-weight",
        type=float,
        default=MECHANISM_LOCAL_WEIGHT,
        help="Dataset B local GLM coefficient on mechanism-shifted devices (default: 12.0)",
    )
    parser.add_argument(
        "--dataset-b-harmless-shift",
        type=float,
        default=HARMLESS_SHIFT_MAGNITUDE,
        help="Dataset B harmless drift magnitude in standardized units (default: 4.0)",
    )
    parser.add_argument(
        "--dataset-b-include-missing-indicators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep Dataset B *_missing indicator columns as process features (default: keep)",
    )
    parser.add_argument(
        "--pool-feature-filter",
        choices=("fdr", "top-k"),
        default=None,
        help=(
            "protocol-level pre-filter: keep only process features associated with the KPI on "
            "normal-pool rows (point-biserial); 'fdr' keeps Benjamini-Hochberg survivors at "
            "--pool-filter-fdr-alpha, 'top-k' keeps the --pool-filter-top-k strongest. The filter "
            "applies to the whole pipeline (KPI-model input and every method's ranking) and is "
            "recorded in every run manifest"
        ),
    )
    parser.add_argument(
        "--pool-filter-fdr-alpha",
        type=float,
        default=0.2,
        help="FDR alpha for --pool-feature-filter fdr (default: 0.2)",
    )
    parser.add_argument(
        "--pool-filter-top-k",
        type=int,
        default=100,
        help="feature count for --pool-feature-filter top-k (default: 100)",
    )
    parser.add_argument(
        "--shap-std-floor-percentile",
        type=float,
        default=None,
        help=(
            "protocol-level ranking floor: exclude process features whose normal-pool OOF SHAP "
            "standard deviation is below this percentile before any method ranks features "
            "(0 <= value < 100); recorded in every run manifest"
        ),
    )
    parser.add_argument("--model-type", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=None,
        help="override XGBoost/LightGBM min_child_weight regularization (default: library value)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="override XGBoost/LightGBM max_depth (default: pipeline value 6)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="override XGBoost/LightGBM n_estimators (default: pipeline value 300)",
    )
    parser.add_argument("--auc-threshold", type=float, default=None)
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument("--gads-distance", choices=GADS_DISTANCES, default="wasserstein")
    parser.add_argument("--gads-bins", type=int, default=50)
    parser.add_argument("--gads-epsilon", type=float, default=1e-10)
    parser.add_argument(
        "--m2oe-max-group-rows",
        type=int,
        default=128,
        help=(
            "cap on abnormal-group rows for M2OE-Group training (seeded deterministic subsample; "
            "default: 128, pass a large value to disable)"
        ),
    )
    parser.add_argument(
        "--xpe-max-rows",
        nargs=2,
        type=int,
        metavar=("TARGET", "SOURCE"),
        default=None,
        help=(
            "optional XPE row caps before the optimal-transport solve: TARGET caps the abnormal "
            "device rows, SOURCE caps the normal pool (e.g. --xpe-max-rows 200 2000). Subsampling "
            "is seeded and stratified by the KPI label; default: no cap"
        ),
    )
    parser.add_argument(
        "--include-invalid-auc",
        action="store_true",
        help="include invalid_auc runs in the aggregate tables (default: exclude them)",
    )
    parser.add_argument("--method", action="append", dest="methods", default=None)
    parser.add_argument(
        "--eqp-mode",
        choices=EQP_MODES,
        default=None,
        help="EQP handling mode (default: contextual_process, the paper protocol)",
    )
    return parser


def build_model_params(args: argparse.Namespace) -> dict[str, Any] | None:
    """Collect explicit estimator overrides; None keeps the pipeline defaults."""
    model_params = {
        key: value
        for key, value in {
            "min_child_weight": args.min_child_weight,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
        }.items()
        if value is not None
    }
    return model_params or None


def build_settings(args: argparse.Namespace) -> ExperimentSettings:
    """Translate parsed CLI arguments into ExperimentSettings."""
    methods = tuple(args.methods) if args.methods else ExperimentSettings().methods
    return ExperimentSettings(
        methods=methods,
        model_type=args.model_type,
        model_seed=args.model_seed,
        cv_seed=args.cv_seed,
        cv_splits=args.cv_splits,
        model_params=build_model_params(args),
        auc_threshold=args.auc_threshold,
        psi_bins=args.psi_bins,
        gads_distance=args.gads_distance,
        gads_bins=args.gads_bins,
        gads_epsilon=args.gads_epsilon,
        eqp_mode=args.eqp_mode if args.eqp_mode is not None else ExperimentSettings().eqp_mode,
        shap_std_floor_percentile=args.shap_std_floor_percentile,
        m2oe_max_group_rows=args.m2oe_max_group_rows,
        xpe_max_rows=tuple(args.xpe_max_rows) if args.xpe_max_rows is not None else None,
    )


def build_pool_feature_filter(args: argparse.Namespace) -> PoolKpiFilterSpec | None:
    """Translate the --pool-feature-filter choice into an explicit spec."""
    if args.pool_feature_filter is None:
        return None
    if args.pool_feature_filter == "fdr":
        return PoolKpiFilterSpec(fdr_alpha=args.pool_filter_fdr_alpha)
    return PoolKpiFilterSpec(top_k=args.pool_filter_top_k)


def main() -> None:
    """Execute the matrix and print a JSON summary."""
    args = build_parser().parse_args()
    if "dataset_b" in args.datasets and not is_secom_raw_available(args.dataset_b_raw_dir):
        print(
            json.dumps(
                {
                    "warning": "dataset_b requested but SECOM raw files are unavailable",
                    "raw_dir": str(args.dataset_b_raw_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    settings = build_settings(args)
    summary = run_experiment_matrix(
        output_dir=args.output,
        scenarios=args.scenarios,
        seeds=args.seeds,
        settings=settings,
        datasets=args.datasets,
        dataset_a_n_samples=args.dataset_a_n_samples,
        dataset_a_noise_level=args.noise_level,
        dataset_a_n_devices=args.n_devices,
        dataset_a_distribution_shift=args.distribution_shift,
        dataset_a_mechanism_scale=args.mechanism_scale,
        dataset_a_harmless_drift=args.harmless_drift,
        dataset_b_raw_dir=args.dataset_b_raw_dir,
        dataset_b_group_seed=args.dataset_b_group_seed,
        dataset_b_physical_shift=args.dataset_b_physical_shift,
        dataset_b_mechanism_weight=args.dataset_b_mechanism_weight,
        dataset_b_harmless_shift=args.dataset_b_harmless_shift,
        dataset_b_include_missing_indicators=args.dataset_b_include_missing_indicators,
        pool_feature_filter=build_pool_feature_filter(args),
        exclude_invalid_auc=not args.include_invalid_auc,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

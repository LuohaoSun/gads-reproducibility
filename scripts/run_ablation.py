#!/usr/bin/env python3
"""Compare GADS distance functions or EQP handling modes on generated bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shap_diff_analysis.experiment_runner import (
    ExperimentSettings,
    load_dataset_bundle,
    run_eqp_ablation,
    run_eqp_ablation_matrix,
    run_gads_distance_ablation,
    run_gads_distance_ablation_matrix,
    write_experiment_outputs,
)
from shap_diff_analysis.experiment_types import EQP_MODES
from shap_diff_analysis.gads import GADS_DISTANCES
from shap_diff_analysis.generate_data import generate_dataset_bundle


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation",
        choices=("distance", "distance-matrix", "eqp", "eqp-matrix"),
        default="distance",
        help=(
            "distance: compare GADS distances on one bundle; "
            "distance-matrix: multi-seed matrix; "
            "eqp: compare EQP modes; eqp-matrix: matrix sweep"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="directory for ablation tables")
    parser.add_argument("--data", type=Path, default=None, help="generated DatasetBundle table")
    parser.add_argument("--manifest", type=Path, default=None, help="JSON ground-truth manifest")
    parser.add_argument("--scenario", default="S1", help="generate Dataset A on the fly when --data is omitted")
    parser.add_argument("--seed", type=int, default=42, help="seed for on-the-fly Dataset A generation")
    parser.add_argument("--n-samples", type=int, default=500, help="rows for on-the-fly Dataset A generation")
    parser.add_argument(
        "--distances",
        nargs="+",
        choices=GADS_DISTANCES,
        default=list(GADS_DISTANCES),
        help="GADS distance functions to compare (distance ablation)",
    )
    parser.add_argument(
        "--eqp-modes",
        nargs="+",
        choices=EQP_MODES,
        default=list(EQP_MODES),
        help="EQP handling modes to compare (eqp ablation)",
    )
    parser.add_argument("--scenarios", nargs="+", default=["S1"], help="scenarios for matrix ablations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="seeds for matrix ablations")
    parser.add_argument("--dataset-a-n-samples", type=int, default=500, help="rows for matrix Dataset A bundles")
    parser.add_argument("--model-type", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--gads-bins", type=int, default=50)
    parser.add_argument("--gads-epsilon", type=float, default=1e-10)
    return parser


def _load_or_generate_bundle(args: argparse.Namespace):
    if args.data is None and args.manifest is None:
        return generate_dataset_bundle(scenario=args.scenario, n_samples=args.n_samples, seed=args.seed)
    if args.data is None or args.manifest is None:
        raise ValueError("--data and --manifest must be supplied together")
    return load_dataset_bundle(args.data, args.manifest)


def main() -> None:
    """Run the selected ablation and persist combined outputs."""
    args = build_parser().parse_args()
    settings = ExperimentSettings(
        methods=("GADS",),
        model_type=args.model_type,
        model_seed=args.model_seed,
        cv_seed=args.cv_seed,
        cv_splits=args.cv_splits,
        gads_bins=args.gads_bins,
        gads_epsilon=args.gads_epsilon,
    )
    if args.ablation == "eqp-matrix":
        summary = run_eqp_ablation_matrix(
            output_dir=args.output,
            scenarios=args.scenarios,
            seeds=args.seeds,
            settings=settings,
            modes=tuple(args.eqp_modes),
            datasets=("dataset_a",),
            dataset_a_n_samples=args.dataset_a_n_samples,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.ablation == "distance-matrix":
        summary = run_gads_distance_ablation_matrix(
            output_dir=args.output,
            scenarios=args.scenarios,
            seeds=args.seeds,
            settings=settings,
            distances=tuple(args.distances),
            datasets=("dataset_a",),
            dataset_a_n_samples=args.dataset_a_n_samples,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    bundle = _load_or_generate_bundle(args)
    if args.ablation == "eqp":
        result = run_eqp_ablation(bundle, modes=tuple(args.eqp_modes), settings=settings)
        payload_key = "eqp_modes"
        payload_values = list(args.eqp_modes)
    else:
        result = run_gads_distance_ablation(bundle, distances=args.distances, settings=settings)
        payload_key = "distances"
        payload_values = list(args.distances)
    paths = write_experiment_outputs(result, args.output)
    payload = {
        payload_key: payload_values,
        "scenario": bundle.scenario,
        "seed": bundle.seed,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

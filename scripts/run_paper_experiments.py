#!/usr/bin/env python3
"""Run the paper experiment engine on a generated DatasetBundle.

Dataset generators intentionally remain separate.  They write one tabular
file and its JSON manifest; this command consumes those two artifacts and
therefore works identically for Dataset A and Dataset B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shap_diff_analysis.experiment_runner import (
    ExperimentSettings,
    load_dataset_bundle,
    run_experiment_bundle,
    write_experiment_outputs,
)
from shap_diff_analysis.gads import GADS_DISTANCES


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="generated DatasetBundle table")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON ground-truth manifest")
    parser.add_argument("--output", type=Path, required=True, help="directory for result tables")
    parser.add_argument("--model-type", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--auc-threshold", type=float, default=None)
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument("--gads-distance", choices=GADS_DISTANCES, default="wasserstein")
    parser.add_argument("--gads-bins", type=int, default=50)
    parser.add_argument("--gads-epsilon", type=float, default=1e-10)
    parser.add_argument("--method", action="append", dest="methods", default=None)
    return parser


def main() -> None:
    """Load a generated bundle, run all requested methods and persist outputs."""
    args = build_parser().parse_args()
    bundle = load_dataset_bundle(args.data, args.manifest)
    methods = tuple(args.methods) if args.methods else ExperimentSettings().methods
    settings = ExperimentSettings(
        methods=methods,
        model_type=args.model_type,
        model_seed=args.model_seed,
        cv_seed=args.cv_seed,
        cv_splits=args.cv_splits,
        auc_threshold=args.auc_threshold,
        psi_bins=args.psi_bins,
        gads_distance=args.gads_distance,
        gads_bins=args.gads_bins,
        gads_epsilon=args.gads_epsilon,
    )
    result = run_experiment_bundle(bundle, settings=settings)
    paths = write_experiment_outputs(result, args.output)
    print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

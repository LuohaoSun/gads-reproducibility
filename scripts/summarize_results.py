#!/usr/bin/env python3
"""Summarize experiment-matrix outputs with bootstrap CIs and paired tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shap_diff_analysis.statistics import (
    DEFAULT_SUMMARY_METRICS,
    PairedTestName,
    load_experiment_metrics,
    summarize_experiment_results,
    write_stats_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="experiment matrix root containing aggregate/metrics.parquet",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="override path to metrics.parquet (default: <input-dir>/aggregate/metrics.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for stats outputs (default: <input-dir>/stats)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_SUMMARY_METRICS),
        help="metric columns to summarize and compare",
    )
    parser.add_argument(
        "--reference-method",
        default="GADS",
        help="reference method for paired comparisons",
    )
    parser.add_argument(
        "--test",
        choices=("permutation", "wilcoxon"),
        default="permutation",
        help="paired significance test",
    )
    parser.add_argument(
        "--holm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply Holm correction within each dataset/scenario/metric group",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000, help="bootstrap resamples for CIs")
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=20_000,
        help="randomization resamples for permutation tests",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for resampling")
    return parser


def main() -> None:
    """Load aggregate metrics, summarize, compare and write tables."""
    args = build_parser().parse_args()
    metrics_path = args.metrics_path or (args.input_dir / "aggregate" / "metrics.parquet")
    output_dir = args.output_dir or (args.input_dir / "stats")
    metrics = load_experiment_metrics(metrics_path)
    summary, comparisons = summarize_experiment_results(
        metrics,
        metric_columns=tuple(args.metrics),
        reference_method=args.reference_method,
        test=cast_test(args.test),
        holm=args.holm,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    paths = write_stats_outputs(summary, comparisons, output_dir)
    payload = {
        "metrics_path": str(metrics_path),
        "output_dir": str(output_dir),
        "n_valid_rows": len(metrics),
        "n_summary_rows": len(summary),
        "n_comparison_rows": len(comparisons),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cast_test(value: str) -> PairedTestName:
    """Validate and return a paired-test name."""
    if value == "permutation":
        return "permutation"
    if value == "wilcoxon":
        return "wilcoxon"
    raise ValueError("test must be 'permutation' or 'wilcoxon'")


if __name__ == "__main__":
    main()

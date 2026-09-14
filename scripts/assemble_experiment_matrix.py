#!/usr/bin/env python3
"""Assemble a canonical experiment matrix from partial run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shap_diff_analysis.experiment_runner import MatrixRunSelection, assemble_experiment_matrix
from shap_diff_analysis.scenario import normalize_scenario


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="canonical matrix root directory")
    parser.add_argument(
        "--dataset",
        default="dataset_b",
        help="dataset prefix used in run directory names (default: dataset_b)",
    )
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        metavar="SOURCE:SCENARIOS:SEEDS",
        help="partial source as source_dir:S1,S2:0,1,2,3,4 (repeatable)",
    )
    parser.add_argument(
        "--include-invalid-auc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include invalid_auc runs in aggregate tables (default: exclude them)",
    )
    return parser


def parse_selection(raw: str) -> MatrixRunSelection:
    """Parse one ``source:scenarios:seeds`` selection string."""
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"selection must be source:scenarios:seeds, got {raw!r}")
    source_dir = Path(parts[0])
    scenarios = tuple(normalize_scenario(item) for item in parts[1].split(",") if item)
    seeds = tuple(int(item) for item in parts[2].split(",") if item)
    if not scenarios:
        raise ValueError(f"selection must include at least one scenario: {raw!r}")
    if not seeds:
        raise ValueError(f"selection must include at least one seed: {raw!r}")
    return MatrixRunSelection(source_dir=source_dir, scenarios=scenarios, seeds=seeds)


def main() -> None:
    """Merge selected partial runs and write aggregate outputs."""
    args = build_parser().parse_args()
    selections = [parse_selection(raw) for raw in args.selection]
    summary = assemble_experiment_matrix(
        output_dir=args.output,
        selections=selections,
        dataset=args.dataset,
        exclude_invalid_auc=not args.include_invalid_auc,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

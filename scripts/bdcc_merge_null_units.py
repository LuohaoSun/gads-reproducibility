"""Merge per-unit null outputs into one cell-per-(dataset, scenario) set of CSVs.

The parallel fan-out writes one directory per (dataset, scenario, seed) unit.
This script collects them, drops duplicate draws, and reproduces the layout the
single-driver run produced, so downstream comparison scripts keep working
unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect(sources: list[Path]) -> pd.DataFrame:
    """Concatenate every per-unit ``null_permutations.csv`` under the sources.

    Two layouts produce these files: the serial driver writes one directory per
    (dataset, scenario) cell, and the fan-out writes one per
    (dataset, scenario, seed) unit.  Both are accepted so a run that was
    partially completed by each can be merged without rerunning either part.
    """
    files = [path for source in sources for path in sorted(source.glob("*/null_permutations.csv"))]
    if not files:
        raise SystemExit(f"no unit outputs under {', '.join(str(s) for s in sources)}")
    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    # A unit that was rerun would otherwise contribute the same draw twice; the
    # draw identity is (cell, seed, permutation, device).
    keys = ["dataset", "scenario", "scenario_seed", "permutation", "device"]
    duplicated = merged.duplicated(subset=keys).sum()
    if duplicated:
        print(f"dropping {duplicated} duplicated draw rows")
    return merged.drop_duplicates(subset=keys).sort_values(keys, ignore_index=True)


def main() -> None:
    """Write merged null CSVs and a completeness check."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[
            Path("results/bdcc_revision/label_permutation_null_paper20/units"),
            Path("results/bdcc_revision/label_permutation_null_paper20/cells"),
        ],
        help="directories whose immediate children hold null_permutations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bdcc_revision/label_permutation_null_paper20"),
    )
    args = parser.parse_args()

    merged = collect(args.sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "null_permutations.csv", index=False)

    coverage = (
        merged.groupby(["dataset", "scenario"])
        .agg(n_seeds=("scenario_seed", "nunique"), n_devices=("device", "nunique"), n_rows=("device", "size"))
        .reset_index()
    )
    print(coverage.to_string(index=False))
    incomplete = coverage.loc[coverage["n_seeds"] < 20]
    if not incomplete.empty:
        raise SystemExit(f"incomplete coverage, fewer than 20 seeds:\n{incomplete.to_string(index=False)}")
    print(f"\nwrote {args.output_dir / 'null_permutations.csv'} ({len(merged)} rows, all cells at 20 seeds)")


if __name__ == "__main__":
    main()

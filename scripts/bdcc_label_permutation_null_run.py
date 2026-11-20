"""Dispatch the label-permutation null one (dataset, scenario) cell per process.

``bdcc_label_permutation_null.py`` walks its cells serially, which wastes the
20 cores available on the compute host: every cell is independent, so the whole
sweep finishes in the time of the slowest cell rather than the sum of all
cells.  This driver launches one subprocess per cell, then concatenates the
per-cell CSVs into the same ``null_permutations.csv`` / ``null_summary.csv``
layout the serial script produces, so downstream tooling does not care which
entry point was used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import pandas as pd


def _cell_command(
    *,
    python: Path,
    script: Path,
    cell_dir: Path,
    dataset: str,
    scenario: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(python),
        "-I",
        str(script),
        "--output-dir",
        str(cell_dir),
        "--datasets",
        dataset,
        "--scenarios",
        scenario,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--n-permutations",
        str(args.n_permutations),
        "--model-seed",
        str(args.model_seed),
        "--cv-splits",
        str(args.cv_splits),
        "--dataset-b-raw-dir",
        str(args.dataset_b_raw_dir),
        "--protocol",
        args.protocol,
    ]


def _run_cell(
    *,
    python: Path,
    script: Path,
    output_dir: Path,
    dataset: str,
    scenario: str,
    args: argparse.Namespace,
    env: dict[str, str],
    threads: int,
) -> tuple[str, str, int, str]:
    cell_dir = output_dir / "cells" / f"{dataset}__{scenario}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    command = _cell_command(
        python=python,
        script=script,
        cell_dir=cell_dir,
        dataset=dataset,
        scenario=scenario,
        args=args,
    )
    cell_env = {**env, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)}
    log_path = cell_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=cell_env, check=False)
    tail = log_path.read_text(encoding="utf-8").strip().splitlines()[-3:]
    return dataset, scenario, completed.returncode, "\n".join(tail)


def _merge(output_dir: Path, cells: list[tuple[str, str]]) -> None:
    """Concatenate per-cell outputs, failing loudly if a cell produced nothing."""
    frame_parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for dataset, scenario in cells:
        path = output_dir / "cells" / f"{dataset}__{scenario}" / "null_permutations.csv"
        if not path.exists():
            missing.append(f"{dataset}/{scenario}")
            continue
        part = pd.read_csv(path)
        if part.empty:
            missing.append(f"{dataset}/{scenario} (empty)")
            continue
        frame_parts.append(part)
    if missing:
        raise SystemExit(f"label-permutation null: cells produced no rows: {', '.join(missing)}")
    frame = pd.concat(frame_parts, ignore_index=True)
    frame.to_csv(output_dir / "null_permutations.csv", index=False)

    summary = cast(
        pd.DataFrame,
        frame.groupby(["dataset", "scenario"], as_index=False).agg(
            n_permutations=("permutation", "size"),
            max_score_mean=("max_score", "mean"),
            max_score_sd=("max_score", "std"),
            max_score_q95=("max_score", lambda s: float(s.quantile(0.95))),
            max_score_q99=("max_score", lambda s: float(s.quantile(0.99))),
            top1_is_root_rate=("top1_is_root", "mean"),
            root_in_top5_rate=("root_in_top5", "mean"),
            mean_auc=("mean_auc", "mean"),
        ),
    ).sort_values(["dataset", "scenario"])
    summary.to_csv(output_dir / "null_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {len(frame)} rows from {len(frame_parts)} cells to {output_dir}")


def main() -> None:
    """Dispatch one process per cell, then merge the per-cell outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["dataset_a", "dataset_b"])
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4-main", "S4-null"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--dataset-b-raw-dir", type=Path, default=Path("data/raw/secom"))
    parser.add_argument(
        "--protocol",
        choices=("paper", "defaults"),
        default="paper",
        help="generation and model protocol forwarded to each cell",
    )
    parser.add_argument("--max-parallel", type=int, default=0, help="0 means one per cell")
    parser.add_argument("--threads-per-process", type=int, default=2)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "bdcc_label_permutation_null.py"
    python = Path(sys.executable)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = [(dataset, scenario) for dataset in args.datasets for scenario in args.scenarios]
    workers = args.max_parallel if args.max_parallel > 0 else min(len(cells), os.cpu_count() or 1)
    if workers * args.threads_per_process > (os.cpu_count() or 1) * 2:
        raise SystemExit(
            f"label-permutation null: {workers} processes x {args.threads_per_process} threads "
            f"oversubscribes {os.cpu_count()} cores; lower --max-parallel or --threads-per-process"
        )

    env = dict(os.environ)
    print(f"dispatching {len(cells)} cells with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_cell,
                python=python,
                script=script,
                output_dir=args.output_dir,
                dataset=dataset,
                scenario=scenario,
                args=args,
                env=env,
                threads=args.threads_per_process,
            )
            for dataset, scenario in cells
        ]
        failures: list[str] = []
        for future in futures:
            dataset, scenario, code, tail = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"[{dataset}/{scenario}] {status}\n{tail}", flush=True)
            if code != 0:
                failures.append(f"{dataset}/{scenario}")
    if failures:
        raise SystemExit(f"label-permutation null: cells failed: {', '.join(failures)}")

    _merge(args.output_dir, cells)


if __name__ == "__main__":
    main()

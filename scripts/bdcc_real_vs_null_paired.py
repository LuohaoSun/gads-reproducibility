"""Compare each observed run against the null draws of the same scenario realization.

The single-seed null asks whether the statistic beats label noise.  It does not
match the observed sweep, which spans 20 scenario realizations: the observed
runs come from realizations the null never saw.  This script pairs them
explicitly.  For every (dataset, scenario, scenario_seed, device) cell it takes

* the observed maximum process statistic at that realization (model seed 42), and
* the 20 label-permutation draws at that same realization,

and reports how often the observed value exceeds the largest null draw, together
with the per-cell ratio to the null mean.  Pooling the cells then gives the
headline numbers used in the manuscript.

Outputs ``real_vs_null_paired.csv`` (one row per cell) and
``real_vs_null_paired_summary.csv`` (one row per dataset and scenario).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def observed_max(path: Path, model_seed: int) -> pd.DataFrame:
    """Per-run maximum process statistic over the real-signal sweep."""
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame = frame[(frame["method"] == "GADS") & (frame["view"] == "process") & (frame["model_seed"] == model_seed)]
    keys = ["dataset", "scenario", "scenario_seed", "equipment"]
    return cast(pd.DataFrame, frame.groupby(keys, as_index=False)["score"].max())


def compare(observed_path: Path, null_path: Path, model_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each observed run with the null draws of the same realization."""
    observed = observed_max(observed_path, model_seed)
    null = pd.read_csv(null_path)
    dataset_labels: dict[str, str] = {"dataset_a": "Dataset A", "dataset_b": "Dataset B"}
    null["dataset"] = null["dataset"].replace(dataset_labels)

    rows: list[dict[str, object]] = []
    for cell, block in observed.groupby(["dataset", "scenario", "scenario_seed"], sort=True):
        dataset, scenario, scenario_seed = cast(tuple[str, str, int], cell)
        devices = sorted(block["equipment"].unique())
        draws = null.loc[
            (null["dataset"] == dataset) & (null["scenario"] == scenario) & (null["scenario_seed"] == scenario_seed)
        ]
        if draws.empty:
            raise SystemExit(f"no null draws for {dataset}/{scenario}/seed={scenario_seed}")
        for device in devices:
            per_device = draws.loc[draws["device"] == device, "max_score"].to_numpy()
            if per_device.size == 0:
                raise SystemExit(f"no null draws for {dataset}/{scenario}/{device}/seed={scenario_seed}")
            device_observed = float(block.loc[block["equipment"] == device, "score"].max())
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "scenario_seed": scenario_seed,
                    "device": device,
                    "observed": device_observed,
                    "null_mean": float(per_device.mean()),
                    "null_sd": float(per_device.std(ddof=1)),
                    "null_max": float(per_device.max()),
                    "null_q99": float(np.quantile(per_device, 0.99)),
                    "n_null": int(per_device.size),
                    "ratio": device_observed / float(per_device.mean()),
                    "above_null_max": bool(device_observed > per_device.max()),
                    # Empirical p at this cell, bounded below by the draw count.
                    "p_cell": float((1 + (per_device >= device_observed).sum()) / (1 + per_device.size)),
                }
            )
    cells = cast(pd.DataFrame, pd.DataFrame(rows)).sort_values(["dataset", "scenario", "scenario_seed", "device"])

    summary = cast(
        pd.DataFrame,
        cells.groupby(["dataset", "scenario"], as_index=False).agg(
            n_cells=("observed", "size"),
            n_devices=("device", "nunique"),
            n_null_per_cell=("n_null", "max"),
            observed_mean=("observed", "mean"),
            null_mean=("null_mean", "mean"),
            ratio_mean=("ratio", "mean"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            above_rate=("above_null_max", "mean"),
            p_max=("p_cell", "max"),
        ),
    ).sort_values(["dataset", "scenario"])
    return cells, summary


def main() -> None:
    """Write the paired observed-versus-null tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observed",
        type=Path,
        default=Path("results/bdcc_revision/pack1_seeds_hyperparams/model_seed_sweep/rankings_long.parquet"),
    )
    parser.add_argument(
        "--null",
        type=Path,
        default=Path("results/bdcc_revision/label_permutation_null_paper20/null_permutations.csv"),
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.null.parent
    cells, summary = compare(args.observed, args.null, args.model_seed)
    cells.to_csv(out_dir / "real_vs_null_paired.csv", index=False)
    summary.to_csv(out_dir / "real_vs_null_paired_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(f"\nwrote {out_dir / 'real_vs_null_paired.csv'}")
    print(f"wrote {out_dir / 'real_vs_null_paired_summary.csv'}")


if __name__ == "__main__":
    main()

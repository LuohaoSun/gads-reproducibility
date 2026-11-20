"""Compare the observed GADS statistic against the label-permutation null.

Reads the existing real-signal ranking sweep (``pack1_seeds_hyperparams``) and
the label-permutation null produced by ``bdcc_label_permutation_null.py``, then
reports how far the observed per-run maximum statistic sits above the null.

Two quantities matter for Reviewer 3.8:

* the null distribution itself (mean, SD, 95th/99th percentile, maximum), and
* how many observed runs exceed the largest null draw, which bounds the
  empirical p-value from below at ``1 / (1 + n_null)``.

Outputs ``real_vs_null.csv`` next to the null CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def _observed_max_scores(path: Path) -> pd.DataFrame:
    """Per-run maximum process statistic over the real-signal sweep."""
    frame = pd.read_csv(path)
    frame = frame[(frame["method"] == "GADS") & (frame["view"] == "process")]
    keys = ["dataset", "scenario", "equipment", "scenario_seed", "model_seed"]
    return cast(pd.DataFrame, frame.groupby(keys, as_index=False)["score"].max())


def _null_max_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    dataset_labels: dict[str, str] = {"dataset_a": "Dataset A", "dataset_b": "Dataset B"}
    frame["dataset"] = frame["dataset"].replace(dataset_labels)
    return frame


def compare(observed_path: Path, null_path: Path, output_path: Path) -> pd.DataFrame:
    """Build the observed-versus-null separation table."""
    observed = _observed_max_scores(observed_path)
    null = _null_max_scores(null_path)

    rows: list[dict[str, object]] = []
    for key, block in observed.groupby(["dataset", "scenario"], sort=True):
        dataset, scenario = cast(tuple[str, str], key)
        draws = null.loc[(null["dataset"] == dataset) & (null["scenario"] == scenario), "max_score"].to_numpy()
        if draws.size == 0:
            raise SystemExit(f"no null draws for {dataset}/{scenario}")
        scores = block["score"].to_numpy()
        # Empirical p-value per observed run: how often a label permutation
        # produces a statistic at least as large as the observed one.
        exceed = (draws[None, :] >= scores[:, None]).sum(axis=1)
        p_values = (1.0 + exceed) / (1.0 + draws.size)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "n_observed_runs": int(scores.size),
                "n_null_draws": int(draws.size),
                "observed_max_score_mean": float(scores.mean()),
                "observed_max_score_min": float(scores.min()),
                "null_max_score_mean": float(draws.mean()),
                "null_max_score_sd": float(draws.std(ddof=1)),
                "null_max_score_q95": float(np.quantile(draws, 0.95)),
                "null_max_score_q99": float(np.quantile(draws, 0.99)),
                "null_max_score_max": float(draws.max()),
                "fold_over_null_mean": float(scores.mean() / draws.mean()),
                "fold_over_null_q99": float(scores.mean() / np.quantile(draws, 0.99)),
                "frac_observed_above_null_max": float((scores > draws.max()).mean()),
                "median_empirical_p": float(np.median(p_values)),
                "max_empirical_p": float(p_values.max()),
            }
        )
    table = pd.DataFrame(rows).sort_values(["dataset", "scenario"])
    table.to_csv(output_path, index=False)
    return table


def main() -> None:
    """Write the observed-versus-null separation table."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observed",
        type=Path,
        default=Path("results/bdcc_revision/pack1_seeds_hyperparams/model_seed_sweep/rankings_long.csv"),
    )
    parser.add_argument(
        "--null",
        type=Path,
        default=Path("results/bdcc_revision/label_permutation_null/null_permutations.csv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.null.parent / "real_vs_null.csv"
    table = compare(args.observed, args.null, output)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()

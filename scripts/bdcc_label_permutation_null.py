"""Label-permutation null distribution for GADS.

Reviewer 3.8 asks for a null distribution "e.g., obtained by permuting tool
labels", so that a real signal can be separated from noise.  Prof. Chen's
annotation specifies the interpretation to implement: permute the KPI label and
refit, rather than reuse the fixed OOF SHAP matrix with permuted equipment
membership.

For each realization this script:

1. builds the benchmark bundle for one (dataset, scenario, seed),
2. permutes the binary KPI label y across rows, destroying any KPI-process
   association while keeping the covariate distribution, group sizes and class
   balance intact,
3. refits the full cross-fitted KPI model on the permuted labels and recomputes
   OOF SHAP (the expensive step, so it is done once per permutation),
4. ranks features with GADS and records the maximum process statistic and the
   rank of the (now meaningless by construction) annotated root cause.

The result is the null distribution of the ranking statistic: any real-signal
claim must exceed what pure label noise produces.

Layout of the outputs mirrors ``bdcc_pack3_diagnostics_and_bsweep.py`` so the
two can be compared directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import pandas as pd

from shap_diff_analysis.experiment_runner import (
    DatasetGenerationParams,
    _generate_matrix_bundle,
    git_commit,
)
from shap_diff_analysis.gads import rank_gads_features
from shap_diff_analysis.modeling import fit_cross_fitted_model
from shap_diff_analysis.protocol_filters import pool_shap_std_floor

SCENARIOS = ("S1", "S2", "S3", "S4-main", "S4-null")
DATASETS = ("dataset_a", "dataset_b")


def _stable_seed(*parts: Any) -> int:
    """Reproducible 32-bit seed from arbitrary parts.

    Hashing the whole joined string (not a truncated prefix) keeps every
    argument in the seed: truncating to the first bytes makes the seed depend
    only on the constant prefix, so all cells would silently share one random
    stream.
    """
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return int(np.frombuffer(digest[:4], dtype=np.uint32)[0] & 0x7FFFFFFF)


def null_realization(
    *,
    dataset: str,
    scenario: str,
    scenario_seed: int,
    n_permutations: int,
    model_seed: int,
    cv_splits: int,
    generation: DatasetGenerationParams,
    model_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the label-permutation null for one (dataset, scenario, seed) cell."""
    bundle = _generate_matrix_bundle(dataset=dataset, scenario=scenario, seed=scenario_seed, generation=generation)
    data = bundle.data
    features = list(bundle.process_features)
    group_column = bundle.group_column
    target_column = bundle.target_column
    model_columns = [*features, group_column]

    X = data.loc[:, model_columns]
    y = cast(pd.Series, pd.to_numeric(data[target_column], errors="raise")).astype(int)
    groups = cast(pd.Series, data[group_column]).astype(str)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(_stable_seed("label-permutation-null", dataset, scenario, scenario_seed))
    y_values = y.to_numpy()
    n_pos = int(y_values.sum())
    n_total = len(y_values)

    for permutation in range(n_permutations):
        # Permute the KPI label; the covariate matrix and group sizes are untouched.
        shuffled = y_values.copy()
        chosen = rng.choice(n_total, size=n_pos, replace=False)
        shuffled[:] = 0
        shuffled[chosen] = 1
        permuted_y = pd.Series(shuffled, index=y.index, name=target_column)

        if permuted_y.nunique() < 2:
            continue  # degenerate draw; a valid permutation needs both classes

        attributions = fit_cross_fitted_model(
            X,
            permuted_y,
            model_seed=model_seed,
            cv_splits=cv_splits,
            model_params=model_params,
        )
        kept, _ = pool_shap_std_floor(attributions.shap_values, groups, bundle.normal_devices, features, 0.0)
        ranking = rank_gads_features(
            attributions.shap_values,
            groups,
            bundle.abnormal_devices,
            bundle.normal_devices,
            process_features=kept,
            view="process",
            distance="wasserstein",
        )
        for device in bundle.abnormal_devices:
            block = ranking.loc[ranking["equipment"] == device]
            if block.empty:
                continue
            ordered = block.sort_values(["rank", "feature"], kind="stable")["feature"].tolist()
            roots = [f for f in bundle.root_causes.get(str(device), ()) if f in ordered]
            rank_of_root = (ordered.index(roots[0]) + 1) if roots else None
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "scenario_seed": scenario_seed,
                    "permutation": permutation,
                    "device": device,
                    "mean_auc": float(attributions.mean_auc),
                    "n_ranked": len(ordered),
                    "max_score": float(block["score"].max()),
                    "top1_feature": ordered[0],
                    "top1_is_root": bool(roots and ordered[0] == roots[0]),
                    "root_rank": rank_of_root,
                    "root_in_top5": bool(rank_of_root is not None and rank_of_root <= 5),
                }
            )
    return rows


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the null distribution per dataset and scenario."""
    if frame.empty:
        return frame
    grouped = frame.groupby(["dataset", "scenario"], as_index=False).agg(
        n_permutations=("permutation", "size"),
        max_score_mean=("max_score", "mean"),
        max_score_sd=("max_score", "std"),
        max_score_q95=("max_score", lambda s: float(np.quantile(s, 0.95))),
        max_score_q99=("max_score", lambda s: float(np.quantile(s, 0.99))),
        top1_is_root_rate=("top1_is_root", "mean"),
        root_in_top5_rate=("root_in_top5", "mean"),
        mean_auc=("mean_auc", "mean"),
    )
    return cast(pd.DataFrame, grouped)


def main() -> None:
    """Run the label-permutation null over every (dataset, scenario, seed) cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--dataset-b-raw-dir", type=Path, default=Path("data/raw/secom"))
    parser.add_argument(
        "--protocol",
        choices=("paper", "defaults"),
        default="paper",
        help=(
            "paper: match the main-sweep protocol (n=5000, missing-indicator columns "
            "excluded, depth 3 / minimum child weight 5). defaults: reproduce the "
            "generation and model defaults, which differ from the main sweep."
        ),
    )
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="stop after this many (dataset, scenario, seed) cells; for pilot runs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The null must be computed under the same generation and model settings as
    # the real-signal sweep, otherwise the ratio compares two different
    # experiments. The paper protocol below mirrors the main results.
    if args.protocol == "paper":
        generation = DatasetGenerationParams(
            dataset_b_raw_dir=args.dataset_b_raw_dir,
            dataset_a_n_samples=5_000,
            dataset_b_include_missing_indicators=False,
        )
        model_params: dict[str, Any] | None = {"max_depth": 3, "min_child_weight": 5}
    else:
        generation = DatasetGenerationParams(dataset_b_raw_dir=args.dataset_b_raw_dir)
        model_params = None
    cells = [
        (dataset, scenario, seed) for dataset in args.datasets for scenario in args.scenarios for seed in args.seeds
    ]
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]

    protocol = {
        "name": "label-permutation-null",
        "git_commit": git_commit(),
        "datasets": list(args.datasets),
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "n_permutations": args.n_permutations,
        "model_seed": args.model_seed,
        "cv_splits": args.cv_splits,
        "permutation_target": "kpi_label",
        "refit_per_permutation": True,
        "protocol": args.protocol,
        "dataset_a_n_samples": generation.dataset_a_n_samples,
        "dataset_b_include_missing_indicators": generation.dataset_b_include_missing_indicators,
        "model_params": model_params,
        "distance": "wasserstein",
        "view": "process",
        "shap_std_floor_percentile": 0.0,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (dataset, scenario, seed) in enumerate(cells, 1):
        cell_start = time.perf_counter()
        rows = null_realization(
            dataset=dataset,
            scenario=scenario,
            scenario_seed=seed,
            n_permutations=args.n_permutations,
            model_seed=args.model_seed,
            cv_splits=args.cv_splits,
            generation=generation,
            model_params=model_params,
        )
        all_rows.extend(rows)
        elapsed = time.perf_counter() - cell_start
        print(
            f"[{index}/{len(cells)}] {dataset} {scenario} seed={seed}: {len(rows)} permutations in {elapsed:.1f}s",
            flush=True,
        )
        pd.DataFrame(all_rows).to_csv(args.output_dir / "null_permutations.csv", index=False)

    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_dir / "null_permutations.csv", index=False)
    summary = summarise(frame)
    summary.to_csv(args.output_dir / "null_summary.csv", index=False)
    total = time.perf_counter() - started
    print(f"\nwrote {len(frame)} rows; total {total:.1f}s; summary in {args.output_dir}")


if __name__ == "__main__":
    main()

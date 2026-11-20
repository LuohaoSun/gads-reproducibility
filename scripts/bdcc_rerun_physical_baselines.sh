#!/usr/bin/env bash
# Rerun the four physical/explanation baselines on Dataset B under the corrected
# benchmark construction, so that manuscript Table 5 draws every one of its six
# columns from the same construction.
#
# Why this exists: the archived P-main-20 batch standardized only the role
# features, while the revision batch standardizes every process feature. Table 5
# currently mixes the two, because the revision batch measured only GADS,
# Global-SHAP and the modern baselines. This script supplies the missing four.
#
# Sharding: one process per scenario seed, so 20 processes fill the 20 cores.
# Each process runs all five scenarios for its seed. A single seed/scenario run
# with the four methods takes about 35 s, so one shard is roughly 3 min.
#
# Usage:
#   bash scripts/bdcc_rerun_physical_baselines.sh [output_root] [workers]
#
# Results land in <output_root>/seed_<n>/aggregate/metrics.parquet.

set -euo pipefail

# One BLAS thread per process. Without this each worker starts its own OpenBLAS
# thread pool, so 20 workers produce several hundred threads on a 20-core box and
# the run stalls in the kernel scheduler instead of computing. Parallelism here
# comes from the number of processes, not from threads inside each process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUT_ROOT="${1:-results/bdcc_rerun/physical_baselines_b}"
WORKERS="${2:-20}"
PY="${PY:-.venv/bin/python}"

cd "$(dirname "$0")/.."
SCRIPT_ABS="$PWD/scripts/bdcc_rerun_physical_baselines.sh"
mkdir -p "$OUT_ROOT/logs"

# Some WSL2 hosts ship a wedged GPU driver: every open of /dev/dxg lands in an
# uninterruptible sleep that even SIGKILL cannot clear, and XGBoost walks that
# path when it enumerates devices at fit time. Hiding the device node behind a
# private mount namespace turns the hang into an ordinary open failure, so the
# probe gives up and the ensemble fits on CPU. The mount exists only inside the
# namespace and needs no root, because an unprivileged user namespace supplies
# the capability; the host sees no change. Passing device="cpu" or setting
# CUDA_VISIBLE_DEVICES is not enough on its own: the block happens before those
# settings are consulted.
if [ -z "${BDCC_DXG_HIDDEN:-}" ] && [ -e /dev/dxg ]; then
    export BDCC_DXG_HIDDEN=1
    exec unshare -r -m -- bash -c 'mount --bind /dev/null /dev/dxg && exec "$@"' _ \
        "$SCRIPT_ABS" "$@"
fi

echo "output: $OUT_ROOT"
echo "workers: $WORKERS"

run_seed() {
    local seed="$1"
    local out="$OUT_ROOT/seed_${seed}"
    "$PY" scripts/run_experiment_matrix.py \
        --output "$out" \
        --datasets dataset_b \
        --scenarios S1 S2 S3 S4-main S4-null \
        --seeds "$seed" \
        --dataset-b-group-seed 2024 \
        --no-dataset-b-include-missing-indicators \
        --max-depth 3 --min-child-weight 5 \
        --dataset-b-physical-shift 3.0 \
        --dataset-b-mechanism-weight 12.0 \
        --dataset-b-harmless-shift 4.0 \
        --eqp-mode contextual_process \
        --method KS-X --method PSI-X --method Domain-SHAP --method Wasserstein-X \
        > "$OUT_ROOT/logs/seed_${seed}.log" 2>&1
}
export -f run_seed
export OUT_ROOT PY

printf '%s\n' $(seq 0 19) | xargs -P "$WORKERS" -I{} bash -c 'run_seed "$@"' _ {}

echo "all seeds finished; merging"

"$PY" - "$OUT_ROOT" <<'PY_EOF'
"""Merge per-seed metric tables into one aggregate and print the Table 5 columns."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
frames = []
for seed_dir in sorted(root.glob("seed_*")):
    path = seed_dir / "aggregate" / "metrics.parquet"
    if not path.exists():
        raise SystemExit(f"missing metrics for {seed_dir.name}: {path}")
    frames.append(pd.read_parquet(path))

merged = pd.concat(frames, ignore_index=True)
merged.to_parquet(root / "merged_metrics.parquet", index=False)
merged.to_csv(root / "merged_metrics.csv", index=False)

n_seeds = merged["seed"].nunique()
if n_seeds != 20:
    raise SystemExit(f"expected 20 seeds, found {n_seeds}")

table = merged.groupby(["method", "scenario"])["mrr"].mean().unstack().round(3)
order = ["KS-X", "PSI-X", "Domain-SHAP", "Wasserstein-X"]
table = table.loc[order, ["S1", "S2", "S3", "S4-main"]]
print()
print("Mean MRR over 20 scenario seeds (Table 5 rows):")
print(table.to_string())
print()
overall = table.mean(axis=1).round(3)
print("Overall MRR (equal weight over S1-S4-main):")
print(overall.to_string())
print()
print(f"wrote {root / 'merged_metrics.parquet'}")
PY_EOF

echo "done"

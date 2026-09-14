# shap_diff_analysis — GADS reproducibility package

Reference implementation and experiment code for **GADS** (*Grouped Attribution Distribution
Shift*), a method for equipment-level root-cause localization in discrete manufacturing, plus
the full set of baselines, scenarios and scripts needed to reproduce the reported results.

The method fits a global KPI (quality label) model, computes cross-fitted SHAP attributions,
groups them by equipment (EQP), and scores process features by the **distribution distance**
between the abnormal equipment's attribution distribution and a reference pool of healthy
equipment (Wasserstein by default). The output is a ranked list of candidate process root
causes, with a built-in distinction between a genuine mechanism shift and a harmless physical
drift.

This repository is a **code-only** export. Manuscript sources, patents and private archives are
not included; see `README` scope notes below.

---

## 1. Concepts and conventions

### Views

The KPI model is trained with the equipment identifier (`EQP`) as a covariate. GADS therefore
produces three ranking views:

| View | Features ranked | Use |
| --- | --- | --- |
| `process` | process features only | **Primary metric view for the paper** |
| `all` | process features + `EQP` | Combined view |
| `residual` | `EQP` column only | Inspect context-only attribution shift |

### EQP handling modes

| `--eqp-mode` | KPI model inputs | Primary ranking view |
| --- | --- | --- |
| `contextual_process` (default) | process features + `EQP` | `process` |
| `no_eqp` | process features | `process` |
| `all_features` | process features + `EQP` | `all` |

### Scenarios

| Scenario | Description | Root cause | Harmless drift | Metric scope |
| --- | --- | --- | --- | --- |
| `S1` | Parameter drift | — (distribution shift only) | not injected | MRR / HR@K / FAR@K |
| `S2` | Mechanism shift | injected mechanism fault | not injected | MRR / HR@K / FAR@K |
| `S3` | Mixed faults | multiple concurrent faults | not injected | MRR / HR@K / FAR@K |
| `S4-main` | Root cause + harmless drift | one real mechanism root cause | 10 harmless features | MRR / HR@K / FAR@K |
| `S4-null` | Negative control | none (`root_causes` empty) | injected | FAR@K only (MRR/HR@K are `NaN`, never reported as 0) |

`S4-main` and `S4-null` are the pair that separates true root causes from harmless physical
drift: an ideal method keeps a high MRR on `S4-main` while keeping FAR@K low on `S4-null`.
For `S1`–`S3` no harmless drift is injected, so FAR@K is undefined (`NaN`) there.

### Methods

Classical baselines (all enabled by default): `GADS`, `KS-X`, `PSI-X`, `Domain-SHAP`,
`Wasserstein-X`, `Global-SHAP`.

Modern baselines (opt in with `--method`): `M2OE-Group` (vendored M2OE port, see
[Licensing](#10-licensing-and-third-party-code)), `XPE`.

### Metrics

`mrr`, `hr_at_1`, `hr_at_3`, `hr_at_5`, `far_at_1`, `far_at_5`. All primary metrics are
computed on the `process` view.

---

## 2. Requirements

- Python **>= 3.10, < 3.12** (3.11 recommended).
- [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies resolve from
  **official PyPI**.
- Optional, only for Figure 1: `rsvg-convert` from
  [librsvg](https://wiki.gnome.org/Projects/Librsvg) on `PATH` (macOS: `brew install librsvg`).
  It is not a Python dependency; the figure script fails loudly if it is missing.

```bash
uv sync                 # creates .venv with runtime + dev dependencies
uv run python -c "import shap_diff_analysis; print('ok')"
```

`matplotlib`, `seaborn`, `pytest`, `ruff` and `pyright` live in the `dev` dependency group,
which `uv sync` installs by default. `--model-type lightgbm` works because `lightgbm` is a
regular dependency.

---

## 3. Data

### Dataset A — fully synthetic

Dataset A is generated in process from `shap_diff_analysis.generate_data.generate_dataset_bundle`
(`EQP` + process features + `target`, with a ground-truth manifest). It has **no download step**
and no standalone CLI generator; use the experiment matrix or the Python API. Default generation
parameters: `n_samples=1000`, `noise_level=0.2`, `n_devices=10`, `distribution_shift=3.0`,
`mechanism_scale=1.0`, `harmless_drift=3.5`.

To persist a bundle for a single-run experiment:

```python
from pathlib import Path

from shap_diff_analysis.experiment_runner import write_dataset_bundle
from shap_diff_analysis.generate_data import generate_dataset_bundle

bundle = generate_dataset_bundle(scenario="S1", n_samples=1000, seed=42)
paths = write_dataset_bundle(bundle, Path("data/generated/dataset_a"), stem="dataset_a_S1_seed42")
print(paths["data"], paths["manifest"])
```

### Dataset B — SECOM semi-synthetic

Dataset B builds on the UCI **SECOM** dataset (McCann & Johnston, 2008; DOI
[10.24432/C54305](https://doi.org/10.24432/C54305)) and injects faults plus a pseudo-EQP
partition locally. Raw SECOM files are **not** committed. Download and verify them with:

```bash
uv run python scripts/download_secom.py
```

The script downloads into `data/raw/secom`, verifies SHA256 and writes
`data/raw/secom/download_manifest.json`:

| File | SHA256 |
| --- | --- |
| `secom.data` | `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c` |
| `secom_labels.data` | `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03` |
| `secom.names` | `6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59` |

Existing files are re-verified, and a mismatch raises. Use `--force` to re-download, and
`--raw-dir` to change the target directory.

Generate one bundle per scenario:

```bash
uv run python scripts/generate_dataset_b.py \
  --scenario S1 --seed 42 --group-seed 2024 \
  --output-dir data/generated/dataset_b
```

Outputs: `dataset_b_{scenario}_seed{seed}_group{group_seed}.parquet` and a matching
`.manifest.json`. The UCI labels themselves are never exported into the bundle.

> Dataset B is **semi-synthetic**: faults and labels are injected on top of real SECOM process
> measurements, so it is a controlled benchmark, not a real production-line validation.

---

## 4. Smoke test (minutes, not hours)

```bash
# tiny Dataset A matrix: all five scenarios, 2 seeds, small n
uv run python scripts/run_experiment_matrix.py \
  --output results/smoke_matrix \
  --datasets dataset_a \
  --scenarios S1 S2 S3 S4-main S4-null \
  --seeds 42 43 \
  --dataset-a-n-samples 400 \
  --cv-splits 3

uv run python scripts/summarize_results.py --input-dir results/smoke_matrix
uv run python scripts/plot_paper_figs.py --input-dir results/smoke_matrix
```

---

## 5. Single run

`scripts/run_paper_experiments.py` runs every (or a selected) method on one generated bundle:

```bash
uv run python scripts/run_paper_experiments.py \
  --data data/generated/dataset_b/dataset_b_S1_seed42_group2024.parquet \
  --manifest data/generated/dataset_b/dataset_b_S1_seed42_group2024.manifest.json \
  --output results/single_run
```

Key flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model-type` | `xgboost` | `xgboost` or `lightgbm` |
| `--model-seed` | `42` | model seed |
| `--cv-seed` / `--cv-splits` | `None` / `5` | cross-fitting fold seed and count |
| `--method` | all six classical | repeatable, e.g. `--method GADS --method KS-X` |
| `--auc-threshold` | `None` | if set and mean OOF AUC is below it, the run is marked `invalid_auc` |
| `--gads-distance` | `wasserstein` | `wasserstein`, `mean`, `kl`, `js` |
| `--gads-bins` / `--gads-epsilon` | `50` / `1e-10` | histogram bins and smoothing for `kl`/`js` only |
| `--psi-bins` | `10` | PSI-X binning |

---

## 6. Experiment matrix

```bash
uv run python scripts/run_experiment_matrix.py \
  --output results/matrix \
  --datasets dataset_a dataset_b \
  --scenarios S1 S2 S3 S4-main S4-null \
  --seeds 0 1 2 3 4
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--datasets` | `dataset_a` | `dataset_a` and/or `dataset_b` |
| `--scenarios` | all five | scenario list |
| `--seeds` | `42` | data-generation seeds |
| `--dataset-a-n-samples` | `1000` | Dataset A rows |
| `--eqp-mode` | `contextual_process` | see modes table |
| `--dataset-b-raw-dir` | `data/raw/secom` | raw SECOM directory |
| `--dataset-b-group-seed` | `2024` | pseudo-EQP partition seed |
| `--dataset-b-include-missing-indicators` | on | add missing-value indicator columns (use the `--no-...` form to exclude them) |
| `--dataset-b-physical-shift` / `--dataset-b-mechanism-weight` / `--dataset-b-harmless-shift` | `3.0` / `12.0` / `4.0` | Dataset B fault strengths |
| `--include-invalid-auc` | off | keep `invalid_auc` runs in aggregation |
| `--method` | all six classical | repeatable; enables `M2OE-Group` / `XPE` |
| `--m2oe-max-group-rows` / `--xpe-max-rows` | `128` / — | modern-baseline caps |
| `--pool-feature-filter` + `--pool-filter-fdr-alpha` / `--pool-filter-top-k` | off | optional reference-pool feature filtering |

Dataset B runs only if the raw SECOM files are present and pass verification; otherwise the
dataset is skipped, a warning JSON is printed, and the reason is recorded in
`matrix_summary.json` under `skipped_datasets`.

### Output layout

```
results/matrix/
├── matrix_summary.json            # run list, aggregate paths, invalid counts
├── runs/{dataset}_{scenario}_seed{seed}/
│   ├── bundle.parquet / bundle.manifest.json
│   ├── rankings.parquet / .csv    # per method × equipment × feature, with view + ground truth
│   ├── metrics.parquet / .csv     # MRR / HR@K / FAR@K on the process view
│   ├── equipment_metrics.parquet / .csv
│   ├── fold_metrics.parquet / .csv
│   ├── manifest.json              # status, settings, git commit, timestamps
│   └── metrics_table.tex
├── aggregate/                     # valid-run aggregate (+ metrics_table.tex)
├── stats/                         # created by summarize_results.py
└── figures/                       # created by plot_paper_figs.py
```

Matrix runs are sequential; independent matrices can be launched in parallel.

---

## 7. Ablations

GADS distance functions:

```bash
# single bundle: generate Dataset A on the fly if --data/--manifest are omitted
uv run python scripts/run_ablation.py \
  --ablation distance --output results/ablation_distance \
  --scenario S1 --seed 42 --n-samples 500 \
  --distances wasserstein mean kl js

# multi-seed distance matrix (Dataset A only)
uv run python scripts/run_ablation.py \
  --ablation distance-matrix --output results/ablation_distance_matrix \
  --scenarios S2 --seeds 0 1 2 3 4 --distances wasserstein mean kl js \
  --dataset-a-n-samples 5000 --cv-splits 5
```

EQP handling ablation:

```bash
# single bundle (requires an explicit --data/--manifest pair)
uv run python scripts/run_ablation.py \
  --ablation eqp --data <bundle>.parquet --manifest <bundle>.manifest.json \
  --output results/eqp_ablation \
  --eqp-modes contextual_process no_eqp all_features

# multi-mode matrix (Dataset A only)
uv run python scripts/run_ablation.py \
  --ablation eqp-matrix --output results/eqp_matrix \
  --scenarios S2 --seeds 0 1 2 3 4 \
  --eqp-modes contextual_process no_eqp all_features \
  --dataset-a-n-samples 5000 --cv-splits 5
```

The `distance-matrix` and `eqp-matrix` modes always use Dataset A.

### Sensitivity sweeps

`scripts/run_sensitivity_sweep.py` runs single-factor Dataset A sweeps. Default factor grids:

| Factor | Levels |
| --- | --- |
| `noise_level` | 0.1, 0.2, 0.4, 0.8 |
| `n_samples` | 1000, 2500, 5000, 10000 |
| `n_devices` | 10, 15, 20 |
| `distribution_shift` | 1.5, 3.0, 4.5 |
| `mechanism_scale` | 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 |
| `harmless_drift` | 2.0, 3.5, 5.0 |

Each factor varies while every other generation parameter stays at its base value. Defaults:
scenarios `S1 S2`, seed `0`, base `n_samples=1000`, methods
`GADS Wasserstein-X Global-SHAP`. Restrict with `--factors` and `--levels NAME=value`
(e.g. `--factors n_devices --levels n_devices=15 --n-samples 1500`).

### Assembling a matrix from partial runs

`scripts/assemble_experiment_matrix.py` merges disjoint partial run directories into one
canonical matrix:

```bash
uv run python scripts/assemble_experiment_matrix.py \
  --output results/matrix_assembled --dataset dataset_b \
  --selection /tmp/part1:S1,S2:0,1,2,3,4 \
  --selection /tmp/part2:S3:0,1,2,3,4
```

---

## 8. Statistics and figures

The pipeline is **matrix → summarize → plot**. Neither statistics nor figures run automatically.

### Statistics

```bash
uv run python scripts/summarize_results.py --input-dir results/matrix
```

- Per method × metric, 10 000 percentile bootstrap resamples of seed means (`seed=42`) → 95% CI.
- Paired tests against a reference method (default `GADS`): two-sided paired sign-flip
  permutation test (20 000 resamples, `seed=42`), or `--test wilcoxon`.
- Holm step-down correction across the five classical baselines within each
  dataset/scenario/metric (`--holm`, default on).
- Only `status=valid` runs are summarized. Non-applicable rows (e.g. `S4-null` MRR, `S1`–`S3`
  FAR) are marked not applicable instead of being silently zeroed.

Outputs: `stats/stats_summary.{csv,parquet,tex}`, `stats/stats_comparisons.{csv,parquet,tex}`,
`stats/stats.{csv,parquet}`.

### Standard figures

```bash
uv run python scripts/plot_paper_figs.py --input-dir results/matrix
```

Produces `main_metrics_bars.png`, `far_bars.png`, `seed_topk_jaccard.png`,
`rank_stability.png`, `fig2_gads_seed_stability.png` in `results/matrix/figures/`.
`--fig-types` selects groups (`bars`, `far`, `jaccard`, `rank-heatmap`, `fig2`, `fig1`,
`tradeoff`, `g3`, `all`). `fig1` renders `assets/fig1_framework.svg` to a vector PDF and needs
`rsvg-convert`; `g3` needs the merged G3 sensitivity table via `--g3-parquet`.

**Figures need results first.** `summarize_results.py` must have produced
`stats/stats_summary.parquet`, and the matrix must contain `aggregate/metrics.parquet` and
`aggregate/rankings.parquet`; the trade-off figure additionally reads the `stats_summary.parquet`
of a second matrix resolved from `--paper-final-dir` (or the parent of `--input-dir`).

### Manuscript figure bundle

`scripts/build_isa_figures.py` renders the four manuscript-facing assets
(`fig1_framework.pdf`, `tradeoff_primary_cohort.pdf`, `fig2.png`, `fig4.png`) into one output
directory. It requires `rsvg-convert` and an archive directory laid out as:

```
<archive>/matrix_a_xgb/{aggregate/{metrics,rankings}.parquet,stats/stats_summary.parquet}
<archive>/matrix_b_xgb/stats/stats_summary.parquet
<archive>/g3_distance_intensity/g3_distance_intensity_merged.parquet
```

```bash
uv run python scripts/build_isa_figures.py \
  --output-dir figures \
  --archive-dir results/paper_final \
  --fig1-svg assets/fig1_framework.svg
```

`--archive-dir` defaults to `<repo>/results/paper_final`; create that archive with the commands
in [Section 9](#9-reproducing-the-paper). The builder fails explicitly (no fallback) when
`rsvg-convert` or any required input is missing.

---

## 9. Reproducing the paper

### Protocol card

Common settings: `model_seed 42`, `cv_splits 5`, top-K = 1/3/5, GADS `wasserstein` distance with
50 bins and epsilon `1e-10` (G3 varies the distance), `eqp_mode contextual_process`,
`auc_threshold null` (OOF `kpi_auc` recorded, no configured gate), `exclude_invalid_auc true`,
PSI bins 10. The **primary metric view is `process`**.

Dataset A defaults: `n_samples 5000` in the main matrices, `noise 0.2`, `n_devices 10`,
`distribution_shift 3.0`, `mechanism_scale 1.0`, `harmless_drift 3.5`.

Dataset B protocol: SECOM raw data in `data/raw/secom`; all-feature standardization fitted on a
**fixed benchmark-construction split** (a deterministic 80% row split derived from `--seed`,
used for median imputation and location/scale — not the KPI model's CV train fold); missing-value
indicators **excluded** (`--no-dataset-b-include-missing-indicators`); injected faults (in
standardized units) `physical_shift 3.0`, `mechanism_weight 12.0`, `harmless_shift 4.0`;
`group_seed 2024`; XGBoost/LightGBM with `--min-child-weight 5 --max-depth 3`.

### Main matrices (4 × 100 runs)

```bash
SEEDS20="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
SEEDS10="0 1 2 3 4 5 6 7 8 9"
SEEDS5="0 1 2 3 4"

uv run python scripts/run_experiment_matrix.py \
  --output results/paper_final/matrix_a_xgb --datasets dataset_a --seeds $SEEDS20 \
  --dataset-a-n-samples 5000 --model-type xgboost
uv run python scripts/run_experiment_matrix.py \
  --output results/paper_final/matrix_a_lgbm --datasets dataset_a --seeds $SEEDS20 \
  --dataset-a-n-samples 5000 --model-type lightgbm
uv run python scripts/run_experiment_matrix.py \
  --output results/paper_final/matrix_b_xgb --datasets dataset_b --seeds $SEEDS20 \
  --model-type xgboost --no-dataset-b-include-missing-indicators --min-child-weight 5 --max-depth 3
uv run python scripts/run_experiment_matrix.py \
  --output results/paper_final/matrix_b_lgbm --datasets dataset_b --seeds $SEEDS20 \
  --model-type lightgbm --no-dataset-b-include-missing-indicators --min-child-weight 5 --max-depth 3

for d in a_xgb a_lgbm b_xgb b_lgbm; do
  uv run python scripts/summarize_results.py --input-dir results/paper_final/matrix_${d}
done
```

### G3 — distance by mechanism intensity (Dataset A, S2, seeds 0–9)

```bash
for dist in wasserstein mean kl js; do
  uv run python scripts/run_sensitivity_sweep.py --output results/paper_final/g3_${dist} \
    --factors mechanism_scale --scenarios S2 --seeds $SEEDS10 --n-samples 5000 \
    --gads-distance ${dist} --method GADS
done
```

Merge the four per-distance `sensitivity_results.parquet` files into
`results/paper_final/g3_distance_intensity/g3_distance_intensity_merged.parquet` with an added
`distance` column (240 rows = 4 distances × 6 levels × 10 seeds); that merged table is what
`build_isa_figures.py` and `--fig-types g3` consume.

### G5 — single-factor sensitivity (Dataset A, seeds 0–9)

```bash
# main sweep: six factors, base n = 1000, scenarios S1+S2, GADS / Wasserstein-X / Global-SHAP
uv run python scripts/run_sensitivity_sweep.py --output results/paper_final/g5_sensitivity/all \
  --seeds $SEEDS10 --n-samples 1000 \
  --method GADS --method Wasserstein-X --method Global-SHAP

# n_devices controls keeping 100 rows/device (n_samples = 100 x n_devices)
uv run python scripts/run_sensitivity_sweep.py --output results/paper_final/g5_sensitivity/ndev_ctrl_15 \
  --factors n_devices --levels n_devices=15 --n-samples 1500 --seeds $SEEDS10 \
  --method GADS --method Wasserstein-X --method Global-SHAP
uv run python scripts/run_sensitivity_sweep.py --output results/paper_final/g5_sensitivity/ndev_ctrl_20 \
  --factors n_devices --levels n_devices=20 --n-samples 2000 --seeds $SEEDS10 \
  --method GADS --method Wasserstein-X --method Global-SHAP
```

### G6 — group-seed robustness (Dataset B)

```bash
for gs in 2025 2026 2027 2028; do
  uv run python scripts/run_experiment_matrix.py --output results/paper_final/g6_group_seeds/groupseed_${gs} \
    --datasets dataset_b --scenarios S2 S4-main S4-null --seeds $SEEDS5 \
    --model-type xgboost --no-dataset-b-include-missing-indicators \
    --min-child-weight 5 --max-depth 3 --dataset-b-group-seed ${gs}
done
```

`group_seed 2024` is the reference partition (matrix B seeds 0–4).

### Modern baselines and EQP ablation

```bash
# modern baselines, Dataset A (GADS + M2OE-Group + XPE, default M2OE cap 128)
uv run python scripts/run_experiment_matrix.py --output results/paper_final/modern_baselines_a \
  --datasets dataset_a --seeds $SEEDS20 --dataset-a-n-samples 5000 --model-type xgboost \
  --method GADS --method M2OE-Group --method XPE

# modern baselines, Dataset B per scenario (cap 64)
for scen in S1 S2 S3 S4-main S4-null; do
  uv run python scripts/run_experiment_matrix.py --output results/paper_final/modern_baselines_b/${scen} \
    --datasets dataset_b --scenarios ${scen} --seeds $SEEDS10 \
    --no-dataset-b-include-missing-indicators --min-child-weight 5 --max-depth 3 \
    --model-type xgboost --method GADS --method M2OE-Group --method XPE --m2oe-max-group-rows 64 &
done
wait

# EQP ablation, Dataset B (S2 x seeds 0-4 x three modes, GADS only).
# eqp-matrix is Dataset A only, so run the three modes against Dataset B explicitly:
for mode in contextual_process no_eqp all_features; do
  uv run python scripts/run_experiment_matrix.py --output results/paper_final/eqp_ablation_b/${mode} \
    --datasets dataset_b --scenarios S2 --seeds $SEEDS5 --eqp-mode ${mode} \
    --model-type xgboost --no-dataset-b-include-missing-indicators \
    --min-child-weight 5 --max-depth 3 --method GADS
done
```

Dataset B matrix runs need a background `&` / `wait` only because the loop above launches one
job per scenario; everything else runs sequentially.

### Runtime

Reproduction is **hours, not minutes**, and `summarize_results.py` / `plot_paper_figs.py` are
fast (seconds) once the matrices exist. Single-run wall-clock measured on an Apple M3 (8 cores,
24 GB RAM, Python 3.11, runs sequential):

| Configuration | One run |
| --- | --- |
| Dataset A, n=5000, cv=5, six classical methods, XGBoost | ~32 s |
| Dataset A, n=5000, cv=5, six classical methods, LightGBM | ~14 s |
| Dataset A, n=5000, cv=5, GADS only, XGBoost | ~20 s |
| Dataset B, cv=5, six classical methods, XGBoost (paper B protocol) | ~17 s |

Extrapolated (sequential, same machine):

| Stage | Runs | Estimated |
| --- | --- | --- |
| `matrix_a_xgb` | 100 | ~50 min |
| `matrix_a_lgbm` | 100 | ~25 min |
| `matrix_b_xgb` | 100 | ~30 min |
| `matrix_b_lgbm` | 100 | ~20 min |
| Four main matrices | 400 | **~2 h** |
| G3 distance-by-intensity | 240 | ~1–1.5 h |
| G5 sensitivity (+ controls) | ~470 | ~1 h |
| G6 group-seed robustness | 60 | ~20 min |
| Modern-baseline slices and G2 diagnosis sweeps | hundreds | not timed; add several hours |

The main paper results (four matrices + G3/G5/G6) are therefore on the order of **4–6 h** on a
laptop-class machine, and longer on slower or busier hardware. Independent matrices can be run
in parallel to cut wall-clock time. Set `SEEDS20`/`SEEDS10` to shorter seed lists (and
`--dataset-a-n-samples` to a smaller value) for a fast approximation before a full run.

---

## 10. Licensing and third-party code

- **Project-root license: not yet chosen.** No root `LICENSE` file is shipped with this export;
  the licensing decision is pending.
- `src/shap_diff_analysis/vendor/m2oe/` is a vendored port of the upstream **M2OE** tabular
  group explainer and **keeps its upstream MIT license**
  (`src/shap_diff_analysis/vendor/m2oe/LICENSE`, Copyright (c) 2025 AIDALab-DIMES). The
  `M2OE-Group` baseline uses this code. Do not remove that license file.
- UCI SECOM is redistributed only by download, under its own terms; no raw SECOM data is
  committed.

---

## 11. Tests

```bash
uv run pytest -q
```

The suite covers attribution metrics, scenario generation, SECOM handling, the experiment
matrix, ablations, statistics, and the figure helpers. Tests that need the raw SECOM files are
skipped until `scripts/download_secom.py` has run; four exact-value regression tests are skipped
until the `results/paper_final` archive from a full reproduction exists. Everything else runs
offline.

Code quality: `uv run ruff check . --fix && uv run ruff format .`, `uv run pyright`.

---

## 12. Repository layout

```
assets/fig1_framework.svg            # editable Figure 1 source (vector)
scripts/
  download_secom.py                  # download + SHA256-verify UCI SECOM
  generate_dataset_b.py              # build a Dataset B bundle
  run_paper_experiments.py           # single-bundle run of all/selected methods
  run_experiment_matrix.py           # scenarios x seeds x datasets matrix
  run_ablation.py                    # GADS distance / EQP handling ablations
  run_sensitivity_sweep.py           # single-factor Dataset A sweeps (G3/G5)
  assemble_experiment_matrix.py      # merge partial run directories
  summarize_results.py               # bootstrap CIs, paired tests, Holm correction
  plot_paper_figs.py                 # standard paper figures
  build_isa_figures.py               # four manuscript figure assets
src/shap_diff_analysis/              # method, baselines, metrics, statistics, data generation
tests/                               # pytest suite
```

| Script | Purpose |
| --- | --- |
| `download_secom.py` | download and verify UCI SECOM |
| `generate_dataset_b.py` | generate a Dataset B bundle |
| `run_paper_experiments.py` | single experiment run |
| `run_experiment_matrix.py` | scenario × seed × dataset matrix |
| `run_ablation.py` | distance-function and EQP ablations |
| `run_sensitivity_sweep.py` | G3/G5 single-factor sweeps |
| `assemble_experiment_matrix.py` | assemble a matrix from partial runs |
| `summarize_results.py` | bootstrap CIs, paired tests, Holm correction |
| `plot_paper_figs.py` | main metrics, FAR, seed stability, rank heatmap, trade-off, Fig. 1/2 |
| `build_isa_figures.py` | build the four manuscript assets from an archive |

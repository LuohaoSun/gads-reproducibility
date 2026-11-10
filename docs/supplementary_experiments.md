# Supplementary experiments

This guide covers three supplementary experiment scripts that accompany the revision work:

| Script | Purpose | Fits a model? |
| --- | --- | --- |
| `scripts/bdcc_internal_review_existing_stats.py` | Recompute confidence intervals and paired summaries from existing Pack 1/Pack 2 tables | No |
| `scripts/bdcc_internal_review_nonlinear.py` | Nonlinear-label robustness benchmark on Dataset A | Yes |
| `scripts/bdcc_internal_review_null_weak.py` | S0-clean null calibration and weak Dataset B fault-intensity sweep | Yes |

The repository is source-only for these experiments. It ships the current runnable scripts and their
tests, not the supplementary result archive, the raw SECOM files, the manuscript, reviewer
correspondence, or internal review material. The commands below generate outputs locally; this
guide does not claim that the archived supplementary tables have been re-run in this checkout.

---

## 1. Environment and prerequisites

```bash
uv sync --all-extras
```

Python is constrained to `>=3.11,<3.12` by `pyproject.toml`.

Additional prerequisites depend on the experiment:

| Experiment | Required input |
| --- | --- |
| `existing_stats` | The Pack 1 and Pack 2 output tables under `results/bdcc_revision/` (not shipped) |
| `nonlinear` | None beyond the installed environment; Dataset A is generated in process |
| `null_weak` | Official UCI SECOM files in `data/raw/secom` (not shipped) |

Download and verify SECOM once when it is needed:

```bash
uv run python scripts/download_secom.py
```

`download_secom.py` verifies the pinned SHA-256 digests and fails on a mismatch. `null_weak` also
verifies the files before doing any work, and `null_weak` never writes to the raw directory.

### 1.1 Why `existing_stats` cannot run on a clean clone

`existing_stats` reads exactly nine existing tables. Eight come from Pack 1 (five under
`model_seed_sweep/` and three under `hyperparameter_grid/`); the ninth is Pack 2's
`comparison_per_seed.csv`:

```
results/bdcc_revision/pack1_seeds_hyperparams/
  model_seed_sweep/metrics_long.csv
  model_seed_sweep/absolute_precision_ci.csv
  model_seed_sweep/variance_decomposition.csv
  model_seed_sweep/rank_correlation_model_seeds.csv
  model_seed_sweep/rank_correlation_model_seeds_summary.csv
  hyperparameter_grid/metrics_long.csv
  hyperparameter_grid/missing_indicators_comparison.csv
  hyperparameter_grid/wasserstein_x_scale_paired_diff.csv
results/bdcc_revision/pack2_competitors_runtime/comparison_per_seed.csv
```

None of these files is committed, so a clean clone has no input for this script. Generate them with
the same public scripts that own those directories. The defaults below already point at the required
paths:

```bash
# Pack 1: 2 datasets x 5 scenarios x 20 scenario seeds x 5 model seeds.
# Needs verified SECOM data for Dataset B.
uv run python scripts/bdcc_pack1_seeds_and_hyperparams.py \
  --output-dir results/bdcc_revision/pack1_seeds_hyperparams

# Pack 2 comparison + glm-extra stages: 2 datasets x 5 scenarios x 20 seeds across the
# competitor set, at model seed 42. The comparison stage writes the base grid; glm-extra
# adds the L1 C=0.5 ranking to every stored run, giving the complete nine-method table.
uv run python scripts/bdcc_pack2_competitors_and_runtime.py \
  --output results/bdcc_revision/pack2_competitors_runtime \
  --stages comparison glm-extra
```

These are full experiment runs, not a smoke test: Pack 1 schedules 1,000 run tasks and Pack 2 runs
the full comparison grid (2 datasets x 5 scenarios x 20 seeds across its competitor set). Their cost
dominates this prerequisite chain. Pack 1 accepts
`--workers` in the range 1-6; Pack 2 defaults to six comparison workers. If you already have the
Pack 1/Pack 2 directories from an earlier run, pass their location through `--input-root` below
instead of regenerating them.

---

## 2. Existing-statistics recomputation

```bash
uv run python scripts/bdcc_internal_review_existing_stats.py \
  --input-root . \
  --output-dir results/bdcc_internal_review/existing_stats
```

`--input-root` is the project root that contains `results/bdcc_revision/`; relative output paths
resolve against it. The script does not train or rerun anything. A missing file, a missing column,
or an empty table raises immediately; it never falls back to a different input.

### Statistical unit

The statistical unit is the `scenario_seed` cluster (n = 20), not the individual run. The five model
seeds `(42, 101, 202, 303, 404)` are averaged inside each scenario cluster first, and 10,000
bootstrap resamples are then drawn over the 20 cluster means. The 100 model-seed x scenario-seed
runs are not treated as independent. Intervals are 95% BCa (`scipy.stats.bootstrap` with a mid-rank
tie correction). Constant cells are reported as degenerate rather than given an artificial interval.
The five model seeds are fixed; the bootstrap does not sample a larger model-seed distribution.

Outputs under the chosen `--output-dir`:

| File | Content |
| --- | --- |
| `gads_absolute_ci.csv` | GADS absolute-precision BCa intervals |
| `gads_variance_decomposition.csv` | between / (between + within) variance shares |
| `spearman_ci.csv` | model-seed rank-correlation intervals |
| `pack2_ci.csv` | competitor intervals, with a `ci_source` column |
| `hyperparameter_overall_ci.csv` | hyperparameter-grid summaries over all scenarios |
| `hyperparameter_per_scenario_ci.csv` | the same grid per scenario |
| `missing_indicator_ci.csv` | missing-indicator comparison |
| `wasserstein_x_scale.csv` | raw versus normal-pool-standardized Wasserstein-X paired differences |
| `summary.md`, `protocol.json`, `verification.json` | human summary, input hashes and protocol, verification record |

Inapplicable metric cells are marked as not applicable, never silently zeroed.

---

## 3. Nonlinear-label robustness benchmark (RSM v2)

### 3.1 Design

The script borrows the Dataset A feature matrix, EQP assignment, scenario injections and ground-truth
manifest from `generate_dataset_bundle`, then replaces only the label-generating conditional
probability. The raw feature marginals and the scenario contract are unchanged.

The normal logit is

```
eta = -2.5
      + sum_{j=1..10} w_j * tanh(1.2 * x_j)
      + 0.35 * tanh(0.8 * x_1) * tanh(0.8 * x_2)
      + sum_s w_s * tanh(c_s * x_s / sigma_s)
```

with useful-feature weights
`(0.90, 1.00, 1.10, 0.85, 1.05, 0.95, 1.15, 0.80, 1.00, 0.90)`, special-feature response pairs
`feat_special_dist = (1.00, 0.60)`, `feat_special_imp = (1.60, 1.00)`,
`feat_special_mix = (1.40, 0.90)`, and generation scales
`sigma = {dist: 1.0, imp: 2.0, mix: 1.0}`. The pairwise term makes the label non-additive as well as
nonlinear.

For a scenario-designated mechanism fault, the affected device's response gains the regime-coupled
bilinear term

```
2.0 * x_load * (x_s / sigma_s),    x_load = feat_useful_1
```

`x_load` (`feat_useful_1`) is a background operating-regime covariate, not an injected root cause.
The special feature `x_s` is the injected root. This is a statistical benchmark mechanism, not a
physical claim.

Scenario use of the mechanism term:

| Scenario | Injected root and label perturbation |
| --- | --- |
| `S1` | `EQP_A` / `feat_special_dist`; the original generator's physical location shift only, no bilinear term |
| `S2` | `EQP_B` / `feat_special_imp`; bilinear mechanism term |
| `S3` | `EQP_A` / `feat_special_dist` physical shift; `EQP_B` / `feat_special_imp` and `EQP_C` / `feat_special_mix` bilinear mechanism terms; `EQP_C` also keeps the original physical shift |
| `S4-main` | `EQP_D` / `feat_special_imp` bilinear mechanism term, plus ten harmless drift features that never enter the label |
| `S4-null` | no root cause; ten harmless drift features; root-cause metrics are not applicable |

The label is `Bernoulli(expit(eta))` with RNG tag `20261109`, derived from the scenario seed. No
additional Gaussian label noise is added. The manifest's `root_causes` and `harmless_features` are
the original scenario contract and validate the evaluation.

### 3.2 Runs and statistics

The full design is five scenarios x 20 scenario seeds = 100 datasets and three methods
(`GADS`, `GLM-Int-Joint-L2`, `GAM-Int`) = 300 method evaluations. GADS uses cross-fitted XGBoost
SHAP and Wasserstein distance on the `process` view; the competitors are a joint L2 interaction GLM
and an additive-spline interaction GAM.

Full run:

```bash
uv run python scripts/bdcc_internal_review_nonlinear.py \
  --output results/bdcc_internal_review/nonlinear/full \
  --jobs 4
```

`--jobs` controls process-level parallelism; each worker fits its own models. Keep the total below
the physical core count and available RAM. The full run writes per-scenario/per-seed directories
under `runs/` and then aggregates. A rerun with the same `--output` reuses `runs/*/seed_*/metrics.csv`
unless `--overwrite` is passed.

Recompute summaries without retraining, after a completed run:

```bash
uv run python scripts/bdcc_internal_review_nonlinear.py \
  --output results/bdcc_internal_review/nonlinear/full \
  --aggregate-only
```

`--aggregate-only` requires the per-run `runs/*/seed_*/metrics.csv` files; it fails if the scenario
x seed grid is incomplete. It reads only those per-run tables and does not need SECOM.
`training_source.json` is optional provenance: when it is absent, the script falls back to the
SHA-256 recorded in `run_manifest.json`.

A local end-to-end validation (five datasets, fifteen method evaluations, no confidence interval):

```bash
uv run python scripts/bdcc_internal_review_nonlinear.py \
  --output /tmp/nonlinear_smoke \
  --scenarios S1 S2 S3 S4-main S4-null --seeds 2 --n-samples 800 --jobs 1
```

The completeness check requires every scenario name in `SCENARIOS`, so all five scenarios must be
requested even though only S1, S2, S3 and S4-main enter the pooled summary. At least one seed must be
outside the pilot set (0 and 1); a single seed runs end to end but has no BCa interval. Reducing the
seed count or `--n-samples` is safe for a smoke test, but reducing the scenario set is not:
`--scenarios` with a subset, or any `--max-runs` truncation, trains a partial cohort that cannot be
aggregated.

Per scenario and metric, the script reports the mean, sample SD and a 95% BCa interval over seeds
(10,000 resamples). GADS-versus-competitor differences are paired by seed and summarized with the
same procedure. A pooled `POOLED_S1_S4M` scope averages MRR/HR@1/HR@5 equally over S1, S2, S3 and
S4-main within each `(seed, method)` cell before the BCa summary. Metrics whose manifest is empty are
`applicable=False` with `n=0` and NaN summaries, never zero.

Seeds 0 and 1 were inspected while specifying the v2 label form. The full 20-seed result remains the
primary summary. The script additionally writes seeds 1-19 and pilot-excluded seeds 2-19 as
secondary sensitivity tables; these do not replace the primary result.

### 3.3 Outputs

| File | Content |
| --- | --- |
| `raw_metrics.csv` | one row per scenario x seed x method |
| `summary_metrics.csv` | per-scenario metric summaries and BCa intervals |
| `paired_differences.csv` | paired GADS-minus-competitor differences |
| `sensitivity_seeds_1_19.csv` | secondary summary over seeds 1-19 |
| `sensitivity_seeds_2_19.csv` | secondary summary excluding pilot seeds 0 and 1 |
| `pooled_summary_metrics.csv` | equal-weight S1-S4-main pooled summaries |
| `pooled_paired_differences.csv` | pooled paired differences |
| `pooled_summary_metrics_seeds_2_19.csv` | pooled summary excluding pilot seeds |
| `diagnostics.csv` | per-run linearity, class-balance, timing and OOF AUC diagnostics |
| `aggregation_provenance.json`, `protocol.json`, `run_manifest.json` | aggregation record, settings and run environment |

---

## 4. S0-clean null calibration and weak Dataset B intensity

### 4.1 What the two blocks do

`S0-clean` is a deliberately fault-free calibration: the base generator is run with no physical,
mechanism or harmless shift. The script runs a fresh cohort of 200 scenario seeds (`0..199`) for
Dataset A and Dataset B, in both the `standard` and within-EQP `centered` views, with 500 permutations
per cell. The reported event is `observed_max > q95_max`, where `q95_max` is the 95th percentile of
the permutation maximum over all process features. The script reports the event rate with a Wilson
95% interval and the across-seed distribution of the thresholds.

Two different S0 summaries are reported and must not be conflated:

| Summary | Threshold | Meaning |
| --- | --- | --- |
| `n_exceed_95`, `exceedance_rate_95`, `fwer_reject_95` | common max-statistic cutoff `q95_max` | process-level: the observed maximum exceeds one common cutoff |
| `n_features_exceed_tau95`, `mean_features_exceed_tau95`, `per_feature_fpr_95_mean` | feature-specific pointwise cutoff `tau95` | feature-level: how many features exceed their own pointwise 95th percentile |

Zero process-level exceedances does not imply zero feature-level pointwise exceedances.

The `weak_b` block is a Dataset B S2 mechanism local-weight sweep at
`0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5` with 20 scenario seeds (`0..19`). The `0.2` level equals the
normal-pool global coefficient and is the no-fault endpoint: its root manifest must be empty and it
is excluded from the MRR/HR grid. Levels `0.4`-`1.5` are injected mechanism faults and are the only
levels summarized as metrics. Only the standard view is fitted in this block.

All means, SDs and BCa intervals are computed over scenario seeds, not rows or equipment. The random
ranking reference `H_590 / 590` is stored as a theoretical reference only; the script does not turn a
small distance from it into a claim of indistinguishability without a formal test.

### 4.2 Commands

Inspect the task grid without fitting (argument validation still verifies the SECOM files):

```bash
uv run python scripts/bdcc_internal_review_null_weak.py \
  --raw-dir data/raw/secom \
  --dry-run
```

Full run (both blocks):

```bash
uv run python scripts/bdcc_internal_review_null_weak.py \
  --raw-dir data/raw/secom \
  --output-dir results/bdcc_internal_review/null_weak \
  --workers 8
```

The S0-clean block is the heavy part of this command: 200 scenario seeds x 2 datasets x 2 views with
500 permutations per cell. Use `--part`, `--s0-seeds`/`--weak-seeds` and `--n-permutations` for a local
check before committing to the full cohort.

Recompute summaries from the partials written by a full or partial run:

```bash
uv run python scripts/bdcc_internal_review_null_weak.py \
  --raw-dir data/raw/secom \
  --output-dir results/bdcc_internal_review/null_weak \
  --aggregate-only
```

`--aggregate-only` requires `s0_clean/partials/` and `weak_b/partials/` under `--output-dir` (or
`--partials-root`); `--partials-root` defaults to `--output-dir`. Those intermediate files are not included in this
source repository; run the full command first if they are absent. Although it fitsnothing, `--aggregate-only` still calls `verify_secom_files` at startup, so `--raw-dir` must point at
the verified SECOM files. `--aggregate-only` cannot be combined with `--force`.
`--pack3-partials` is only a read-only cross-check and does not supply the partials used for
aggregation.

Small local checks. Both commands validate the SECOM files at startup, even the Dataset A-only S0
run, so `--raw-dir` must point at the verified files:

```bash
# S0-clean: two scenario seeds x both views for Dataset A (four cells).
uv run python scripts/bdcc_internal_review_null_weak.py \
  --part s0 --datasets a --s0-seeds 0:2 --n-permutations 100 \
  --raw-dir data/raw/secom --output-dir /tmp/null_weak_smoke_s0

# Weak: one evaluation seed x all seven intensity levels (including the 0.2 endpoint) for Dataset B.
uv run python scripts/bdcc_internal_review_null_weak.py \
  --part weak --datasets b --weak-seeds 0:1 --raw-dir data/raw/secom \
  --output-dir /tmp/null_weak_smoke_weak
```

`--n-permutations` must be at least 100 and `--n-boot` at least 100. The default permutation count is
500. `--force` reruns every task; without it, existing partials are reused. `--pack3-partials` is an
optional read-only compatibility check against a Pack 3 partial directory and is not required for
the standalone results; if it is omitted, the manifest records the check as `not_checked`.

### 4.3 Outputs

Under `s0_clean/`:

| File | Content |
| --- | --- |
| `cells.csv` | per-seed S0 cells, max-statistic and pointwise feature counts |
| `exceedance_summary.csv` | max-statistic exceedance with Wilson intervals and pointwise averages |
| `threshold_distribution.csv` | distribution of `observed_max`, `q95_max`, `q99_max` and `p_value_max` |
| `per_feature_fpr_summary.csv` | per-feature pointwise false-positive rate summary |
| `compatibility_pack3_seeds_0_19.csv` | only when `--pack3-partials` is supplied |
| `manifest.json` | part status, protocol and source record |

Under `weak_b/`:

| File | Content |
| --- | --- |
| `no_fault_endpoint.csv` | weight-0.2 endpoint validation (empty root manifest, zero effect) |
| `metrics_per_seed.csv` | per-seed metrics at weights 0.4-1.5 |
| `metric_summary.csv` | metric summaries and BCa intervals over seeds |
| `manifest.json` | part status, protocol and source record |

The root `run_manifest.json` reports overall status and both part statuses. Completion is judged from
that manifest and the two part manifests, not from `run_status.json` progress files.

---

## 5. Targeted tests

```bash
uv run pytest -q tests/test_bdcc_internal_review_nonlinear.py \
  tests/test_bdcc_internal_review_null_weak.py
```

The nonlinear tests validate the scenario contract, the unchanged S2 raw marginal with a changed
conditional probability, the exclusion of harmless features from the label, the bilinear term's zero
population linear slope, and the BCa helpers. The null/weak tests validate the no-fault endpoint and
metric-grid exclusion, the 200-seed S0 grid, the environment manifest, the random-MRR reference and
the mid-rank BCa correction. The two null/weak tests that require SECOM skip with an explicit
message when `data/raw/secom/secom.data` is absent. None of these tests fits a full experiment.

---

## 6. Provenance and scope

The repository provides the current runnable version of these scripts. It does not ship a
byte-identical historical training snapshot or the original training-time environment. A run
manifest records the SHA-256 of the script file that created that manifest. In the nonlinear script,
`aggregation_provenance.json` records the two stages separately: `training.script_sha256` comes from
`training_source.json` when present, otherwise from the existing `run_manifest.json`, while
`aggregation.script_sha256` is the hash of the script performing the aggregation. An
`--aggregate-only` run therefore preserves the original training hash and records the current
aggregation hash; the two are equal only when training and aggregation ran from the same checkout.
Treat these records as provenance for the local run, not as a claim that the run used a specific
archived binary or byte-identical historical file.

The repository contains no supplementary result tables, no raw SECOM data, no manuscript sources and
no reviewer correspondence. The commands above produce outputs locally; they do not download the
supplementary result archive and they do not assert that the archived numbers have already been
reproduced here.

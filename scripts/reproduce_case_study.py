#!/usr/bin/env python3
"""Deterministically reproduce the manuscript mechanism-shift case study.

The manuscript reports, for Dataset A / scenario S2 / seed 0 under the
contextual-process protocol (XGBoost, K=5 stratified cross-fitting):

* raw ``feat_special_imp`` KS statistic ``0.0549`` with ``p = 0.1290``
  (500 ``EQP_B`` versus 4500 normal-pool observations),
* out-of-fold (OOF) SHAP standard deviations ``1.2406`` (abnormal) and
  ``0.3936`` (normal pool), both with ``ddof=1``,
* Wasserstein distance ``0.7966`` between the abnormal and normal OOF SHAP
  distributions,
* GADS ranks ``feat_special_imp`` first for ``EQP_B``.

The values are recomputed here from the repository's own generator and model
entry points rather than read from the archived aggregate parquet (which does
not carry the per-sample SHAP arrays).  The script prints a reproduced-versus-
manuscript comparison table and exits with a non-zero status when any value
falls outside the manuscript's quoted precision.  It writes nothing to disk and
never modifies the manuscript, bibliography or figures.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.gads import rank_gads_features
from shap_diff_analysis.generate_data import generate_dataset_bundle
from shap_diff_analysis.modeling import CrossFittedAttributions, ModelType, fit_cross_fitted_model

DATASET = "Dataset A"
SCENARIO = "S2"
SEED = 0
N_SAMPLES = 5000
NOISE_LEVEL = 0.2
N_DEVICES = 10
DISTRIBUTION_SHIFT = 3.0
MECHANISM_SCALE = 1.0
HARMLESS_DRIFT = 3.5
MODEL_TYPE: ModelType = "xgboost"
MODEL_SEED = 42
CV_SEED = 42
CV_SPLITS = 5
CASE_STUDY_FEATURE = "feat_special_imp"
ABNORMAL_DEVICE = "EQP_B"

# The manuscript quotes four decimal places, so a faithful reproduction must
# agree to within half a unit in the last quoted digit.
QUOTED_DECIMALS = 4
ABS_TOLERANCE = 0.5 * 10.0**-QUOTED_DECIMALS


@dataclass(frozen=True)
class PaperCaseStudyValues:
    """Manuscript values for the mechanism-shift case study."""

    n_abnormal: int
    n_normal: int
    ks_statistic: float
    ks_pvalue: float
    abnormal_shap_std: float
    normal_shap_std: float
    wasserstein: float
    gads_rank: int


PAPER_VALUES = PaperCaseStudyValues(
    n_abnormal=500,
    n_normal=4500,
    ks_statistic=0.0549,
    ks_pvalue=0.1290,
    abnormal_shap_std=1.2406,
    normal_shap_std=0.3936,
    wasserstein=0.7966,
    gads_rank=1,
)


@dataclass(frozen=True)
class CaseStudyStatistics:
    """Computed case-study quantities plus non-compared diagnostics."""

    n_abnormal: int
    n_normal: int
    ks_statistic: float
    ks_pvalue: float
    abnormal_shap_std: float
    normal_shap_std: float
    wasserstein: float
    gads_rank: int
    gads_score: float
    mean_auc: float


@dataclass(frozen=True)
class ComparisonRow:
    """One reproduced-versus-manuscript quantity."""

    metric: str
    reproduced: float
    paper: float
    tolerance: float
    decimals: int = QUOTED_DECIMALS

    @property
    def difference(self) -> float:
        """Return the signed reproduced-minus-manuscript difference."""
        return self.reproduced - self.paper

    @property
    def matches(self) -> bool:
        """Return whether the difference is inside the stated tolerance."""
        return abs(self.difference) <= self.tolerance

    def format_row(self) -> str:
        """Render this comparison as one Markdown table row."""
        return (
            f"| {self.metric} | {_format_value(self.reproduced, self.decimals)} "
            f"| {_format_value(self.paper, self.decimals)} "
            f"| {_format_signed(self.difference, self.decimals)} "
            f"| {self.tolerance:.0e} | {'OK' if self.matches else 'MISMATCH'} |"
        )


def _format_value(value: float, decimals: int) -> str:
    return f"{value:.0f}" if decimals == 0 else f"{value:.{decimals}f}"


def _format_signed(value: float, decimals: int) -> str:
    return f"{value:+.0f}" if decimals == 0 else f"{value:+.{decimals}f}"


def build_case_study_bundle(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    scenario: str = SCENARIO,
    noise_level: float = NOISE_LEVEL,
    n_devices: int = N_DEVICES,
    distribution_shift: float = DISTRIBUTION_SHIFT,
    mechanism_scale: float = MECHANISM_SCALE,
    harmless_drift: float = HARMLESS_DRIFT,
) -> DatasetBundle:
    """Generate the S2 bundle through the repository's canonical entry point.

    The arguments default to the case-study protocol; tests pass a smaller
    ``n_samples`` to keep the fixture cheap.  Any deviation from the expected
    scenario contract is raised rather than silently accepted.
    """
    bundle = generate_dataset_bundle(
        scenario=scenario,
        n_samples=n_samples,
        seed=seed,
        noise_level=noise_level,
        n_devices=n_devices,
        distribution_shift=distribution_shift,
        mechanism_scale=mechanism_scale,
        harmless_drift=harmless_drift,
    )
    if bundle.scenario != scenario:
        raise ValueError(f"generated scenario {bundle.scenario!r} does not match requested {scenario!r}")
    if bundle.seed != seed:
        raise ValueError(f"generated seed {bundle.seed} does not match requested {seed}")
    if ABNORMAL_DEVICE not in bundle.abnormal_devices:
        raise ValueError(f"scenario {scenario!r} does not mark {ABNORMAL_DEVICE!r} abnormal")
    if CASE_STUDY_FEATURE not in bundle.process_features:
        raise ValueError(f"process features do not contain the case-study feature {CASE_STUDY_FEATURE!r}")
    return bundle


def fit_oof_attributions(
    bundle: DatasetBundle,
    *,
    model_type: ModelType = MODEL_TYPE,
    model_seed: int = MODEL_SEED,
    cv_seed: int = CV_SEED,
    cv_splits: int = CV_SPLITS,
) -> CrossFittedAttributions:
    """Fit the contextual-process KPI model and return its OOF SHAP values.

    The equipment identifier is kept as a contextual covariate (the paper's
    ``contextual-process`` protocol); the GADS ranking later restricts the
    scoring surface to ``bundle.process_features``.
    """
    feature_columns = [*bundle.process_features, bundle.group_column]
    X = bundle.data.loc[:, feature_columns]
    y = cast(pd.Series, pd.to_numeric(bundle.data[bundle.target_column], errors="raise")).astype(int)
    return fit_cross_fitted_model(
        X,
        y,
        model_type=model_type,
        model_seed=model_seed,
        cv_seed=cv_seed,
        cv_splits=cv_splits,
    )


def compute_case_study_statistics(
    bundle: DatasetBundle,
    attributions: CrossFittedAttributions,
    *,
    feature: str = CASE_STUDY_FEATURE,
    abnormal_device: str = ABNORMAL_DEVICE,
) -> CaseStudyStatistics:
    """Compute the KS, OOF SHAP standard deviation, Wasserstein and GADS values."""
    if feature not in bundle.process_features:
        raise ValueError(f"feature {feature!r} is not a process feature of the bundle")
    if abnormal_device not in bundle.abnormal_devices:
        raise ValueError(f"device {abnormal_device!r} is not an abnormal device of the bundle")

    groups = cast(pd.Series, bundle.data[bundle.group_column]).astype(str)
    abnormal_mask = groups == abnormal_device
    normal_mask = groups.isin(bundle.normal_devices)
    if not abnormal_mask.any():
        raise ValueError(f"abnormal device {abnormal_device!r} has no observations")
    if not normal_mask.any():
        raise ValueError("normal pool has no observations")

    raw_abnormal = bundle.data.loc[abnormal_mask, feature].to_numpy(dtype=float)
    raw_normal = bundle.data.loc[normal_mask, feature].to_numpy(dtype=float)
    # scipy's axis/nan-policy wrapper erases the KstestResult return type, so the
    # statistic/pvalue attributes are recovered with the repo's established cast.
    ks_result = cast(Any, ks_2samp(raw_abnormal, raw_normal))

    shap_abnormal = attributions.shap_values.loc[abnormal_mask, feature].to_numpy(dtype=float)
    shap_normal = attributions.shap_values.loc[normal_mask, feature].to_numpy(dtype=float)

    ranking = rank_gads_features(
        attributions.shap_values,
        groups,
        bundle.abnormal_devices,
        bundle.normal_devices,
        process_features=bundle.process_features,
        view="process",
        distance="wasserstein",
    )
    selected = ranking[(ranking["equipment"] == abnormal_device) & (ranking["feature"] == feature)]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one GADS row for {abnormal_device}/{feature}, got {len(selected)}")

    return CaseStudyStatistics(
        n_abnormal=int(abnormal_mask.sum()),
        n_normal=int(normal_mask.sum()),
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        abnormal_shap_std=float(np.std(shap_abnormal, ddof=1)),
        normal_shap_std=float(np.std(shap_normal, ddof=1)),
        wasserstein=float(wasserstein_distance(shap_abnormal, shap_normal)),
        gads_rank=int(cast(pd.Series, selected["rank"]).iloc[0]),
        gads_score=float(cast(pd.Series, selected["score"]).iloc[0]),
        mean_auc=attributions.mean_auc,
    )


def run_case_study(
    *,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    cv_splits: int = CV_SPLITS,
    cv_seed: int = CV_SEED,
    model_seed: int = MODEL_SEED,
) -> CaseStudyStatistics:
    """Generate the bundle, fit the OOF model and compute all quantities."""
    bundle = build_case_study_bundle(n_samples=n_samples, seed=seed)
    attributions = fit_oof_attributions(
        bundle,
        model_seed=model_seed,
        cv_seed=cv_seed,
        cv_splits=cv_splits,
    )
    return compute_case_study_statistics(bundle, attributions)


def comparison_rows(
    statistics: CaseStudyStatistics,
    *,
    paper: PaperCaseStudyValues = PAPER_VALUES,
    tolerance: float = ABS_TOLERANCE,
) -> list[ComparisonRow]:
    """Pair every reproduced quantity with its manuscript value."""
    return [
        ComparisonRow("n abnormal observations", float(statistics.n_abnormal), float(paper.n_abnormal), 0.0, 0),
        ComparisonRow("n normal observations", float(statistics.n_normal), float(paper.n_normal), 0.0, 0),
        ComparisonRow("KS statistic (raw)", statistics.ks_statistic, paper.ks_statistic, tolerance),
        ComparisonRow("KS p-value (raw)", statistics.ks_pvalue, paper.ks_pvalue, tolerance),
        ComparisonRow("SHAP std (abnormal)", statistics.abnormal_shap_std, paper.abnormal_shap_std, tolerance),
        ComparisonRow("SHAP std (normal pool)", statistics.normal_shap_std, paper.normal_shap_std, tolerance),
        ComparisonRow("Wasserstein distance (SHAP)", statistics.wasserstein, paper.wasserstein, tolerance),
        ComparisonRow("GADS rank (EQP_B)", float(statistics.gads_rank), float(paper.gads_rank), 0.0, 0),
    ]


def mismatched_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    """Return only the comparison rows that fall outside tolerance."""
    return [row for row in rows if not row.matches]


def format_comparison_table(rows: list[ComparisonRow]) -> str:
    """Render the comparison rows as a Markdown table."""
    header = [
        "| Quantity | Reproduced | Manuscript | Difference | Tolerance | Status |",
        "| :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    return "\n".join([*header, *(row.format_row() for row in rows)])


def report_case_study(
    statistics: CaseStudyStatistics,
    *,
    paper: PaperCaseStudyValues = PAPER_VALUES,
    tolerance: float = ABS_TOLERANCE,
) -> int:
    """Print the comparison table and return 0 on full reproduction, else 1."""
    rows = comparison_rows(statistics, paper=paper, tolerance=tolerance)
    failures = mismatched_rows(rows)
    print(format_comparison_table(rows))
    print()
    print(
        f"Diagnostics (not compared): OOF mean AUC = {statistics.mean_auc:.6f}; "
        f"GADS Wasserstein score = {statistics.gads_score:.6f}"
    )
    if failures:
        print()
        print(f"REPRODUCTION FAILED: {len(failures)} of {len(rows)} quantities are outside tolerance.")
        for row in failures:
            print(f"  - {row.metric}: reproduced {row.reproduced!r}, manuscript {row.paper!r}")
        return 1
    print()
    print(f"REPRODUCTION OK: all {len(rows)} quantities agree with the manuscript values.")
    return 0


def main() -> int:
    """Run the full n=5000 case study and compare it with the manuscript."""
    print("Case study reproduction: Dataset A | scenario S2 | seed 0 | XGBoost | contextual-process")
    print(
        f"  generator: generate_dataset_bundle(scenario='S2', n_samples={N_SAMPLES}, seed={SEED}, "
        f"noise_level={NOISE_LEVEL}, n_devices={N_DEVICES}, distribution_shift={DISTRIBUTION_SHIFT}, "
        f"mechanism_scale={MECHANISM_SCALE}, harmless_drift={HARMLESS_DRIFT})"
    )
    print(
        f"  protocol : model_seed={MODEL_SEED}, cv_seed={CV_SEED}, "
        f"StratifiedKFold(n_splits={CV_SPLITS}); process features plus the EQP covariate"
    )
    print()
    statistics = run_case_study()
    return report_case_study(statistics)


if __name__ == "__main__":
    sys.exit(main())

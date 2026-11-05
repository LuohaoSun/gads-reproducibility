"""Tests for the deterministic case-study reproduction script.

The heavy n=5000 run belongs to the script itself; these tests exercise the
same functions at a small scale (n=400, 3 folds) to prove they are runnable and
that the computed statistics are internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from shap_diff_analysis.experiment_types import DatasetBundle
from shap_diff_analysis.modeling import CrossFittedAttributions

if TYPE_CHECKING:
    from scripts.reproduce_case_study import CaseStudyStatistics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

reproduce_case_study: ModuleType = importlib.import_module("scripts.reproduce_case_study")

SMALL_N_SAMPLES = 400
SMALL_CV_SPLITS = 3
SMALL_SEED = 0


@dataclass(frozen=True)
class SmallCaseStudy:
    """Small-scale bundle, OOF attributions and derived statistics."""

    bundle: DatasetBundle
    attributions: CrossFittedAttributions
    statistics: CaseStudyStatistics


@pytest.fixture(scope="module")
def small_case_study() -> SmallCaseStudy:
    """Build one small-scale case study shared by the assertions below."""
    bundle = reproduce_case_study.build_case_study_bundle(n_samples=SMALL_N_SAMPLES, seed=SMALL_SEED)
    attributions = reproduce_case_study.fit_oof_attributions(
        bundle,
        model_seed=SMALL_SEED,
        cv_seed=SMALL_SEED,
        cv_splits=SMALL_CV_SPLITS,
    )
    statistics = reproduce_case_study.compute_case_study_statistics(bundle, attributions)
    return SmallCaseStudy(bundle=bundle, attributions=attributions, statistics=statistics)


def test_paper_values_are_the_manuscript_numbers() -> None:
    """The constants must hold exactly the manuscript's reported values."""
    paper = reproduce_case_study.PAPER_VALUES
    assert paper.n_abnormal == 500
    assert paper.n_normal == 4500
    assert paper.ks_statistic == 0.0549
    assert paper.ks_pvalue == 0.1290
    assert paper.abnormal_shap_std == 1.2406
    assert paper.normal_shap_std == 0.3936
    assert paper.wasserstein == 0.7966
    assert paper.gads_rank == 1


def test_small_bundle_follows_the_s2_contract(small_case_study: SmallCaseStudy) -> None:
    """The bundled generator must yield the S2 abnormal/normal partition."""
    bundle = small_case_study.bundle
    assert bundle.scenario == "S2"
    assert bundle.seed == SMALL_SEED
    assert bundle.abnormal_devices == ("EQP_B",)
    assert "feat_special_imp" in bundle.process_features
    counts = bundle.data["EQP"].astype(str).value_counts()
    assert counts["EQP_B"] == SMALL_N_SAMPLES // 10
    assert counts.sum() == SMALL_N_SAMPLES


def test_small_statistics_are_self_consistent(small_case_study: SmallCaseStudy) -> None:
    """Every statistic must be in range and mutually consistent."""
    statistics = small_case_study.statistics
    bundle = small_case_study.bundle
    attributions = small_case_study.attributions

    assert attributions.shap_values.shape == (SMALL_N_SAMPLES, len(bundle.process_features) + 1)
    assert not attributions.shap_values.isna().to_numpy().any()

    assert statistics.n_abnormal + statistics.n_normal == SMALL_N_SAMPLES
    assert 0.0 <= statistics.ks_statistic <= 1.0
    assert 0.0 <= statistics.ks_pvalue <= 1.0
    assert statistics.abnormal_shap_std > 0.0
    assert statistics.normal_shap_std > 0.0
    assert statistics.wasserstein > 0.0
    assert 0.0 <= statistics.mean_auc <= 1.0
    assert 1 <= statistics.gads_rank <= len(bundle.process_features)
    # GADS scores the same normal-pool-versus-device SHAP comparison, so the
    # selected feature's GADS score must equal its Wasserstein distance.
    assert statistics.gads_score == pytest.approx(statistics.wasserstein, rel=1e-12, abs=1e-12)


def test_small_run_orchestrator_matches_direct_calls(small_case_study: SmallCaseStudy) -> None:
    """``run_case_study`` must reproduce the fixture statistics exactly."""
    recomputed = reproduce_case_study.run_case_study(
        n_samples=SMALL_N_SAMPLES,
        seed=SMALL_SEED,
        cv_splits=SMALL_CV_SPLITS,
        cv_seed=SMALL_SEED,
        model_seed=SMALL_SEED,
    )
    assert recomputed == small_case_study.statistics


def test_bundle_generation_is_deterministic() -> None:
    """Two generations of the same (scenario, seed) must be identical."""
    first = reproduce_case_study.build_case_study_bundle(n_samples=SMALL_N_SAMPLES, seed=SMALL_SEED)
    second = reproduce_case_study.build_case_study_bundle(n_samples=SMALL_N_SAMPLES, seed=SMALL_SEED)
    assert first.data.equals(second.data)


def test_comparison_rows_accept_exact_manuscript_values() -> None:
    """Matching values must produce no mismatches."""
    exact = reproduce_case_study.CaseStudyStatistics(
        n_abnormal=500,
        n_normal=4500,
        ks_statistic=0.0549,
        ks_pvalue=0.1290,
        abnormal_shap_std=1.2406,
        normal_shap_std=0.3936,
        wasserstein=0.7966,
        gads_rank=1,
        gads_score=0.7966,
        mean_auc=0.9,
    )
    rows = reproduce_case_study.comparison_rows(exact)
    assert reproduce_case_study.mismatched_rows(rows) == []
    assert reproduce_case_study.report_case_study(exact) == 0


def test_report_fails_loudly_on_mismatch() -> None:
    """A wrong value must be flagged and yield a non-zero exit status."""
    wrong = reproduce_case_study.CaseStudyStatistics(
        n_abnormal=500,
        n_normal=4500,
        ks_statistic=0.1549,
        ks_pvalue=0.1290,
        abnormal_shap_std=1.2406,
        normal_shap_std=0.3936,
        wasserstein=0.7966,
        gads_rank=2,
        gads_score=0.7966,
        mean_auc=0.9,
    )
    rows = reproduce_case_study.comparison_rows(wrong)
    failures = reproduce_case_study.mismatched_rows(rows)
    assert {row.metric for row in failures} == {"KS statistic (raw)", "GADS rank (EQP_B)"}
    assert reproduce_case_study.report_case_study(wrong) == 1

import math

import pandas as pd

from shap_diff_analysis.metrics import evaluate_rankings, evaluate_rankings_per_equipment


def _rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["GADS"] * 6,
            "equipment": ["EQP_A"] * 3 + ["EQP_B"] * 3,
            "feature": ["noise", "root_a", "harmless", "root_b2", "noise", "root_b1"],
            "rank": [1, 2, 3, 1, 2, 3],
        }
    )


def test_metrics_support_multiple_root_causes_and_far() -> None:
    """Multiple ground-truth roots and harmless false alarms are aggregated."""
    result = evaluate_rankings(
        _rankings(),
        root_causes={"EQP_A": ("root_a",), "EQP_B": ("root_b1", "root_b2")},
        harmless_features=("harmless",),
    ).iloc[0]
    assert result["mrr"] == 0.75
    assert result["hr_at_1"] == 0.5
    assert result["hr_at_3"] == 1.0
    assert result["root_cause_coverage_at_3"] == 1.0
    assert result["far_at_1"] == 0.0
    assert result["far_at_3"] == 0.5


def test_null_scenario_leaves_root_cause_metrics_undefined() -> None:
    """S4-null has no root-cause ranking target, so root metrics are NaN."""
    result = evaluate_rankings_per_equipment(
        _rankings(),
        root_causes={},
        harmless_features=("harmless",),
    )
    assert bool(result["mrr"].isna().all())
    assert bool(result["hr_at_1"].isna().all())
    assert bool(result["root_cause_coverage_at_5"].isna().all())
    assert not math.isnan(result.loc[result["equipment"] == "EQP_A", "far_at_5"].iloc[0])

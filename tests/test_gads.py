import pandas as pd

from shap_diff_analysis.gads import rank_gads_features


def test_gads_uses_explicit_normal_pool_and_excludes_eqp_from_process_view() -> None:
    """GADS must keep abnormal devices out of the one-vs-rest reference pool."""
    groups = pd.Series(["A", "A", "B", "B", "C", "C"], name="EQP")
    shap_values = pd.DataFrame(
        {
            "root_a": [10.0, 10.0, 100.0, 100.0, 0.0, 0.0],
            "noise": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "EQP": [3.0, 3.0, 4.0, 4.0, 0.0, 0.0],
        }
    )
    process = rank_gads_features(
        shap_values,
        groups,
        abnormal_devices=("A", "B"),
        normal_devices=("C",),
        process_features=("root_a", "noise"),
    )
    assert "EQP" not in process["feature"].to_list()
    assert process.loc[(process["equipment"] == "A") & (process["feature"] == "root_a"), "score"].iloc[0] == 10.0

    residual = rank_gads_features(
        shap_values,
        groups,
        abnormal_devices=("A",),
        normal_devices=("C",),
        process_features=("root_a", "noise"),
        view="residual",
    )
    assert residual["feature"].to_list() == ["EQP"]

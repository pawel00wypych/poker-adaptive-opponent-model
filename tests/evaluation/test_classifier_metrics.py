import pandas as pd
import pytest

from src.evaluation.metrics.classifier_metrics import (
    calculate_classifier_summary,
)


def test_calculate_classifier_summary(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_predicted_type": "calling",
                "classifier_accuracy": 0.8,
                "classifier_coverage": 0.6,
                "policy_switches": 1,
                "first_classification_hand": 4,
                "first_correct_classification_hand": 5,
            },
            {
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_predicted_type": "tight",
                "classifier_accuracy": 0.5,
                "classifier_coverage": 0.8,
                "policy_switches": 3,
                "first_classification_hand": 2,
                "first_correct_classification_hand": 6,
            },
            {
                "game_id": 2,
                "agent_name": "rule_based",
                "opponent_name": "calling",
                "final_predicted_type": "",
                "classifier_accuracy": 0.0,
                "classifier_coverage": 0.0,
                "policy_switches": 0,
                "first_classification_hand": None,
                "first_correct_classification_hand": None,
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    summary = calculate_classifier_summary(
        str(csv_path)
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["opponent_name"] == "calling"
    assert row["games"] == 2
    assert row["final_prediction_accuracy"] == 50
    assert row["mean_classifier_accuracy"] == pytest.approx(
        65
    )
    assert row["mean_classifier_coverage"] == pytest.approx(
        70
    )
    assert row["mean_policy_switches"] == 2
    assert row["mean_first_classification_hand"] == 3
    assert (
        row["mean_first_correct_classification_hand"]
        == 5.5
    )


def test_classifier_summary_groups_by_opponent(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "tight",
                "final_predicted_type": "tight",
                "classifier_accuracy": 1.0,
                "classifier_coverage": 0.8,
                "policy_switches": 1,
                "first_classification_hand": 3,
                "first_correct_classification_hand": 3,
            },
            {
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "opponent_name": "aggressive",
                "final_predicted_type": "aggressive",
                "classifier_accuracy": 1.0,
                "classifier_coverage": 0.7,
                "policy_switches": 1,
                "first_classification_hand": 4,
                "first_correct_classification_hand": 4,
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    summary = calculate_classifier_summary(
        str(csv_path)
    )

    assert set(summary["opponent_name"]) == {
        "tight",
        "aggressive",
    }
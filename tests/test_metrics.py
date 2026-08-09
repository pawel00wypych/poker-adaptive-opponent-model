import pandas as pd
import pytest

from src.evaluation.metrics import calculate_bb_per_100


def test_calculate_extended_metrics(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 240,
                "initial_stack": 200,
                "profit": 40,
                "profit_bb": 4,
                "hands_played": 50,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 0,
                "ended_by_round_limit": 1,
                "classified_decisions": 8,
                "correct_classifications": 6,
                "incorrect_classifications": 2,
                "unknown_classifications": 2,
                "classifier_accuracy": 0.75,
                "classifier_coverage": 0.8,
                "policy_switches": 1,
                "first_classification_hand": 3,
                "first_correct_classification_hand": 4,
                "final_predicted_type": "calling",
            },
            {
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 160,
                "initial_stack": 200,
                "profit": -40,
                "profit_bb": -4,
                "hands_played": 50,
                "won_game": 0,
                "busted": 0,
                "ended_by_bust": 0,
                "ended_by_round_limit": 1,
                "classified_decisions": 6,
                "correct_classifications": 3,
                "incorrect_classifications": 3,
                "unknown_classifications": 4,
                "classifier_accuracy": 0.5,
                "classifier_coverage": 0.6,
                "policy_switches": 3,
                "first_classification_hand": 5,
                "first_correct_classification_hand": 7,
                "final_predicted_type": "tight",
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    summary = calculate_bb_per_100(
        str(csv_path)
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["total_profit_bb"] == 0
    assert row["total_hands"] == 100
    assert row["games"] == 2
    assert row["mean_profit_bb"] == 0
    assert row["bb_per_100"] == 0

    assert row["win_rate"] == 50
    assert row["bust_rate"] == 0
    assert row["ended_by_bust_rate"] == 0
    assert row["ended_by_round_limit_rate"] == 100

    assert row["total_classified_decisions"] == 14
    assert row["total_correct_classifications"] == 9
    assert row["total_incorrect_classifications"] == 5
    assert row["total_unknown_classifications"] == 6

    assert row["mean_classifier_accuracy"] == pytest.approx(
        62.5
    )
    assert row["mean_classifier_coverage"] == pytest.approx(
        70.0
    )

    assert row["global_classifier_accuracy"] == pytest.approx(
        9 / 14 * 100
    )
    assert row["global_classifier_coverage"] == pytest.approx(
        14 / 20 * 100
    )

    assert row["mean_policy_switches"] == 2
    assert row["mean_first_classification_hand"] == 4
    assert (
        row["mean_first_correct_classification_hand"]
        == 5.5
    )


def test_calculate_metrics_for_non_adaptive_player(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"

    df = pd.DataFrame(
        [
            {
                "experiment_name": "rule_based_vs_tight",
                "game_id": 0,
                "agent_name": "rule_based",
                "opponent_name": "tight",
                "final_stack": 400,
                "initial_stack": 200,
                "profit": 200,
                "profit_bb": 20,
                "hands_played": 20,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 1,
                "ended_by_round_limit": 0,
                "classified_decisions": 0,
                "correct_classifications": 0,
                "incorrect_classifications": 0,
                "unknown_classifications": 0,
                "classifier_accuracy": 0.0,
                "classifier_coverage": 0.0,
                "policy_switches": 0,
                "first_classification_hand": None,
                "first_correct_classification_hand": None,
                "final_predicted_type": "",
            }
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    summary = calculate_bb_per_100(
        str(csv_path)
    )

    row = summary.iloc[0]

    assert row["global_classifier_accuracy"] == 0
    assert row["global_classifier_coverage"] == 0
    assert row["ended_by_bust_rate"] == 100
    assert row["ended_by_round_limit_rate"] == 0
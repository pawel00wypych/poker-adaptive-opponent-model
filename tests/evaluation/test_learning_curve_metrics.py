import pandas as pd
import pytest

from src.evaluation.metrics.learning_curve_metrics import (
    calculate_learning_curve_metrics,
)


def test_calculate_learning_curve_metrics_groups_by_seed_and_checkpoint(
    tmp_path,
):
    csv_path = tmp_path / "checkpoint_eval.csv"

    df = pd.DataFrame(
        [
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "checkpoint",
                "training_episode": None,
                "checkpoint_episode": 5000,
                "experiment_id": "seed_42_episodes_5000",
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 220,
                "initial_stack": 200,
                "profit": 20,
                "profit_bb": 2,
                "hands_played": 10,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 1,
                "ended_by_round_limit": 0,
                "classified_decisions": 10,
                "correct_classifications": 10,
                "incorrect_classifications": 0,
                "unknown_classifications": 2,
                "other_classifications": 0,
                "classifier_accuracy": 1.0,
                "classifier_coverage": 10 / 12,
                "policy_switches": 1,
                "first_classification_hand": 2,
                "first_correct_classification_hand": 2,
                "first_classification_action_count": 5,
                "first_correct_classification_action_count": 5,
                "final_predicted_type": "calling",
                "policy_decisions": 0,
                "unseen_state_decisions": 0,
                "untried_action_selections": 0,
                "unseen_state_decision_rate": 0.0,
                "untried_action_selection_rate": 0.0,
            },
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "checkpoint",
                "training_episode": None,
                "checkpoint_episode": 5000,
                "experiment_id": "seed_42_episodes_5000",
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 1,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 180,
                "initial_stack": 200,
                "profit": -20,
                "profit_bb": -2,
                "hands_played": 10,
                "won_game": 0,
                "busted": 0,
                "ended_by_bust": 0,
                "ended_by_round_limit": 1,
                "classified_decisions": 8,
                "correct_classifications": 6,
                "incorrect_classifications": 2,
                "unknown_classifications": 4,
                "other_classifications": 0,
                "classifier_accuracy": 0.75,
                "classifier_coverage": 8 / 12,
                "policy_switches": 2,
                "first_classification_hand": 3,
                "first_correct_classification_hand": 4,
                "first_classification_action_count": 6,
                "first_correct_classification_action_count": 7,
                "final_predicted_type": "calling",
                "policy_decisions": 0,
                "unseen_state_decisions": 0,
                "untried_action_selections": 0,
                "unseen_state_decision_rate": 0.0,
                "untried_action_selection_rate": 0.0,
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    result = calculate_learning_curve_metrics(str(csv_path))

    assert len(result) == 1

    row = result.iloc[0]

    assert row["model_seed"] == 42
    assert row["checkpoint_episode"] == 5000
    assert row["games"] == 2
    assert row["total_profit_bb"] == 0
    assert row["total_hands"] == 20
    assert row["bb_per_100"] == 0
    assert row["global_classifier_accuracy"] == pytest.approx(88.88888888888889)

    assert row["global_classifier_coverage"] == pytest.approx(75.0)

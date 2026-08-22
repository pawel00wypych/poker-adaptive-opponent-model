import pandas as pd
import pytest

from src.evaluation.metrics.evaluation_metrics import (
    calculate_final_model_metrics,
)


def test_calculate_final_model_metrics_groups_by_seed_and_training_episode(
    tmp_path,
):
    csv_path = tmp_path / "final_model_eval.csv"

    df = pd.DataFrame(
        [
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "final",
                "training_episode": 5000,
                "checkpoint_episode": None,
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
            },
            {
                "training_run": "state_v2",
                "model_seed": 42,
                "model_source": "final",
                "training_episode": 5000,
                "checkpoint_episode": None,
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
            },
        ]
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    result = calculate_final_model_metrics(str(csv_path))

    assert len(result) == 1

    row = result.iloc[0]

    assert row["model_seed"] == 42
    assert row["training_episode"] == 5000
    assert row["games"] == 2
    assert row["total_profit_bb"] == 0
    assert row["total_hands"] == 20
    assert row["bb_per_100"] == 0
    assert row["global_classifier_accuracy"] == pytest.approx(88.88888888888889)

    assert row["global_classifier_coverage"] == pytest.approx(75.0)


def test_global_classifier_coverage_excludes_other_classifications(tmp_path):
    """'other' is an unconverted opportunity, so it lowers coverage."""
    csv_path = tmp_path / "other_classifications.csv"

    row = {
        "training_run": "state_v2",
        "model_seed": 42,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
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
        "classified_decisions": 5,
        "correct_classifications": 5,
        "incorrect_classifications": 0,
        "unknown_classifications": 3,
        "other_classifications": 2,
        "classifier_accuracy": 1.0,
        "classifier_coverage": 0.5,
        "policy_switches": 1,
        "first_classification_hand": 2,
        "first_correct_classification_hand": 2,
        "first_classification_action_count": 5,
        "first_correct_classification_action_count": 5,
        "final_predicted_type": "calling",
    }

    pd.DataFrame([row]).to_csv(csv_path, index=False)

    result = calculate_final_model_metrics(str(csv_path))
    metrics = result.iloc[0]

    assert metrics["total_other_classifications"] == 2
    # 5 covered out of 5 + 3 unknown + 2 other, not 5 / 8.
    assert metrics["global_classifier_coverage"] == pytest.approx(50.0)
    assert metrics["global_other_rate"] == pytest.approx(20.0)
    # Accuracy is scored only over committed specialist classifications.
    assert metrics["global_classifier_accuracy"] == pytest.approx(100.0)


def test_final_model_metrics_ignore_untrained_baseline_replicates(tmp_path):
    csv_path = tmp_path / "mixed_head_to_head.csv"
    final_row = {
        "training_run": "state_v2",
        "model_seed": 42,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
        "evaluation_replicate_id": None,
        "experiment_name": "adaptive_mc_vs_rule_based",
        "game_id": 0,
        "agent_name": "adaptive_mc",
        "opponent_name": "rule_based",
        "profit_bb": 2.0,
        "hands_played": 10,
        "won_game": 1,
        "busted": 0,
        "ended_by_bust": 0,
        "ended_by_round_limit": 1,
        "classified_decisions": 0,
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "unknown_classifications": 0,
        "other_classifications": 0,
        "classifier_accuracy": 0.0,
        "classifier_coverage": 0.0,
        "policy_switches": 0,
        "first_classification_hand": None,
        "first_correct_classification_hand": None,
        "first_classification_action_count": None,
        "first_correct_classification_action_count": None,
    }
    baseline_row = {
        **final_row,
        "training_run": None,
        "model_seed": None,
        "model_source": None,
        "training_episode": None,
        "evaluation_replicate_id": 0,
        "experiment_name": "always_call_vs_rule_based",
        "agent_name": "always_call",
    }
    pd.DataFrame([final_row, baseline_row]).to_csv(csv_path, index=False)

    result = calculate_final_model_metrics(str(csv_path))

    assert len(result) == 1
    assert result.iloc[0]["agent_name"] == "adaptive_mc"

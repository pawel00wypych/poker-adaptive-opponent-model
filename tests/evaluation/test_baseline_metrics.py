import math

import pandas as pd
import pytest

from src.evaluation.metrics.baseline_metrics import (
    BASELINE_REPLICATE_CI_LOWER_COLUMN,
    BASELINE_REPLICATE_CI_UPPER_COLUMN,
    BASELINE_REPLICATE_STD_COLUMN,
    aggregate_across_evaluation_replicates,
    calculate_baseline_replicate_metrics,
)


def baseline_game_row(
    *,
    replicate_id,
    game_id,
    profit_bb,
    agent_name="always_call",
    opponent_name="rule_based",
):
    return {
        "training_run": None,
        "model_seed": None,
        "checkpoint_episode": None,
        "evaluation_replicate_id": replicate_id,
        "experiment_id": f"evaluation_replicate_{replicate_id}",
        "experiment_name": f"{agent_name}_vs_{opponent_name}",
        "game_id": game_id,
        "matchup_game_index": game_id,
        "evaluation_seed": 300_000 + game_id,
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "final_stack": 200,
        "initial_stack": 200,
        "profit": 0,
        "profit_bb": profit_bb,
        "hands_played": 20,
        "won_game": int(profit_bb > 0),
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
        "final_predicted_type": "",
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }


def test_baseline_metrics_group_by_evaluation_replicate_not_model_seed(
    tmp_path,
):
    csv_path = tmp_path / "baseline.csv"
    rows = [
        baseline_game_row(replicate_id=0, game_id=0, profit_bb=1.0),
        baseline_game_row(replicate_id=0, game_id=1, profit_bb=3.0),
        baseline_game_row(replicate_id=1, game_id=0, profit_bb=5.0),
        baseline_game_row(replicate_id=1, game_id=1, profit_bb=7.0),
    ]
    learned_row = baseline_game_row(
        replicate_id=None,
        game_id=9,
        profit_bb=100.0,
        agent_name="adaptive_mc",
    )
    learned_row["model_seed"] = 42
    learned_row["checkpoint_episode"] = 2000
    rows.append(learned_row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    replicate_metrics = calculate_baseline_replicate_metrics(csv_path)
    aggregated = aggregate_across_evaluation_replicates(replicate_metrics)
    row = aggregated.iloc[0]

    assert list(replicate_metrics["evaluation_replicate_id"]) == [0, 1]
    assert list(replicate_metrics["mean_profit_bb"]) == [2.0, 6.0]
    assert row["evaluation_replicates"] == 2
    assert row["games"] == 4
    assert row["mean_profit_bb"] == 4.0
    assert math.isclose(row[BASELINE_REPLICATE_STD_COLUMN], math.sqrt(8.0))
    assert row[BASELINE_REPLICATE_CI_LOWER_COLUMN] < row["mean_profit_bb"]
    assert row[BASELINE_REPLICATE_CI_UPPER_COLUMN] > row["mean_profit_bb"]
    assert "seeds" not in aggregated.columns
    assert "mean_profit_bb_std_across_seeds" not in aggregated.columns


def test_baseline_metrics_require_evaluation_replicate_id(tmp_path):
    csv_path = tmp_path / "legacy.csv"
    row = baseline_game_row(replicate_id=0, game_id=0, profit_bb=1.0)
    row.pop("evaluation_replicate_id")
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="evaluation_replicate_id"):
        calculate_baseline_replicate_metrics(csv_path)


def test_baseline_metrics_reject_model_metadata_on_replicate_rows(tmp_path):
    csv_path = tmp_path / "conflicting_metadata.csv"
    row = baseline_game_row(replicate_id=0, game_id=0, profit_bb=1.0)
    row["model_seed"] = 42
    row["checkpoint_episode"] = 2000
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="trained-model metadata"):
        calculate_baseline_replicate_metrics(csv_path)


@pytest.mark.parametrize("invalid_replicate_id", [-1, 1.5, "invalid"])
def test_baseline_metrics_reject_invalid_replicate_ids(
    tmp_path,
    invalid_replicate_id,
):
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        [
            baseline_game_row(
                replicate_id=invalid_replicate_id,
                game_id=0,
                profit_bb=1.0,
            )
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-negative integers"):
        calculate_baseline_replicate_metrics(csv_path)

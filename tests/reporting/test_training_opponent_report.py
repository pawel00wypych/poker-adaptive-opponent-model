import pandas as pd
import pytest

from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_MAX_COLUMN,
    SEED_MIN_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
)
from src.evaluation.reporting.training_opponent_report import (
    aggregate_across_seeds,
    load_training_opponent_report_data,
    write_training_opponent_html_report,
    write_training_opponent_markdown_report,
)


def final_row(seed: int, profit_bb: float) -> dict:
    return {
        "training_run": "sample_run",
        "model_seed": seed,
        "model_source": "final",
        "training_episode": 5000,
        "checkpoint_episode": None,
        "experiment_id": f"seed_{seed}_final_episode_5000",
        "experiment_name": "adaptive_mc_vs_calling",
        "game_id": 0,
        "agent_name": "adaptive_mc",
        "opponent_name": "calling",
        "final_stack": 200 + int(profit_bb * 10),
        "initial_stack": 200,
        "profit": int(profit_bb * 10),
        "profit_bb": profit_bb,
        "hands_played": 10,
        "won_game": int(profit_bb > 0),
        "busted": 0,
        "ended_by_bust": 0,
        "ended_by_round_limit": 1,
        "classified_decisions": 10,
        "correct_classifications": 8,
        "incorrect_classifications": 2,
        "unknown_classifications": 0,
        "other_classifications": 0,
        "classifier_accuracy": 0.8,
        "classifier_coverage": 1.0,
        "policy_switches": 1,
        "first_classification_hand": 1,
        "first_correct_classification_hand": 1,
        "first_classification_action_count": 5,
        "first_correct_classification_action_count": 5,
        "final_predicted_type": "calling",
    }


def write_final_csv(path) -> None:
    pd.DataFrame([final_row(42, 2.0), final_row(456, 1.0)]).to_csv(
        path,
        index=False,
    )


def test_load_training_opponent_data_uses_final_model_metadata(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_final_csv(csv_path)

    result = load_training_opponent_report_data(csv_path, opponent="calling")

    assert len(result) == 2
    assert set(result["training_episode"]) == {5000}


def test_aggregate_across_seeds_calculates_final_model_statistics(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_final_csv(csv_path)

    aggregated = aggregate_across_seeds(load_training_opponent_report_data(csv_path))

    row = aggregated.iloc[0]
    assert row["seeds"] == 2
    assert row["games"] == 2
    assert row["mean_profit_bb"] == 1.5
    assert row["mean_profit_bb_std_across_seeds"] == pytest.approx(0.7071067811865476)
    assert row[SEED_STANDARD_ERROR_COLUMN] == pytest.approx(0.5)
    assert row[SEED_CI_LOWER_COLUMN] == pytest.approx(-4.853102368087348)
    assert row[SEED_CI_UPPER_COLUMN] == pytest.approx(7.853102368087348)
    assert row[SEED_MIN_COLUMN] == 1.0
    assert row[SEED_MAX_COLUMN] == 2.0
    assert row[SEED_SPREAD_COLUMN] == 1.0


def test_training_opponent_reports_name_final_model_scope(tmp_path):
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    write_final_csv(csv_path)

    html_path = write_training_opponent_html_report(csv_path, output_dir)
    markdown_path = write_training_opponent_markdown_report(csv_path, output_dir)

    assert html_path.name == "training_opponent_report.html"
    assert markdown_path.name == "training_opponent_report.md"
    assert "Only final trained models" in html_path.read_text(encoding="utf-8")
    assert "checkpoints belong to learning-curve" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_final_report_rejects_checkpoint_rows(tmp_path):
    row = final_row(42, 1.0)
    row.update(
        {
            "model_source": "checkpoint",
            "training_episode": None,
            "checkpoint_episode": 1000,
        }
    )
    csv_path = tmp_path / "checkpoint.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-final model sources"):
        load_training_opponent_report_data(csv_path)

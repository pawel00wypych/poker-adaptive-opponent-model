import pandas as pd

from src.evaluation.reporting.checkpoint_report import (
    aggregate_across_seeds,
    best_rows_by_agent,
    load_checkpoint_report_data,
    write_checkpoint_html_report,
)


def write_sample_checkpoint_csv(path):
    df = pd.DataFrame(
        [
            {
                "training_run": "sample_run",
                "model_seed": 42,
                "checkpoint_episode": 500,
                "experiment_id": "a",
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
                "unknown_classifications": 0,
                "classifier_accuracy": 1.0,
                "classifier_coverage": 1.0,
                "policy_switches": 1,
                "first_classification_hand": 1,
                "first_correct_classification_hand": 1,
                "first_classification_action_count": 5,
                "first_correct_classification_action_count": 5,
                "final_predicted_type": "calling",
            },
            {
                "training_run": "sample_run",
                "model_seed": 456,
                "checkpoint_episode": 500,
                "experiment_id": "b",
                "experiment_name": "adaptive_mc_vs_calling",
                "game_id": 0,
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "final_stack": 210,
                "initial_stack": 200,
                "profit": 10,
                "profit_bb": 1,
                "hands_played": 10,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 1,
                "ended_by_round_limit": 0,
                "classified_decisions": 8,
                "correct_classifications": 8,
                "incorrect_classifications": 0,
                "unknown_classifications": 2,
                "classifier_accuracy": 1.0,
                "classifier_coverage": 0.8,
                "policy_switches": 1,
                "first_classification_hand": 2,
                "first_correct_classification_hand": 2,
                "first_classification_action_count": 6,
                "first_correct_classification_action_count": 6,
                "final_predicted_type": "calling",
            },
        ]
    )
    df.to_csv(path, index=False)


def test_load_checkpoint_report_data_filters_opponent(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_checkpoint_csv(csv_path)

    result = load_checkpoint_report_data(csv_path, opponent="calling")

    assert len(result) == 2
    assert set(result["opponent_name"]) == {"calling"}


def test_aggregate_across_seeds_calculates_agent_checkpoint_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_checkpoint_csv(csv_path)
    metrics = load_checkpoint_report_data(csv_path)

    aggregated = aggregate_across_seeds(metrics)

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["seeds"] == 2
    assert row["games"] == 2
    assert row["mean_profit_bb"] == 1.5


def test_best_rows_by_agent_selects_highest_mean_profit(tmp_path):
    df = pd.DataFrame(
        [
            {
                "training_run": "sample",
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "checkpoint_episode": 500,
                "mean_profit_bb": 1.0,
            },
            {
                "training_run": "sample",
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "checkpoint_episode": 1500,
                "mean_profit_bb": 3.0,
            },
        ]
    )

    result = best_rows_by_agent(df)

    assert len(result) == 1
    assert result.iloc[0]["checkpoint_episode"] == 1500


def test_write_checkpoint_html_report_creates_file_and_plots(tmp_path):
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "report"
    write_sample_checkpoint_csv(csv_path)

    output_path = write_checkpoint_html_report(csv_path, output_dir)

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Checkpoint evaluation report" in html
    assert "Metric glossary" in html
    assert any((output_dir / "plots").glob("*.png"))

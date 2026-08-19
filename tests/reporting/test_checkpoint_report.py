import pandas as pd
import pytest

from src.evaluation.algorithm_metadata import ADAPTIVE_AGENTS
from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_MAX_COLUMN,
    SEED_MIN_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
)
from src.evaluation.reporting.checkpoint_report import (
    aggregate_across_seeds,
    best_rows_by_agent,
    create_checkpoint_plots,
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
    assert row["mean_profit_bb_std_across_seeds"] == pytest.approx(
        0.7071067811865476
    )
    assert row[SEED_STANDARD_ERROR_COLUMN] == pytest.approx(0.5)
    assert row[SEED_CI_LOWER_COLUMN] == pytest.approx(-4.853102368087348)
    assert row[SEED_CI_UPPER_COLUMN] == pytest.approx(7.853102368087348)
    assert row[SEED_MIN_COLUMN] == 1.0
    assert row[SEED_MAX_COLUMN] == 2.0
    assert row[SEED_SPREAD_COLUMN] == 1.0


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


def test_classifier_plots_include_all_adaptive_agents(monkeypatch, tmp_path):
    rows = []
    for agent_name in [*ADAPTIVE_AGENTS, "rule_based"]:
        rows.append(
            {
                "agent_name": agent_name,
                "checkpoint_episode": 1000,
                "mean_profit_bb": 1.0,
                "bb_per_100": 10.0,
                "win_rate": 60.0,
                "bust_rate": 10.0,
                "global_classifier_accuracy": 80.0,
                "global_classifier_coverage": 90.0,
            }
        )

    classifier_plot_agents = []

    def capture_plot(aggregated, metric, ylabel, output_path):
        if metric.startswith("global_classifier_"):
            classifier_plot_agents.append(set(aggregated["agent_name"]))

    monkeypatch.setattr(
        "src.evaluation.reporting.checkpoint_report.plot_metric_by_checkpoint",
        capture_plot,
    )

    create_checkpoint_plots(pd.DataFrame(rows), tmp_path)

    assert classifier_plot_agents == [set(ADAPTIVE_AGENTS)] * 2

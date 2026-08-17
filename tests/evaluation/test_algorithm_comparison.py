import json

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ALGORITHM_DOUBLE_Q_LEARNING,
    ALGORITHM_MONTE_CARLO,
    ALGORITHM_Q_LEARNING,
    ALGORITHM_SARSA,
)
from src.evaluation.reporting.algorithm_comparison import (
    add_algorithm_deltas,
    add_algorithm_ranking,
    build_algorithm_comparison,
    build_algorithm_rows,
    build_global_algorithm_ranking,
    write_algorithm_comparison_outputs,
)
from src.experiments.reporting.create_algorithm_comparison import parse_args

REQUIRED_RESULT_COLUMNS = {
    "training_run": "sample_run",
    "experiment_id": "sample_experiment",
    "final_stack": 220,
    "initial_stack": 200,
    "profit": 20,
    "ended_by_bust": 0,
    "ended_by_round_limit": 1,
    "classified_decisions": 0,
    "correct_classifications": 0,
    "incorrect_classifications": 0,
    "unknown_classifications": 0,
    "classifier_accuracy": 0.0,
    "classifier_coverage": 0.0,
    "policy_switches": 0,
    "first_classification_hand": 0,
    "first_correct_classification_hand": 0,
    "first_classification_action_count": 0,
    "first_correct_classification_action_count": 0,
    "final_predicted_type": "unknown",
}


def make_game_row(
    *,
    seed,
    checkpoint,
    agent,
    opponent,
    game_id,
    profit_bb,
    hands_played=20,
):
    row = REQUIRED_RESULT_COLUMNS.copy()
    row.update(
        {
            "model_seed": seed,
            "checkpoint_episode": checkpoint,
            "experiment_name": f"{agent}_vs_{opponent}",
            "agent_name": agent,
            "opponent_name": opponent,
            "game_id": game_id,
            "profit_bb": profit_bb,
            "hands_played": hands_played,
            "won_game": 1 if profit_bb > 0 else 0,
            "busted": 1 if profit_bb < 0 else 0,
        }
    )
    return row


def add_group(
    rows,
    *,
    agent,
    opponent,
    profits,
    checkpoint=1000,
    seeds=(42, 123),
):
    for index, seed in enumerate(seeds):
        rows.append(
            make_game_row(
                seed=seed,
                checkpoint=checkpoint,
                agent=agent,
                opponent=opponent,
                game_id=index,
                profit_bb=profits[index],
            )
        )


def write_sample_results_csv(path):
    rows = []
    # calling: Q-learning is best, MC is second.
    add_group(rows, agent="adaptive_mc", opponent="calling", profits=(10.0, 10.0))
    add_group(rows, agent="adaptive_q_learning", opponent="calling", profits=(12.0, 12.0))
    add_group(rows, agent="adaptive_sarsa", opponent="calling", profits=(11.0, 11.0))
    add_group(rows, agent="adaptive_double_q_learning", opponent="calling", profits=(9.0, 9.0))
    add_group(rows, agent="rule_based", opponent="calling", profits=(0.0, 0.0))
    add_group(rows, agent="oracle_mc", opponent="calling", profits=(14.0, 14.0))
    add_group(rows, agent="oracle_q_learning", opponent="calling", profits=(15.0, 15.0))
    add_group(rows, agent="oracle_sarsa", opponent="calling", profits=(14.0, 14.0))
    add_group(rows, agent="oracle_double_q_learning", opponent="calling", profits=(13.0, 13.0))

    # aggressive_extreme: Double Q-learning is best and beats MC.
    add_group(rows, agent="adaptive_mc", opponent="aggressive_extreme", profits=(-8.0, -8.0))
    add_group(rows, agent="adaptive_q_learning", opponent="aggressive_extreme", profits=(-2.0, -2.0))
    add_group(rows, agent="adaptive_sarsa", opponent="aggressive_extreme", profits=(-3.0, -3.0))
    add_group(rows, agent="adaptive_double_q_learning", opponent="aggressive_extreme", profits=(1.0, 1.0))
    add_group(rows, agent="rule_based", opponent="aggressive_extreme", profits=(-10.0, -10.0))
    add_group(rows, agent="oracle_mc", opponent="aggressive_extreme", profits=(2.0, 2.0))
    add_group(rows, agent="oracle_q_learning", opponent="aggressive_extreme", profits=(3.0, 3.0))
    add_group(rows, agent="oracle_sarsa", opponent="aggressive_extreme", profits=(2.5, 2.5))
    add_group(rows, agent="oracle_double_q_learning", opponent="aggressive_extreme", profits=(4.0, 4.0))

    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_algorithm_rows_maps_only_adaptive_rl_agents():
    aggregated = pd.DataFrame(
        [
            {"agent_name": "adaptive_mc", "mean_profit_bb": 1.0},
            {"agent_name": "adaptive_q_learning", "mean_profit_bb": 2.0},
            {"agent_name": "adaptive_sarsa", "mean_profit_bb": 3.0},
            {"agent_name": "adaptive_double_q_learning", "mean_profit_bb": 4.0},
            {"agent_name": "rule_based", "mean_profit_bb": 5.0},
        ]
    )

    rows = build_algorithm_rows(aggregated)

    assert set(rows["algorithm"]) == {
        ALGORITHM_MONTE_CARLO,
        ALGORITHM_Q_LEARNING,
        ALGORITHM_SARSA,
        ALGORITHM_DOUBLE_Q_LEARNING,
    }
    assert "rule_based" not in set(rows["agent_name"])


def test_algorithm_ranking_and_deltas_are_computed_per_opponent():
    aggregated = pd.DataFrame(
        [
            {
                "training_run": "run",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "agent_name": "adaptive_mc",
                "mean_profit_bb": 10.0,
                "bb_per_100": 20.0,
                "win_rate": 80.0,
                "bust_rate": 5.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "agent_name": "adaptive_q_learning",
                "mean_profit_bb": 12.0,
                "bb_per_100": 24.0,
                "win_rate": 85.0,
                "bust_rate": 4.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "agent_name": "rule_based",
                "mean_profit_bb": 1.0,
                "bb_per_100": 2.0,
                "win_rate": 50.0,
                "bust_rate": 10.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "agent_name": "oracle_mc",
                "mean_profit_bb": 14.0,
                "bb_per_100": 28.0,
                "win_rate": 90.0,
                "bust_rate": 2.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "agent_name": "oracle_q_learning",
                "mean_profit_bb": 15.0,
                "bb_per_100": 30.0,
                "win_rate": 92.0,
                "bust_rate": 1.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
        ]
    )

    algorithm_rows = build_algorithm_rows(aggregated)
    ranking = add_algorithm_ranking(algorithm_rows)
    with_deltas = add_algorithm_deltas(ranking, aggregated)

    q_row = with_deltas[with_deltas["algorithm"] == ALGORITHM_Q_LEARNING].iloc[0]
    mc_row = with_deltas[with_deltas["algorithm"] == ALGORITHM_MONTE_CARLO].iloc[0]

    assert q_row["rank"] == 1
    assert q_row["delta_vs_monte_carlo"] == 2.0
    assert q_row["delta_vs_rule_based"] == 11.0
    assert q_row["delta_vs_oracle"] == -3.0
    assert mc_row["delta_vs_oracle"] == -4.0
    assert mc_row["delta_vs_monte_carlo"] == 0.0


def test_global_ranking_uses_average_profit_and_average_rank():
    rows = pd.DataFrame(
        [
            {"algorithm": ALGORITHM_MONTE_CARLO, "rank": 2, "mean_profit_bb": 10.0, "bb_per_100": 20.0, "win_rate": 80.0, "bust_rate": 5.0, "mean_profit_bb_std_across_seeds": 1.0},
            {"algorithm": ALGORITHM_MONTE_CARLO, "rank": 4, "mean_profit_bb": -8.0, "bb_per_100": -16.0, "win_rate": 20.0, "bust_rate": 80.0, "mean_profit_bb_std_across_seeds": 2.0},
            {"algorithm": ALGORITHM_DOUBLE_Q_LEARNING, "rank": 1, "mean_profit_bb": 9.0, "bb_per_100": 18.0, "win_rate": 90.0, "bust_rate": 4.0, "mean_profit_bb_std_across_seeds": 1.0},
            {"algorithm": ALGORITHM_DOUBLE_Q_LEARNING, "rank": 1, "mean_profit_bb": 1.0, "bb_per_100": 2.0, "win_rate": 55.0, "bust_rate": 45.0, "mean_profit_bb_std_across_seeds": 1.0},
        ]
    )

    global_ranking = build_global_algorithm_ranking(rows)

    assert global_ranking.iloc[0]["algorithm"] == ALGORITHM_DOUBLE_Q_LEARNING
    assert global_ranking.iloc[0]["best_matchup_count"] == 2
    assert global_ranking.iloc[0]["positive_matchup_count"] == 2


def test_build_algorithm_comparison_from_raw_results_csv(tmp_path):
    input_path = tmp_path / "results.csv"
    write_sample_results_csv(input_path)

    report, global_ranking, algorithm_by_opponent, deltas = build_algorithm_comparison(input_path)

    assert report.overview["algorithms"] == [
        ALGORITHM_MONTE_CARLO,
        ALGORITHM_Q_LEARNING,
        ALGORITHM_SARSA,
        ALGORITHM_DOUBLE_Q_LEARNING,
    ]
    assert len(global_ranking) == 4
    assert len(algorithm_by_opponent) == 8
    assert "delta_vs_monte_carlo" in deltas.columns

    aggressive_rows = algorithm_by_opponent[
        algorithm_by_opponent["opponent_name"] == "aggressive_extreme"
    ]
    best = aggressive_rows.sort_values("rank").iloc[0]
    assert best["algorithm"] == ALGORITHM_DOUBLE_Q_LEARNING


def test_write_algorithm_comparison_outputs_creates_all_formats_and_charts(tmp_path):
    input_path = tmp_path / "results.csv"
    output_dir = tmp_path / "algorithm_comparison"
    write_sample_results_csv(input_path)

    created_paths = write_algorithm_comparison_outputs(
        input_path=input_path,
        output_dir=output_dir,
        report_format="all",
        include_charts=True,
    )

    expected_files = {
        "algorithm_comparison.md",
        "algorithm_comparison.json",
        "algorithm_global_ranking.csv",
        "algorithm_by_opponent.csv",
        "algorithm_deltas.csv",
        "algorithm_global_ranking.tex",
        "algorithm_by_opponent.tex",
        "algorithm_deltas.tex",
        "algorithm_mean_profit_by_opponent.png",
        "algorithm_seed_stability_by_opponent.png",
        "algorithm_global_mean_profit.png",
    }
    created_names = {path.name for path in created_paths}

    assert expected_files.issubset(created_names)
    assert (output_dir / "algorithm_comparison.md").exists()
    assert (output_dir / "charts" / "algorithm_global_mean_profit.png").exists()

    markdown = (output_dir / "algorithm_comparison.md").read_text(encoding="utf-8")
    assert "# RL algorithm comparison" in markdown
    assert "source_raw_games" in markdown
    assert "algorithm_summary_rows" in markdown
    assert "delta_vs_monte_carlo" in markdown

    data = json.loads((output_dir / "algorithm_comparison.json").read_text(encoding="utf-8"))
    assert data["overview"]["algorithm_summary_rows"] == 8


def test_parse_args_supports_algorithm_comparison_options():
    args = parse_args(
        [
            "--input-path",
            "results.csv",
            "--output-dir",
            "reports/algorithm_comparison",
            "--format",
            "all",
            "--no-export-latex",
            "--no-include-charts",
            "--max-std-across-seeds-bb",
            "7.5",
        ]
    )

    assert args.input_path == "results.csv"
    assert args.output_dir == "reports/algorithm_comparison"
    assert args.format == "all"
    assert args.export_latex is False
    assert args.include_charts is False
    assert args.max_std_across_seeds_bb == 7.5

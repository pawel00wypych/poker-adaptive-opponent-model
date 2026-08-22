import json

import pandas as pd

from src.evaluation.metrics.oracle_gap import ORACLE_GAP_BB_COLUMN
from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
)
from src.evaluation.reporting.experiment_summary import (
    QUALITY_FAIL,
    QUALITY_OK,
    QUALITY_WARNING,
    SummaryThresholds,
    add_baseline_deltas,
    add_quality_flags,
    build_agent_ranking,
    build_experiment_summary,
    write_experiment_summary_outputs,
)
from src.experiments.reporting.create_experiment_summary import (
    build_thresholds,
    parse_args,
)

REQUIRED_RESULT_COLUMNS = {
    "training_run": "sample_run",
    "experiment_id": "sample_experiment",
    "final_stack": 220,
    "initial_stack": 200,
    "profit": 20,
    "ended_by_bust": 1,
    "ended_by_round_limit": 0,
    "total_unknown_classifications": 0,
    "total_other_classifications": 0,
    "first_classification_hand": 1,
    "first_correct_classification_hand": 1,
    "first_classification_action_count": 1,
    "first_correct_classification_action_count": 1,
    "final_predicted_type": "unknown",
}


def make_game_row(
    *,
    seed,
    training_episode,
    agent,
    opponent,
    game_id,
    profit_bb,
    hands_played=20,
    won_game=1,
    busted=0,
    classified_decisions=0,
    correct_classifications=0,
    incorrect_classifications=0,
    unknown_classifications=0,
    classifier_accuracy=0.0,
    classifier_coverage=0.0,
    policy_switches=0,
):
    row = REQUIRED_RESULT_COLUMNS.copy()
    row.update(
        {
            "model_seed": seed,
            "model_source": "final",
            "training_episode": training_episode,
            "experiment_name": f"{agent}_vs_{opponent}",
            "agent_name": agent,
            "opponent_name": opponent,
            "game_id": game_id,
            "profit_bb": profit_bb,
            "hands_played": hands_played,
            "won_game": won_game,
            "busted": busted,
            "classified_decisions": classified_decisions,
            "correct_classifications": correct_classifications,
            "incorrect_classifications": incorrect_classifications,
            "unknown_classifications": unknown_classifications,
            "other_classifications": 0,
            "classifier_accuracy": classifier_accuracy,
            "classifier_coverage": classifier_coverage,
            "policy_switches": policy_switches,
        }
    )
    return row


def add_group(
    rows,
    *,
    agent,
    opponent,
    training_episode=2000,
    seed_values=(42, 123),
    profit_by_seed=(10.0, 12.0),
    win_by_seed=(1, 1),
    busted_by_seed=(0, 0),
):
    for index, seed in enumerate(seed_values):
        rows.append(
            make_game_row(
                seed=seed,
                training_episode=training_episode,
                agent=agent,
                opponent=opponent,
                game_id=index,
                profit_bb=profit_by_seed[index],
                won_game=win_by_seed[index],
                busted=busted_by_seed[index],
            )
        )


def write_sample_results_csv(path):
    rows = []
    add_group(
        rows,
        agent="adaptive_mc",
        opponent="calling",
        profit_by_seed=(18.0, 20.0),
    )
    add_group(
        rows,
        agent="rule_based",
        opponent="calling",
        profit_by_seed=(-1.0, -2.0),
        win_by_seed=(0, 0),
        busted_by_seed=(1, 1),
    )
    add_group(
        rows,
        agent="oracle_mc",
        opponent="calling",
        profit_by_seed=(19.0, 21.0),
    )
    for adaptive_agent, oracle_agent, adaptive_profit, oracle_profit in [
        (
            "adaptive_q_learning",
            "oracle_q_learning",
            (10.0, 12.0),
            (12.0, 14.0),
        ),
        (
            "adaptive_sarsa",
            "oracle_sarsa",
            (-3.0, -2.0),
            (0.0, 1.0),
        ),
        (
            "adaptive_double_q_learning",
            "oracle_double_q_learning",
            (5.0, 7.0),
            (7.0, 9.0),
        ),
    ]:
        add_group(
            rows,
            agent=adaptive_agent,
            opponent="calling",
            profit_by_seed=adaptive_profit,
        )
        add_group(
            rows,
            agent=oracle_agent,
            opponent="calling",
            profit_by_seed=oracle_profit,
        )
    add_group(
        rows,
        agent="always_raise",
        opponent="tight",
        profit_by_seed=(19.0, 20.0),
    )
    add_group(
        rows,
        agent="adaptive_mc",
        opponent="tight",
        profit_by_seed=(18.0, 19.0),
    )
    add_group(
        rows,
        agent="rule_based",
        opponent="tight",
        profit_by_seed=(17.0, 18.0),
    )
    add_group(
        rows,
        agent="adaptive_mc",
        opponent="aggressive",
        profit_by_seed=(2.0, 12.0),
    )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_agent_ranking_sorts_agents_per_opponent_and_training_episode():
    aggregated = pd.DataFrame(
        [
            {
                "training_run": "run",
                "opponent_name": "calling",
                "model_source": "final",
                "training_episode": 2000,
                "agent_name": "rule_based",
                "mean_profit_bb": -1.0,
                "win_rate": 40.0,
                "bust_rate": 10.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "model_source": "final",
                "training_episode": 2000,
                "agent_name": "adaptive_mc",
                "mean_profit_bb": 18.0,
                "win_rate": 95.0,
                "bust_rate": 2.0,
                "mean_profit_bb_std_across_seeds": 1.0,
            },
        ]
    )

    ranking = build_agent_ranking(aggregated)

    assert list(ranking["agent_name"]) == ["adaptive_mc", "rule_based"]
    assert list(ranking["rank"]) == [1, 2]


def test_add_baseline_deltas_calculates_rule_based_and_oracle_gap():
    ranking = pd.DataFrame(
        [
            {
                "training_run": "run",
                "opponent_name": "calling",
                "model_source": "final",
                "training_episode": 2000,
                "rank": 1,
                "agent_name": "adaptive_mc",
                "mean_profit_bb": 18.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "model_source": "final",
                "training_episode": 2000,
                "rank": 2,
                "agent_name": "rule_based",
                "mean_profit_bb": -1.0,
            },
            {
                "training_run": "run",
                "opponent_name": "calling",
                "model_source": "final",
                "training_episode": 2000,
                "rank": 3,
                "agent_name": "oracle_mc",
                "mean_profit_bb": 20.0,
            },
        ]
    )

    result = add_baseline_deltas(ranking)
    adaptive = result[result["agent_name"] == "adaptive_mc"].iloc[0]

    assert adaptive["delta_vs_rule_based"] == 19.0
    assert adaptive[ORACLE_GAP_BB_COLUMN] == 2.0
    assert "delta_vs_oracle" not in result.columns


def test_add_quality_flags_marks_ok_warning_and_fail():
    ranking = pd.DataFrame(
        [
            {
                "agent_name": "adaptive_mc",
                "mean_profit_bb": 5.0,
                "mean_profit_bb_std_across_seeds": 1.0,
                "win_rate": 80.0,
            },
            {
                "agent_name": "adaptive_mc",
                "mean_profit_bb": 5.0,
                "mean_profit_bb_std_across_seeds": 8.0,
                "win_rate": 80.0,
            },
            {
                "agent_name": "adaptive_mc",
                "mean_profit_bb": -1.0,
                "mean_profit_bb_std_across_seeds": 1.0,
                "win_rate": 40.0,
            },
        ]
    )

    result = add_quality_flags(ranking, SummaryThresholds())

    assert list(result["quality_status"]) == [
        QUALITY_OK,
        QUALITY_WARNING,
        QUALITY_FAIL,
    ]


def test_build_experiment_summary_creates_ranking_deltas_and_findings(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_sample_results_csv(csv_path)

    report, ranking, deltas, quality_flags = build_experiment_summary(csv_path)

    assert report.overview["summary_rows"] == len(ranking)
    assert "delta_vs_rule_based" in ranking.columns
    assert ORACLE_GAP_BB_COLUMN in deltas.columns
    assert "delta_vs_oracle" not in deltas.columns
    assert "quality_status" in quality_flags.columns
    assert SEED_STANDARD_ERROR_COLUMN in ranking.columns
    assert SEED_CI_LOWER_COLUMN in ranking.columns
    assert SEED_CI_UPPER_COLUMN in ranking.columns
    assert SEED_SPREAD_COLUMN in ranking.columns
    for algorithm_name in [
        "Monte Carlo",
        "Q-learning",
        "SARSA",
        "Double Q-learning",
    ]:
        assert any(
            f"Adaptive {algorithm_name} beats the rule-based baseline" in finding
            for finding in report.main_findings
        )
        assert any(
            f"Average Oracle gap (Oracle - adaptive) for Adaptive {algorithm_name}"
            in finding
            for finding in report.main_findings
        )


def test_write_experiment_summary_outputs_creates_markdown_json_csv_and_latex(tmp_path):
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "summary"
    write_sample_results_csv(csv_path)

    created_paths = write_experiment_summary_outputs(
        input_path=csv_path,
        output_dir=output_dir,
        report_format="all",
    )
    created_names = {path.name for path in created_paths}

    assert "experiment_summary.md" in created_names
    assert "experiment_summary.json" in created_names
    assert "mean_profit_ci_by_opponent.png" in created_names
    assert "seed_stability_by_opponent.png" in created_names
    assert "agent_ranking.csv" in created_names
    assert "deltas.csv" in created_names
    assert "quality_flags.csv" in created_names
    assert "agent_ranking.tex" in created_names

    summary_json = json.loads(
        (output_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )
    assert "main_findings" in summary_json
    assert SEED_CI_LOWER_COLUMN in summary_json["ranking"][0]

    ranking_csv = pd.read_csv(output_dir / "agent_ranking.csv")
    assert SEED_STANDARD_ERROR_COLUMN in ranking_csv.columns
    assert SEED_CI_LOWER_COLUMN in ranking_csv.columns
    assert ORACLE_GAP_BB_COLUMN in ranking_csv.columns
    assert "delta_vs_oracle" not in ranking_csv.columns

    markdown = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")
    assert "Oracle mean profit minus adaptive mean profit" in markdown
    assert "## Charts" in markdown
    assert "charts/mean_profit_ci_by_opponent.png" in markdown


def test_create_experiment_summary_parser_accepts_thresholds():
    args = parse_args(
        [
            "--input-path",
            "results/evaluation/results.csv",
            "--output-dir",
            "reports/summary",
            "--format",
            "both",
            "--max-std-across-seeds-bb",
            "7",
            "--no-export-latex",
            "--no-include-charts",
            "--chart-ci-multiplier",
            "2.0",
        ]
    )
    thresholds = build_thresholds(args)

    assert args.format == "both"
    assert args.export_latex is False
    assert args.include_charts is False
    assert args.chart_ci_multiplier == 2.0
    assert thresholds.max_std_across_seeds_bb == 7.0

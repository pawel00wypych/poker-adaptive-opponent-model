import json

import pandas as pd

from src.evaluation.experiment_validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationThresholds,
    render_validation_markdown,
    validate_checkpoint_results,
    write_validation_json_report,
    write_validation_markdown_report,
)
from src.experiments.validate_checkpoint_evaluation import (
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
    "first_classification_hand": 1,
    "first_correct_classification_hand": 1,
    "first_classification_action_count": 1,
    "first_correct_classification_action_count": 1,
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
            "checkpoint_episode": checkpoint,
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
    checkpoint=2000,
    seed_values=(42, 123),
    profit_by_seed=(10.0, 12.0),
    hands_played=20,
    win_rate=100.0,
    bust_rate=0.0,
    classifier_accuracy=0.0,
    classifier_coverage=0.0,
    policy_switches=0,
):
    for index, seed in enumerate(seed_values):
        profit_bb = profit_by_seed[index]
        won_game = 1 if win_rate >= 50.0 else 0
        busted = 1 if bust_rate >= 50.0 else 0
        classified_decisions = 10 if classifier_coverage > 0 else 0
        correct_classifications = (
            9 if classifier_accuracy >= 80.0 else 6
        ) if classified_decisions else 0
        incorrect_classifications = (
            classified_decisions - correct_classifications
        ) if classified_decisions else 0
        unknown_classifications = (
            0 if classifier_coverage >= 100.0 else 2
        ) if classified_decisions else 0

        rows.append(
            make_game_row(
                seed=seed,
                checkpoint=checkpoint,
                agent=agent,
                opponent=opponent,
                game_id=index,
                profit_bb=profit_bb,
                hands_played=hands_played,
                won_game=won_game,
                busted=busted,
                classified_decisions=classified_decisions,
                correct_classifications=correct_classifications,
                incorrect_classifications=incorrect_classifications,
                unknown_classifications=unknown_classifications,
                classifier_accuracy=classifier_accuracy / 100.0,
                classifier_coverage=classifier_coverage / 100.0,
                policy_switches=policy_switches,
            )
        )


def write_sample_checkpoint_csv(path):
    rows = []

    for opponent in ["aggressive", "calling", "fish"]:
        add_group(
            rows,
            agent="adaptive_mc",
            opponent=opponent,
            profit_by_seed=(14.0, 16.0) if opponent != "fish" else (19.8, 19.9),
            classifier_accuracy=100.0 if opponent != "fish" else 70.0,
            classifier_coverage=95.0,
            policy_switches=1,
        )
        add_group(
            rows,
            agent="oracle_adaptive",
            opponent=opponent,
            profit_by_seed=(15.0, 17.0) if opponent != "fish" else (19.8, 19.9),
            classifier_accuracy=100.0,
            classifier_coverage=100.0,
        )
        add_group(
            rows,
            agent="rule_based",
            opponent=opponent,
            profit_by_seed=(8.0, 9.0) if opponent != "fish" else (19.0, 19.2),
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_checkpoint_results_generates_expected_statuses(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)
    counts = report.status_counts()

    assert report.passed
    assert counts[STATUS_FAIL] == 0
    assert counts[STATUS_PASS] > 0
    assert counts[STATUS_WARNING] >= 1

    check_names = {
        check.check_name
        for check in report.checks
    }
    assert "Adaptive beats rule-based vs aggressive" in check_names
    assert "Adaptive beats rule-based vs calling" in check_names
    assert "Adaptive exploits FishPlayer" in check_names


def test_validate_checkpoint_results_fails_when_adaptive_loses_to_rule_based(
    tmp_path,
):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    df = pd.read_csv(csv_path)
    mask = (
        (df["agent_name"] == "adaptive_mc")
        & (df["opponent_name"] == "calling")
    )
    df.loc[mask, "profit_bb"] = -5.0
    df.to_csv(csv_path, index=False)

    report = validate_checkpoint_results(csv_path)

    failing_checks = [
        check
        for check in report.checks
        if check.status == STATUS_FAIL
    ]

    assert not report.passed
    assert any(
        check.check_name == "Adaptive beats rule-based vs calling"
        for check in failing_checks
    )


def test_validate_checkpoint_results_skips_missing_required_rows(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    df = pd.read_csv(csv_path)
    df = df[df["agent_name"] != "oracle_adaptive"]
    df.to_csv(csv_path, index=False)

    report = validate_checkpoint_results(csv_path)

    assert any(
        check.status == STATUS_SKIPPED
        and check.agent_name == "oracle_adaptive"
        for check in report.checks
    )


def test_validation_markdown_and_json_reports_are_written(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    output_dir = tmp_path / "validation"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)
    markdown_path = write_validation_markdown_report(report, output_dir)
    json_path = write_validation_json_report(report, output_dir)

    assert markdown_path.exists()
    assert json_path.exists()

    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "Experiment validation report" in markdown
    assert "Adaptive beats rule-based vs aggressive" in markdown
    assert payload["passed"] is True
    assert "checks" in payload


def test_render_validation_markdown_contains_thresholds(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)
    thresholds = ValidationThresholds(
        max_std_across_seeds_bb=2.5,
    )

    report = validate_checkpoint_results(
        csv_path,
        thresholds=thresholds,
    )
    markdown = render_validation_markdown(report)

    assert "max_std_across_seeds_bb" in markdown
    assert "2.5" in markdown


def test_validation_cli_parser_accepts_threshold_overrides():
    args = parse_args(
        [
            "--input-path",
            "results/evaluation/results.csv",
            "--output-dir",
            "reports/validation",
            "--format",
            "json",
            "--min-classifier-accuracy",
            "75",
            "--max-std-across-seeds-bb",
            "7",
        ]
    )
    thresholds = build_thresholds(args)

    assert args.format == "json"
    assert thresholds.min_classifier_accuracy == 75.0
    assert thresholds.max_std_across_seeds_bb == 7.0

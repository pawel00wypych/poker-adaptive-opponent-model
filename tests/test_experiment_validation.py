import json

import pandas as pd

from src.evaluation.experiment_validation import (
    STATUS_FAIL,
    VALIDATION_MODE_HEAD_TO_HEAD,
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

    for opponent in ["aggressive", "calling", "tight"]:
        add_group(
            rows,
            agent="adaptive_mc",
            opponent=opponent,
            profit_by_seed=(14.0, 16.0) if opponent != "tight" else (19.8, 19.9),
            classifier_accuracy=100.0 if opponent != "tight" else 70.0,
            classifier_coverage=95.0,
            policy_switches=1,
        )
        add_group(
            rows,
            agent="oracle_adaptive",
            opponent=opponent,
            profit_by_seed=(15.0, 17.0) if opponent != "tight" else (19.8, 19.9),
            classifier_accuracy=100.0,
            classifier_coverage=100.0,
        )
        add_group(
            rows,
            agent="rule_based",
            opponent=opponent,
            profit_by_seed=(8.0, 9.0) if opponent != "tight" else (19.0, 19.2),
        )
        add_group(
            rows,
            agent="always_raise",
            opponent=opponent,
            profit_by_seed=(19.3, 19.4)
            if opponent in {"aggressive", "tight"}
            else (-1.0, -0.8),
            win_rate=100.0
            if opponent in {"aggressive", "tight"}
            else 45.0,
            bust_rate=0.0
            if opponent in {"aggressive", "tight"}
            else 52.0,
        )

    pd.DataFrame(rows).to_csv(path, index=False)




def write_sample_head_to_head_csv(path):
    rows = []

    for agent, rule_based_profit in [
        ("policy_unknown", (10.0, 12.0)),
        ("adaptive_mc", (11.0, 13.0)),
        ("policy_tight", (-18.0, -17.0)),
        ("policy_aggressive", (-20.0, -20.0)),
        ("policy_calling", (15.0, 16.0)),
    ]:
        add_group(
            rows,
            agent=agent,
            opponent="rule_based",
            profit_by_seed=rule_based_profit,
            win_rate=80.0 if max(rule_based_profit) > 0 else 5.0,
            bust_rate=10.0 if max(rule_based_profit) > 0 else 95.0,
            classifier_accuracy=0.0,
            classifier_coverage=90.0 if agent == "adaptive_mc" else 0.0,
            policy_switches=1 if agent == "adaptive_mc" else 0,
        )

    for agent, always_raise_profit in [
        ("policy_unknown", (-19.0, -20.0)),
        ("adaptive_mc", (-16.0, -17.0)),
        ("policy_tight", (-20.0, -20.0)),
        ("policy_aggressive", (-12.0, -18.0)),
        ("policy_calling", (-20.0, -20.0)),
    ]:
        add_group(
            rows,
            agent=agent,
            opponent="always_raise",
            profit_by_seed=always_raise_profit,
            win_rate=5.0,
            bust_rate=95.0,
            classifier_accuracy=0.0,
            classifier_coverage=85.0 if agent == "adaptive_mc" else 0.0,
            policy_switches=1 if agent == "adaptive_mc" else 0,
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
    assert "Adaptive exploits TightPlayer" in check_names


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
            "--validation-mode",
            "head-to-head",
            "--min-classifier-accuracy",
            "75",
            "--max-std-across-seeds-bb",
            "7",
            "--always-raise-adaptive-warning-gap-bb",
            "4",
            "--high-always-raise-mean-profit-bb",
            "17",
            "--high-always-raise-win-rate",
            "90",
            "--min-head-to-head-mean-profit-bb",
            "1",
            "--max-adaptive-underperformance-vs-unknown-bb",
            "2",
            "--always-raise-stress-loss-bb",
            "-12",
            "--always-raise-stress-bust-rate",
            "75",
        ]
    )
    thresholds = build_thresholds(args)

    assert args.format == "json"
    assert args.validation_mode == "head-to-head"
    assert thresholds.min_classifier_accuracy == 75.0
    assert thresholds.max_std_across_seeds_bb == 7.0
    assert thresholds.always_raise_adaptive_warning_gap_bb == 4.0
    assert thresholds.high_always_raise_mean_profit_bb == 17.0
    assert thresholds.high_always_raise_win_rate == 90.0
    assert thresholds.min_head_to_head_mean_profit_bb == 1.0
    assert thresholds.max_adaptive_underperformance_vs_unknown_bb == 2.0
    assert thresholds.always_raise_stress_loss_bb == -12.0
    assert thresholds.always_raise_stress_bust_rate == 75.0



def test_validation_warns_when_always_raise_beats_adaptive_by_large_margin(
    tmp_path,
):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)

    checks = [
        check
        for check in report.checks
        if check.check_name
        == "Always-raise dominance sanity check vs aggressive"
    ]

    assert len(checks) == 1
    assert checks[0].status == STATUS_WARNING
    assert checks[0].observed_value > 3.0


def test_validation_warns_about_trivial_always_raise_exploit(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)

    warning_names = {
        check.check_name
        for check in report.checks
        if check.status == STATUS_WARNING
    }

    assert (
        "Always-raise trivial exploit sanity check vs aggressive"
        in warning_names
    )
    assert "Always-raise trivial exploit sanity check vs tight" in warning_names
    assert (
        "Always-raise trivial exploit sanity check vs calling"
        not in warning_names
    )


def test_validation_warns_when_tight_is_saturated_by_simple_baselines(
    tmp_path,
):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)

    checks = [
        check
        for check in report.checks
        if check.check_name == "TightPlayer baseline saturation sanity check"
    ]

    assert len(checks) == 1
    assert checks[0].status == STATUS_WARNING
    assert checks[0].opponent_name == "tight"
    assert "always_raise" in checks[0].details["saturated_agents"]


def test_head_to_head_validation_mode_uses_direct_matchup_checks(tmp_path):
    csv_path = tmp_path / "head_to_head_results.csv"
    write_sample_head_to_head_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
    )

    check_names = {check.check_name for check in report.checks}

    assert report.validation_mode == "head-to-head"
    assert "Fixed unknown policy beats RuleBasedPlayer" in check_names
    assert "Adaptive Monte Carlo beats RuleBasedPlayer" in check_names
    assert "At least one specialist beats RuleBasedPlayer" in check_names
    assert (
        "Adaptive not significantly worse than fixed unknown "
        "vs RuleBasedPlayer"
    ) in check_names
    assert "OOD classifier coverage vs rule_based" in check_names
    assert "OOD classifier coverage vs always_raise" in check_names
    assert "AlwaysRaise stress test vs adaptive_mc" in check_names
    assert "Adaptive exploits TightPlayer" not in check_names

    assert not any(
        check.status == STATUS_SKIPPED
        and check.opponent_name in {"tight", "aggressive", "calling"}
        for check in report.checks
    )


def test_head_to_head_validation_warns_for_always_raise_stress_test(
    tmp_path,
):
    csv_path = tmp_path / "head_to_head_results.csv"
    write_sample_head_to_head_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode="head-to-head",
    )

    stress_checks = [
        check
        for check in report.checks
        if check.category == "head_to_head_stress_test"
    ]

    assert stress_checks
    assert any(
        check.check_name == "AlwaysRaise stress test vs adaptive_mc"
        and check.status == STATUS_WARNING
        for check in stress_checks
    )


def test_head_to_head_validation_fails_when_adaptive_loses_to_rule_based(
    tmp_path,
):
    csv_path = tmp_path / "head_to_head_results.csv"
    write_sample_head_to_head_csv(csv_path)

    df = pd.read_csv(csv_path)
    mask = (
        (df["agent_name"] == "adaptive_mc")
        & (df["opponent_name"] == "rule_based")
    )
    df.loc[mask, "profit_bb"] = -5.0
    df.to_csv(csv_path, index=False)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode="head-to-head",
    )

    assert any(
        check.check_name == "Adaptive Monte Carlo beats RuleBasedPlayer"
        and check.status == STATUS_FAIL
        for check in report.checks
    )



def write_sample_generalization_csv(path):
    rows = []
    variants = [
        "strong_calling",
        "tight_extreme",
        "aggressive_extreme",
    ]

    adaptive_profits = {
        "strong_calling": (18.0, 18.5),
        "tight_extreme": (14.0, 14.5),
        "legacy_calling_variant": (8.0, 8.5),
        "tight_extreme": (6.0, 6.5),
        "aggressive_extreme": (-8.0, -7.5),
    }
    oracle_profits = {
        "strong_calling": (19.0, 19.5),
        "tight_extreme": (15.0, 15.5),
        "legacy_calling_variant": (9.0, 9.5),
        "tight_extreme": (10.5, 11.0),
        "aggressive_extreme": (-4.0, -3.5),
    }
    unknown_profits = {
        "strong_calling": (10.0, 10.5),
        "tight_extreme": (9.0, 9.5),
        "legacy_calling_variant": (7.0, 7.5),
        "tight_extreme": (5.0, 5.5),
        "aggressive_extreme": (-10.0, -9.5),
    }
    rule_based_profits = {
        "strong_calling": (12.0, 12.5),
        "tight_extreme": (10.0, 10.5),
        "legacy_calling_variant": (9.0, 9.5),
        "tight_extreme": (8.0, 8.5),
        "aggressive_extreme": (-6.0, -5.5),
    }
    always_raise_profits = {
        "strong_calling": (4.0, 4.5),
        "tight_extreme": (2.0, 2.5),
        "legacy_calling_variant": (-2.0, -1.5),
        "tight_extreme": (18.5, 19.0),
        "aggressive_extreme": (17.0, 17.5),
    }
    specialist_profits = {
        "policy_calling": {
            "strong_calling": (17.0, 17.5),
            "tight_extreme": (13.0, 13.5),
            "legacy_calling_variant": (9.0, 9.5),
            "tight_extreme": (-5.0, -4.5),
            "aggressive_extreme": (-12.0, -11.5),
        },
        "policy_aggressive": {
            "strong_calling": (-3.0, -2.5),
            "tight_extreme": (-2.0, -1.5),
            "legacy_calling_variant": (-1.0, -0.5),
            "tight_extreme": (7.0, 7.5),
            "aggressive_extreme": (-6.0, -5.5),
        },
        "policy_tight": {
            "strong_calling": (-1.0, -0.5),
            "tight_extreme": (-1.0, -0.5),
            "legacy_calling_variant": (-2.0, -1.5),
            "tight_extreme": (-7.0, -6.5),
            "aggressive_extreme": (-15.0, -14.5),
        },
    }

    for variant in variants:
        classifier_accuracy = 95.0
        classifier_coverage = 90.0
        bust_rate = 5.0

        if variant == "aggressive_extreme":
            classifier_accuracy = 0.0
            classifier_coverage = 0.0
            bust_rate = 90.0

        add_group(
            rows,
            agent="adaptive_mc",
            opponent=variant,
            profit_by_seed=adaptive_profits[variant],
            win_rate=80.0 if variant != "aggressive_extreme" else 10.0,
            bust_rate=bust_rate,
            classifier_accuracy=classifier_accuracy,
            classifier_coverage=classifier_coverage,
            policy_switches=2,
        )
        add_group(
            rows,
            agent="oracle_adaptive",
            opponent=variant,
            profit_by_seed=oracle_profits[variant],
            win_rate=85.0 if variant != "aggressive_extreme" else 30.0,
            bust_rate=10.0 if variant != "aggressive_extreme" else 70.0,
            classifier_accuracy=100.0,
            classifier_coverage=100.0,
        )
        add_group(
            rows,
            agent="policy_unknown",
            opponent=variant,
            profit_by_seed=unknown_profits[variant],
            win_rate=70.0 if variant != "aggressive_extreme" else 5.0,
            bust_rate=20.0 if variant != "aggressive_extreme" else 95.0,
        )
        add_group(
            rows,
            agent="rule_based",
            opponent=variant,
            profit_by_seed=rule_based_profits[variant],
            win_rate=70.0 if variant != "aggressive_extreme" else 20.0,
            bust_rate=20.0 if variant != "aggressive_extreme" else 80.0,
        )
        add_group(
            rows,
            agent="always_raise",
            opponent=variant,
            profit_by_seed=always_raise_profits[variant],
            win_rate=95.0 if variant == "tight_extreme" else 50.0,
            bust_rate=5.0 if variant == "tight_extreme" else 50.0,
        )

        for agent_name, profits_by_variant in specialist_profits.items():
            add_group(
                rows,
                agent=agent_name,
                opponent=variant,
                profit_by_seed=profits_by_variant[variant],
                win_rate=70.0 if max(profits_by_variant[variant]) > 0 else 20.0,
                bust_rate=10.0 if max(profits_by_variant[variant]) > 0 else 85.0,
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_generalization_validation_mode_uses_variant_checks(tmp_path):
    csv_path = tmp_path / "generalization_results.csv"
    write_sample_generalization_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode="generalization",
    )

    check_names = {check.check_name for check in report.checks}

    assert report.validation_mode == "generalization"
    assert "Adaptive positive on generalization variants" in check_names
    assert (
        "Adaptive beats fixed unknown on generalization variants"
        in check_names
    )
    assert (
        "Adaptive beats rule-based on generalization variants"
        in check_names
    )
    assert "Generalization oracle gap vs tight_extreme" in check_names
    assert (
        "Generalization classifier coverage vs aggressive_extreme"
        in check_names
    )
    assert "Aggressive extreme robustness check" in check_names
    assert "Adaptive exploits TightPlayer" not in check_names


def test_generalization_validation_warns_for_oracle_gap_and_classifier(
    tmp_path,
):
    csv_path = tmp_path / "generalization_results.csv"
    write_sample_generalization_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode="generalization",
    )

    warnings = {
        check.check_name
        for check in report.checks
        if check.status == STATUS_WARNING
    }

    assert "Generalization oracle gap vs tight_extreme" in warnings
    assert "Generalization oracle gap vs aggressive_extreme" in warnings
    assert (
        "Generalization classifier accuracy vs aggressive_extreme"
        in warnings
    )
    assert (
        "Generalization classifier coverage vs aggressive_extreme"
        in warnings
    )
    assert "Aggressive extreme robustness check" in warnings


def test_generalization_validation_fails_when_adaptive_is_not_positive(
    tmp_path,
):
    csv_path = tmp_path / "generalization_results.csv"
    write_sample_generalization_csv(csv_path)

    df = pd.read_csv(csv_path)
    mask = df["agent_name"] == "adaptive_mc"
    df.loc[mask, "profit_bb"] = -5.0
    df.to_csv(csv_path, index=False)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode="generalization",
    )

    assert any(
        check.check_name == "Adaptive positive on generalization variants"
        and check.status == STATUS_FAIL
        for check in report.checks
    )
    assert not report.passed


def test_generalization_validation_cli_parser_accepts_thresholds():
    args = parse_args(
        [
            "--input-path",
            "results/evaluation/generalization.csv",
            "--output-dir",
            "reports/generalization_validation",
            "--validation-mode",
            "generalization",
            "--min-generalization-positive-variants",
            "4",
            "--min-generalization-adaptive-beats-unknown-variants",
            "4",
            "--min-generalization-adaptive-beats-rule-based-variants",
            "2",
            "--max-generalization-oracle-gap-bb",
            "2.5",
            "--generalization-extreme-aggressive-min-profit-bb",
            "-8",
            "--generalization-extreme-aggressive-max-bust-rate",
            "90",
        ]
    )
    thresholds = build_thresholds(args)

    assert args.validation_mode == "generalization"
    assert thresholds.min_generalization_positive_variants == 4
    assert (
        thresholds.min_generalization_adaptive_beats_unknown_variants
        == 4
    )
    assert (
        thresholds.min_generalization_adaptive_beats_rule_based_variants
        == 2
    )
    assert thresholds.max_generalization_oracle_gap_bb == 2.5
    assert thresholds.generalization_extreme_aggressive_min_profit_bb == -8.0
    assert thresholds.generalization_extreme_aggressive_max_bust_rate == 90.0

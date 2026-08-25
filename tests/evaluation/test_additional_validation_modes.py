import pandas as pd

from src.evaluation.algorithm_metadata import ALGORITHM_VALIDATION_SPECS
from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    VALIDATION_MODE_BASELINE_SANITY,
    VALIDATION_MODE_STRESS_TEST,
    validate_evaluation_results,
)
from src.experiments.validation.validate_evaluation_results import (
    build_thresholds,
    parse_args,
)
from tests.evaluation.test_experiment_validation import add_group

SPEC = ALGORITHM_VALIDATION_SPECS[0]
BASELINES = ("always_call", "always_raise", "rule_based")


def write_sample_stress_test_csv(path):
    rows = []
    profits = {
        "always_call": {
            SPEC.adaptive_agent: (12.0, 13.0),
            SPEC.general_policy_agent: (9.0, 10.0),
        },
        "always_raise": {
            SPEC.adaptive_agent: (-16.0, -17.0),
            SPEC.general_policy_agent: (-5.0, -4.0),
        },
        "rule_based": {
            SPEC.adaptive_agent: (6.0, 7.0),
            SPEC.general_policy_agent: (4.0, 5.0),
        },
    }

    for opponent_name, profits_by_agent in profits.items():
        for agent_name, profit_by_seed in profits_by_agent.items():
            is_adaptive = agent_name == SPEC.adaptive_agent
            add_group(
                rows,
                agent=agent_name,
                opponent=opponent_name,
                profit_by_seed=profit_by_seed,
                win_rate=70.0 if max(profit_by_seed) >= 0.0 else 5.0,
                bust_rate=(
                    95.0 if opponent_name == "always_raise" and is_adaptive else 10.0
                ),
                classifier_accuracy=0.0,
                classifier_coverage=90.0 if is_adaptive else 0.0,
                policy_switches=1 if is_adaptive else 0,
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def write_sample_baseline_sanity_csv(path):
    rows = []
    mean_profits = {
        ("always_call", "always_call"): 0.1,
        ("always_raise", "always_raise"): -0.1,
        ("rule_based", "rule_based"): 0.0,
        ("always_call", "always_raise"): -5.0,
        ("always_raise", "always_call"): 5.2,
        ("always_call", "rule_based"): -2.0,
        ("rule_based", "always_call"): 2.0,
        ("always_raise", "rule_based"): 4.0,
        ("rule_based", "always_raise"): -4.5,
    }

    for (agent_name, opponent_name), mean_profit in mean_profits.items():
        for replicate_id, profit_bb in enumerate(
            (mean_profit - 0.1, mean_profit + 0.1)
        ):
            rows.append(
                {
                    "training_run": None,
                    "model_seed": None,
                    "model_source": None,
                    "training_episode": None,
                    "evaluation_replicate_id": replicate_id,
                    "experiment_id": f"evaluation_replicate_{replicate_id}",
                    "experiment_name": f"{agent_name}_vs_{opponent_name}",
                    "game_id": replicate_id,
                    "matchup_game_index": 0,
                    "evaluation_seed": 300_000 + replicate_id,
                    "agent_name": agent_name,
                    "opponent_name": opponent_name,
                    "final_stack": 200,
                    "initial_stack": 200,
                    "profit": 0,
                    "profit_bb": profit_bb,
                    "hands_played": 20,
                    "won_game": int(profit_bb > 0.0),
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
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_stress_test_mode_runs_dedicated_checks_and_coverage(tmp_path):
    csv_path = tmp_path / "stress_test.csv"
    write_sample_stress_test_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_STRESS_TEST,
        algorithm_specs=(SPEC,),
        require_all_algorithms=True,
    )
    checks_by_name = {check.check_name: check for check in report.checks}

    assert report.validation_mode == VALIDATION_MODE_STRESS_TEST
    assert (
        checks_by_name["Monte Carlo: Algorithm result coverage"].status == STATUS_PASS
    )
    assert (
        checks_by_name["Monte Carlo: Required matchup coverage"].details[
            "required_matchup_count"
        ]
        == 6
    )
    assert (
        checks_by_name["Monte Carlo: Adaptive exploits AlwaysCallPlayer"].status
        == STATUS_PASS
    )
    assert (
        checks_by_name["Monte Carlo: Fixed general policy beats RuleBasedPlayer"].status
        == STATUS_PASS
    )
    assert (
        checks_by_name["Monte Carlo: Adaptive resilience vs AlwaysRaisePlayer"].status
        == STATUS_WARNING
    )
    assert (
        "Monte Carlo: Adaptive not significantly worse than fixed general "
        "vs always_raise" in checks_by_name
    )
    assert (
        "Monte Carlo: Stress-test classifier coverage vs always_call" in checks_by_name
    )


def test_stress_test_mode_fails_required_matchup_coverage(tmp_path):
    csv_path = tmp_path / "stress_test.csv"
    write_sample_stress_test_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows = rows[
        ~(
            (rows["agent_name"] == SPEC.general_policy_agent)
            & (rows["opponent_name"] == "rule_based")
        )
    ]
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_STRESS_TEST,
        algorithm_specs=(SPEC,),
        require_all_algorithms=True,
    )
    coverage = next(
        check for check in report.checks if check.category == "matchup_coverage"
    )

    assert coverage.status == STATUS_FAIL
    assert coverage.details["missing_matchup_count"] == 1
    assert not report.passed


def test_stress_test_underperformance_is_a_non_blocking_diagnostic(
    tmp_path,
):
    csv_path = tmp_path / "stress_test.csv"
    write_sample_stress_test_csv(csv_path)
    rows = pd.read_csv(csv_path)
    mask = (rows["agent_name"] == SPEC.adaptive_agent) & (
        rows["opponent_name"] == "always_call"
    )
    rows.loc[mask, "profit_bb"] = -2.0
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_STRESS_TEST,
        algorithm_specs=(SPEC,),
    )
    check = next(
        check
        for check in report.checks
        if check.check_name == "Monte Carlo: Adaptive exploits AlwaysCallPlayer"
    )

    assert check.status == STATUS_WARNING
    assert report.technically_valid


def test_baseline_sanity_mode_validates_complete_matrix(tmp_path):
    csv_path = tmp_path / "baseline_sanity.csv"
    write_sample_baseline_sanity_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_BASELINE_SANITY,
    )
    categories = [check.category for check in report.checks]
    coverage = next(
        check
        for check in report.checks
        if check.category == "baseline_matchup_coverage"
    )

    assert report.validation_mode == VALIDATION_MODE_BASELINE_SANITY
    assert report.training_episode is None
    assert report.model_selection == "not_applicable"
    assert coverage.status == STATUS_PASS
    assert coverage.details["required_matchup_count"] == 9
    assert categories.count("baseline_mirror_neutrality") == 3
    assert categories.count("baseline_pair_reciprocity") == 3
    assert categories.count("baseline_extreme_result") == 4
    assert categories.count("evaluation_replicate_coverage") == 9
    assert categories.count("simulation_stability") == 9
    assert "seed_coverage" not in categories
    assert "seed_stability" not in categories


def test_baseline_sanity_mode_fails_when_a_matchup_is_missing(tmp_path):
    csv_path = tmp_path / "baseline_sanity.csv"
    write_sample_baseline_sanity_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows = rows[
        ~(
            (rows["agent_name"] == "rule_based")
            & (rows["opponent_name"] == "always_raise")
        )
    ]
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_BASELINE_SANITY,
    )
    coverage = next(
        check
        for check in report.checks
        if check.category == "baseline_matchup_coverage"
    )

    assert coverage.status == STATUS_FAIL
    assert coverage.details["missing_matchup_count"] == 1
    assert not report.passed


def test_baseline_sanity_warns_for_non_neutral_mirror(tmp_path):
    csv_path = tmp_path / "baseline_sanity.csv"
    write_sample_baseline_sanity_csv(csv_path)
    rows = pd.read_csv(csv_path)
    mask = (rows["agent_name"] == "always_call") & (
        rows["opponent_name"] == "always_call"
    )
    rows.loc[mask, "profit_bb"] = 3.0
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_BASELINE_SANITY,
    )
    check = next(
        check
        for check in report.checks
        if check.check_name == "Baseline mirror neutrality for always_call"
    )

    assert check.status == STATUS_WARNING
    assert check.observed_value == 3.0
    assert report.passed


def test_baseline_sanity_warns_for_non_reciprocal_pair(tmp_path):
    csv_path = tmp_path / "baseline_sanity.csv"
    write_sample_baseline_sanity_csv(csv_path)
    rows = pd.read_csv(csv_path)
    mask = (rows["agent_name"] == "always_call") & (
        rows["opponent_name"] == "always_raise"
    )
    rows.loc[mask, "profit_bb"] = 5.0
    rows.to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_BASELINE_SANITY,
    )
    check = next(
        check
        for check in report.checks
        if check.check_name
        == "Baseline pair reciprocity for always_call and always_raise"
    )

    assert check.status == STATUS_WARNING
    assert abs(check.observed_value - 10.2) < 1e-12


def test_cli_accepts_new_modes_and_baseline_thresholds():
    args = parse_args(
        [
            "--input-path",
            "results.csv",
            "--output-dir",
            "reports",
            "--validation-mode",
            VALIDATION_MODE_BASELINE_SANITY,
            "--max-baseline-mirror-abs-profit-bb",
            "1.5",
            "--max-baseline-pair-sum-abs-profit-bb",
            "2.5",
            "--min-evaluation-replicates-per-matchup",
            "3",
            "--max-std-across-evaluation-replicates-bb",
            "4.5",
        ]
    )
    thresholds = build_thresholds(args)
    stress_args = parse_args(
        [
            "--input-path",
            "results.csv",
            "--output-dir",
            "reports",
            "--validation-mode",
            VALIDATION_MODE_STRESS_TEST,
        ]
    )

    assert args.validation_mode == VALIDATION_MODE_BASELINE_SANITY
    assert stress_args.validation_mode == VALIDATION_MODE_STRESS_TEST
    assert thresholds.max_baseline_mirror_abs_profit_bb == 1.5
    assert thresholds.max_baseline_pair_sum_abs_profit_bb == 2.5
    assert thresholds.min_evaluation_replicates_per_matchup == 3
    assert thresholds.max_std_across_evaluation_replicates_bb == 4.5


def test_baseline_sanity_rejects_legacy_model_seed_rows(tmp_path):
    csv_path = tmp_path / "legacy_baseline_sanity.csv"
    rows = []
    add_group(
        rows,
        agent="always_call",
        opponent="always_call",
        profit_by_seed=(0.0, 0.1),
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_BASELINE_SANITY,
    )

    assert not report.technically_valid
    metadata_check = next(
        check
        for check in report.checks
        if check.check_id == "replicate_metadata_values"
    )
    assert metadata_check.status == STATUS_FAIL
    assert metadata_check.details["legacy_model_seed_rows"] == 2

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ALGORITHM_DOUBLE_Q_LEARNING,
    ALGORITHM_MONTE_CARLO,
    ALGORITHM_Q_LEARNING,
    ALGORITHM_SARSA,
    ALGORITHM_VALIDATION_SPECS,
    available_algorithm_specs,
)
from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_HEAD_TO_HEAD,
    ValidationThresholds,
)
from src.evaluation.validation.evaluation_validation import (
    validate_classifier_quality,
    validate_expected_algorithms_present,
    validate_oracle_not_worse_than_adaptive,
    validate_required_matchups_present,
)
from src.evaluation.validation.generalization_validation import (
    validate_generalization_adaptive_beats_agent,
)
from src.evaluation.validation.head_to_head_validation import (
    validate_adaptive_not_worse_than_general_rule_based,
)


def make_best_row(
    *,
    agent_name,
    opponent_name,
    mean_profit_bb=10.0,
    win_rate=80.0,
    bust_rate=10.0,
    classifier_accuracy=95.0,
    classifier_coverage=90.0,
):
    return {
        "training_run": "sample_run",
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "model_source": "final",
        "training_episode": 1000,
        "seeds": 2,
        "games": 20,
        "mean_profit_bb": mean_profit_bb,
        "mean_profit_bb_std_across_seeds": 1.0,
        "bb_per_100": 50.0,
        "win_rate": win_rate,
        "bust_rate": bust_rate,
        "global_classifier_accuracy": classifier_accuracy,
        "global_classifier_coverage": classifier_coverage,
        "mean_policy_switches": 1.0,
        "mean_hands_played": 20.0,
    }


def make_multi_algorithm_rows(opponents=("aggressive", "calling")):
    rows = []

    for spec in ALGORITHM_VALIDATION_SPECS:
        for opponent_name in opponents:
            rows.append(
                make_best_row(
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    mean_profit_bb=12.0,
                )
            )
            rows.append(
                make_best_row(
                    agent_name=spec.oracle_agent,
                    opponent_name=opponent_name,
                    mean_profit_bb=13.0,
                )
            )
            rows.append(
                make_best_row(
                    agent_name=spec.general_policy_agent,
                    opponent_name=opponent_name,
                    mean_profit_bb=9.0,
                )
            )

    return pd.DataFrame(rows)


def make_algorithm_matchup_rows(
    spec,
    opponents,
    roles=("adaptive", "oracle", "policy_general"),
):
    agents_by_role = {
        "adaptive": spec.adaptive_agent,
        "oracle": spec.oracle_agent,
        "policy_general": spec.general_policy_agent,
    }
    return pd.DataFrame(
        [
            make_best_row(
                agent_name=agents_by_role[role],
                opponent_name=opponent_name,
            )
            for role in roles
            for opponent_name in opponents
        ]
    )


def test_available_algorithm_specs_detects_present_algorithms_only():
    rows = pd.DataFrame(
        [
            make_best_row(
                agent_name=ALGORITHM_VALIDATION_SPECS[0].adaptive_agent,
                opponent_name="calling",
            ),
            make_best_row(
                agent_name=ALGORITHM_VALIDATION_SPECS[1].general_policy_agent,
                opponent_name="calling",
            ),
        ]
    )

    specs = available_algorithm_specs(rows)

    assert [spec.algorithm_name for spec in specs] == [
        ALGORITHM_MONTE_CARLO,
        ALGORITHM_Q_LEARNING,
    ]


def test_training_episode_coverage_requires_all_three_algorithm_roles():
    spec = ALGORITHM_VALIDATION_SPECS[0]
    rows = pd.DataFrame(
        [
            make_best_row(
                agent_name=spec.adaptive_agent,
                opponent_name="calling",
            )
        ]
    )

    check = validate_expected_algorithms_present(
        rows,
        (spec,),
        fail_when_missing=True,
    )[0]

    assert check.status == STATUS_FAIL
    assert check.details["required_roles"] == [
        "adaptive",
        "oracle",
        "policy_general",
    ]
    assert check.details["present_roles"] == ["adaptive"]
    assert check.details["missing_roles"] == ["oracle", "policy_general"]
    assert check.details["missing_agents"] == [
        spec.oracle_agent,
        spec.general_policy_agent,
    ]
    assert not check.details["present"]


def test_training_episode_coverage_passes_for_complete_algorithm_triplet():
    spec = ALGORITHM_VALIDATION_SPECS[2]
    rows = pd.DataFrame(
        [
            make_best_row(agent_name=agent_name, opponent_name="calling")
            for agent_name in (
                spec.adaptive_agent,
                spec.oracle_agent,
                spec.general_policy_agent,
            )
        ]
    )

    check = validate_expected_algorithms_present(
        rows,
        (spec,),
        fail_when_missing=True,
    )[0]

    assert check.status == STATUS_PASS
    assert check.details["present_agents"] == [
        spec.adaptive_agent,
        spec.oracle_agent,
        spec.general_policy_agent,
    ]
    assert check.details["missing_agents"] == []
    assert check.details["present"]


def test_generalization_coverage_warns_when_oracle_role_is_missing():
    spec = ALGORITHM_VALIDATION_SPECS[1]
    rows = pd.DataFrame(
        [
            make_best_row(agent_name=spec.adaptive_agent, opponent_name="calling"),
            make_best_row(
                agent_name=spec.general_policy_agent,
                opponent_name="calling",
            ),
        ]
    )

    check = validate_expected_algorithms_present(
        rows,
        (spec,),
        fail_when_missing=False,
        validation_mode=VALIDATION_MODE_GENERALIZATION,
    )[0]

    assert check.status == STATUS_WARNING
    assert check.details["missing_roles"] == ["oracle"]
    assert check.details["missing_agents"] == [spec.oracle_agent]


def test_head_to_head_coverage_does_not_require_oracle_role():
    spec = ALGORITHM_VALIDATION_SPECS[3]
    rows = pd.DataFrame(
        [
            make_best_row(
                agent_name=spec.adaptive_agent,
                opponent_name="rule_based",
            ),
            make_best_row(
                agent_name=spec.general_policy_agent,
                opponent_name="rule_based",
            ),
        ]
    )

    check = validate_expected_algorithms_present(
        rows,
        (spec,),
        fail_when_missing=True,
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
    )[0]

    assert check.status == STATUS_PASS
    assert check.details["required_roles"] == ["adaptive", "policy_general"]
    assert check.details["required_agents"] == [
        spec.adaptive_agent,
        spec.general_policy_agent,
    ]
    assert check.details["missing_agents"] == []


def test_head_to_head_results_use_mode_specific_coverage_roles(tmp_path):
    from src.evaluation.validation import validate_evaluation_results
    from tests.evaluation.test_experiment_validation import (
        write_sample_head_to_head_csv,
    )

    csv_path = tmp_path / "head_to_head_results.csv"
    write_sample_head_to_head_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
        algorithm_specs=(ALGORITHM_VALIDATION_SPECS[0],),
    )

    coverage_check = next(
        check for check in report.checks if check.category == "algorithm_coverage"
    )
    assert coverage_check.status == STATUS_PASS
    assert coverage_check.details["required_roles"] == [
        "adaptive",
        "policy_general",
    ]
    assert coverage_check.details["missing_agents"] == []

    matchup_coverage_check = next(
        check for check in report.checks if check.category == "matchup_coverage"
    )
    assert matchup_coverage_check.status == STATUS_PASS
    assert matchup_coverage_check.details["required_matchup_count"] == 4
    assert matchup_coverage_check.details["missing_matchups"] == []


def test_algorithm_coverage_passes_for_all_complete_algorithms():
    rows = make_multi_algorithm_rows(opponents=("calling",))

    checks = validate_expected_algorithms_present(
        rows,
        ALGORITHM_VALIDATION_SPECS,
        fail_when_missing=True,
    )

    assert len(checks) == 4
    assert all(check.status == STATUS_PASS for check in checks)
    assert all(check.details["missing_agents"] == [] for check in checks)


def test_training_episode_matchup_coverage_reports_exact_missing_pair():
    spec = ALGORITHM_VALIDATION_SPECS[0]
    rows = make_algorithm_matchup_rows(
        spec,
        ("tight", "aggressive", "calling"),
    )
    rows = rows[
        ~(
            (rows["agent_name"] == spec.oracle_agent)
            & (rows["opponent_name"] == "calling")
        )
    ]

    check = validate_required_matchups_present(
        rows,
        (spec,),
        fail_when_missing=True,
    )[0]

    assert check.status == STATUS_FAIL
    assert check.category == "matchup_coverage"
    assert check.details["required_matchup_count"] == 9
    assert check.details["present_matchup_count"] == 8
    assert check.details["missing_matchup_count"] == 1
    assert check.details["missing_matchups"] == [
        {
            "role": "oracle",
            "agent_name": spec.oracle_agent,
            "opponent_name": "calling",
        }
    ]
    assert "oracle_mc vs calling" in check.message


def test_generalization_matchup_coverage_uses_all_extreme_opponents():
    spec = ALGORITHM_VALIDATION_SPECS[1]
    opponents = (
        "calling_extreme",
        "aggressive_extreme",
        "tight_extreme",
    )
    rows = make_algorithm_matchup_rows(spec, opponents)

    check = validate_required_matchups_present(
        rows,
        (spec,),
        fail_when_missing=True,
        validation_mode=VALIDATION_MODE_GENERALIZATION,
    )[0]

    assert check.status == STATUS_PASS
    assert check.details["required_opponents"] == list(opponents)
    assert check.details["required_matchup_count"] == 9
    assert check.details["missing_matchups"] == []


def test_generalization_results_use_mode_specific_matchup_coverage(tmp_path):
    from src.evaluation.validation import validate_evaluation_results
    from tests.evaluation.test_experiment_validation import (
        write_sample_generalization_csv,
    )

    csv_path = tmp_path / "generalization_results.csv"
    write_sample_generalization_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        validation_mode=VALIDATION_MODE_GENERALIZATION,
        algorithm_specs=(ALGORITHM_VALIDATION_SPECS[0],),
    )

    matchup_coverage_check = next(
        check for check in report.checks if check.category == "matchup_coverage"
    )
    assert matchup_coverage_check.status == STATUS_PASS
    assert matchup_coverage_check.details["required_matchup_count"] == 9
    assert matchup_coverage_check.details["missing_matchups"] == []


def test_head_to_head_matchup_coverage_requires_four_algorithm_pairs():
    spec = ALGORITHM_VALIDATION_SPECS[3]
    rows = make_algorithm_matchup_rows(
        spec,
        ("rule_based", "always_raise"),
        roles=("adaptive", "policy_general"),
    )

    check = validate_required_matchups_present(
        rows,
        (spec,),
        fail_when_missing=True,
        validation_mode=VALIDATION_MODE_HEAD_TO_HEAD,
    )[0]

    assert check.status == STATUS_PASS
    assert check.details["required_roles"] == ["adaptive", "policy_general"]
    assert check.details["required_opponents"] == [
        "rule_based",
        "always_raise",
    ]
    assert check.details["required_matchup_count"] == 4
    assert check.details["missing_matchups"] == []


def test_selected_matchup_coverage_warns_in_non_strict_mode():
    spec = ALGORITHM_VALIDATION_SPECS[2]
    rows = make_algorithm_matchup_rows(
        spec,
        ("tight", "aggressive"),
    )

    check = validate_required_matchups_present(
        rows,
        (spec,),
        fail_when_missing=False,
    )[0]

    assert check.status == STATUS_WARNING
    assert check.details["missing_matchup_count"] == 3
    assert {matchup["agent_name"] for matchup in check.details["missing_matchups"]} == {
        spec.adaptive_agent,
        spec.oracle_agent,
        spec.general_policy_agent,
    }
    assert {
        matchup["opponent_name"] for matchup in check.details["missing_matchups"]
    } == {"calling"}


def test_oracle_gap_validation_runs_for_all_present_algorithms():
    rows = make_multi_algorithm_rows(opponents=("aggressive",))

    checks = validate_oracle_not_worse_than_adaptive(
        rows,
        ValidationThresholds(),
        opponents=("aggressive",),
    )

    assert {check.algorithm_name for check in checks} == {
        ALGORITHM_MONTE_CARLO,
        ALGORITHM_Q_LEARNING,
        ALGORITHM_SARSA,
        ALGORITHM_DOUBLE_Q_LEARNING,
    }
    assert len(checks) == 4
    assert all(check.status == STATUS_PASS for check in checks)
    assert all(check.details["oracle_agent"] for check in checks)
    assert all(check.details["adaptive_agent"] for check in checks)


def test_classifier_quality_validation_runs_for_all_adaptive_algorithms():
    rows = make_multi_algorithm_rows(opponents=("calling",))

    checks = validate_classifier_quality(
        rows,
        ValidationThresholds(),
        opponents=("calling",),
    )

    accuracy_checks = [
        check
        for check in checks
        if check.details["metric"] == "global_classifier_accuracy"
    ]
    coverage_checks = [
        check
        for check in checks
        if check.details["metric"] == "global_classifier_coverage"
    ]

    assert len(accuracy_checks) == 4
    assert len(coverage_checks) == 4
    assert {check.agent_name for check in accuracy_checks} == {
        spec.adaptive_agent for spec in ALGORITHM_VALIDATION_SPECS
    }


def test_generalization_adaptive_vs_general_uses_matching_algorithm_policy():
    rows = make_multi_algorithm_rows(opponents=("calling_extreme",))

    checks = validate_generalization_adaptive_beats_agent(
        rows,
        ValidationThresholds(),
        min_successful_variants=1,
        check_name="Adaptive beats fixed general on generalization variants",
        category="generalization_adaptive_delta_vs_general",
        opponents=("calling_extreme",),
    )

    assert len(checks) == 4
    for check in checks:
        spec = next(
            spec
            for spec in ALGORITHM_VALIDATION_SPECS
            if spec.algorithm_name == check.algorithm_name
        )
        assert check.agent_name == spec.adaptive_agent
        assert check.details["baseline_agent_name"] == spec.general_policy_agent
        assert check.status == STATUS_PASS


def test_head_to_head_adaptive_vs_general_uses_matching_algorithm_policy():
    rows = make_multi_algorithm_rows(opponents=("rule_based",))

    checks = validate_adaptive_not_worse_than_general_rule_based(
        rows,
        ValidationThresholds(),
    )

    assert len(checks) == 4
    for check in checks:
        spec = next(
            spec
            for spec in ALGORITHM_VALIDATION_SPECS
            if spec.algorithm_name == check.algorithm_name
        )
        assert check.agent_name == spec.adaptive_agent
        assert check.details["general_policy_agent"] == spec.general_policy_agent
        assert check.status == STATUS_PASS


def test_missing_algorithm_rows_do_not_create_unrelated_checks():
    rows = pd.DataFrame(
        [
            make_best_row(
                agent_name=ALGORITHM_VALIDATION_SPECS[0].adaptive_agent,
                opponent_name="calling",
            ),
            make_best_row(
                agent_name=ALGORITHM_VALIDATION_SPECS[0].oracle_agent,
                opponent_name="calling",
            ),
        ]
    )

    checks = validate_oracle_not_worse_than_adaptive(
        rows,
        ValidationThresholds(),
        opponents=("calling",),
    )

    assert len(checks) == 1
    assert checks[0].algorithm_name == ALGORITHM_MONTE_CARLO


def test_validate_evaluation_results_require_all_algorithms_fails_when_missing(
    tmp_path,
):
    from src.evaluation.validation import (
        STATUS_FAIL,
        validate_evaluation_results,
    )
    from tests.evaluation.test_experiment_validation import (
        write_sample_final_model_csv,
    )

    csv_path = tmp_path / "training_episode_results.csv"
    write_sample_final_model_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        require_all_algorithms=True,
    )

    coverage_checks = [
        check for check in report.checks if check.category == "algorithm_coverage"
    ]

    assert len(coverage_checks) == 4
    assert any(
        check.algorithm_name == ALGORITHM_Q_LEARNING and check.status == STATUS_FAIL
        for check in coverage_checks
    )
    matchup_coverage_checks = [
        check for check in report.checks if check.category == "matchup_coverage"
    ]
    assert len(matchup_coverage_checks) == 4
    q_learning_matchup_coverage = next(
        check
        for check in matchup_coverage_checks
        if check.algorithm_name == ALGORITHM_Q_LEARNING
    )
    assert q_learning_matchup_coverage.status == STATUS_FAIL
    assert q_learning_matchup_coverage.details["missing_matchup_count"] == 9
    assert not report.passed


def test_validate_evaluation_results_selected_algorithms_adds_coverage_warnings(
    tmp_path,
):
    from src.evaluation.algorithm_metadata import (
        ALGORITHM_VALIDATION_SPEC_BY_KEY,
    )
    from src.evaluation.validation import (
        STATUS_WARNING,
        validate_evaluation_results,
    )
    from tests.evaluation.test_experiment_validation import (
        write_sample_final_model_csv,
    )

    csv_path = tmp_path / "training_episode_results.csv"
    write_sample_final_model_csv(csv_path)

    report = validate_evaluation_results(
        csv_path,
        algorithm_specs=(
            ALGORITHM_VALIDATION_SPEC_BY_KEY["monte_carlo"],
            ALGORITHM_VALIDATION_SPEC_BY_KEY["q_learning"],
        ),
    )

    coverage_checks = [
        check for check in report.checks if check.category == "algorithm_coverage"
    ]

    assert len(coverage_checks) == 2
    assert any(
        check.algorithm_name == ALGORITHM_Q_LEARNING and check.status == STATUS_WARNING
        for check in coverage_checks
    )
    monte_carlo_check = next(
        check
        for check in coverage_checks
        if check.algorithm_name == ALGORITHM_MONTE_CARLO
    )
    assert monte_carlo_check.status == STATUS_WARNING
    assert monte_carlo_check.details["missing_roles"] == ["policy_general"]
    assert monte_carlo_check.details["missing_agents"] == ["policy_general_mc"]

    matchup_coverage_checks = [
        check for check in report.checks if check.category == "matchup_coverage"
    ]
    assert len(matchup_coverage_checks) == 2
    monte_carlo_matchup_coverage = next(
        check
        for check in matchup_coverage_checks
        if check.algorithm_name == ALGORITHM_MONTE_CARLO
    )
    assert monte_carlo_matchup_coverage.status == STATUS_WARNING
    assert monte_carlo_matchup_coverage.details["missing_matchup_count"] == 3
    assert {
        matchup["agent_name"]
        for matchup in monte_carlo_matchup_coverage.details["missing_matchups"]
    } == {"policy_general_mc"}

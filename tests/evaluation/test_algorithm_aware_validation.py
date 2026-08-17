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
    STATUS_PASS,
    ValidationThresholds,
)
from src.evaluation.validation.checkpoint_validation import (
    validate_classifier_quality,
    validate_oracle_not_worse_than_adaptive,
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
        "checkpoint_episode": 1000,
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


def test_validate_checkpoint_results_require_all_algorithms_fails_when_missing(tmp_path):
    from tests.evaluation.test_experiment_validation import (
        write_sample_checkpoint_csv,
    )
    from src.evaluation.validation import (
        STATUS_FAIL,
        validate_checkpoint_results,
    )

    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        require_all_algorithms=True,
    )

    coverage_checks = [
        check
        for check in report.checks
        if check.category == "algorithm_coverage"
    ]

    assert len(coverage_checks) == 4
    assert any(
        check.algorithm_name == ALGORITHM_Q_LEARNING
        and check.status == STATUS_FAIL
        for check in coverage_checks
    )
    assert not report.passed


def test_validate_checkpoint_results_selected_algorithms_adds_coverage_warnings(tmp_path):
    from tests.evaluation.test_experiment_validation import (
        write_sample_checkpoint_csv,
    )
    from src.evaluation.algorithm_metadata import (
        ALGORITHM_VALIDATION_SPEC_BY_KEY,
    )
    from src.evaluation.validation import (
        STATUS_WARNING,
        validate_checkpoint_results,
    )

    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        algorithm_specs=(
            ALGORITHM_VALIDATION_SPEC_BY_KEY["monte_carlo"],
            ALGORITHM_VALIDATION_SPEC_BY_KEY["q_learning"],
        ),
    )

    coverage_checks = [
        check
        for check in report.checks
        if check.category == "algorithm_coverage"
    ]

    assert len(coverage_checks) == 2
    assert any(
        check.algorithm_name == ALGORITHM_Q_LEARNING
        and check.status == STATUS_WARNING
        for check in coverage_checks
    )

import pandas as pd
import pytest

from src.evaluation.validation import (
    ALGORITHM_MONTE_CARLO,
    STATUS_FAIL,
    STATUS_PASS,
    VALIDATION_MODE_CHECKPOINT,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_HEAD_TO_HEAD,
    ValidationThresholds,
    validate_checkpoint_results,
    validate_minimum_seed_coverage,
)
from tests.evaluation.test_experiment_validation import (
    write_sample_checkpoint_csv,
    write_sample_generalization_csv,
    write_sample_head_to_head_csv,
)


def test_minimum_seed_coverage_reports_pass_and_fail_per_matchup():
    rows = pd.DataFrame(
        [
            {
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "seeds": 1,
            },
            {
                "agent_name": "rule_based",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
                "seeds": 3,
            },
        ]
    )

    checks = validate_minimum_seed_coverage(
        rows,
        ValidationThresholds(min_seeds_per_matchup=2),
    )
    checks_by_agent = {check.agent_name: check for check in checks}

    adaptive_check = checks_by_agent["adaptive_mc"]
    assert adaptive_check.status == STATUS_FAIL
    assert adaptive_check.algorithm_name == ALGORITHM_MONTE_CARLO
    assert adaptive_check.observed_value == 1.0
    assert adaptive_check.threshold == 2.0
    assert adaptive_check.details == {
        "seed_count": 1,
        "min_seeds_per_matchup": 2,
        "missing_seed_count": 1,
    }

    rule_based_check = checks_by_agent["rule_based"]
    assert rule_based_check.status == STATUS_PASS
    assert rule_based_check.details["missing_seed_count"] == 0


def test_minimum_seed_coverage_treats_missing_seed_count_as_zero():
    rows = pd.DataFrame(
        [
            {
                "agent_name": "adaptive_mc",
                "opponent_name": "calling",
                "checkpoint_episode": 1000,
            }
        ]
    )

    check = validate_minimum_seed_coverage(
        rows,
        ValidationThresholds(),
    )[0]

    assert check.status == STATUS_FAIL
    assert check.observed_value == 0.0
    assert check.details["missing_seed_count"] == 2


def test_minimum_seed_coverage_rejects_non_positive_threshold():
    with pytest.raises(
        ValueError,
        match="min_seeds_per_matchup must be at least 1",
    ):
        validate_minimum_seed_coverage(
            pd.DataFrame(),
            ValidationThresholds(min_seeds_per_matchup=0),
        )


@pytest.mark.parametrize(
    ("writer", "validation_mode"),
    [
        (write_sample_checkpoint_csv, VALIDATION_MODE_CHECKPOINT),
        (write_sample_generalization_csv, VALIDATION_MODE_GENERALIZATION),
        (write_sample_head_to_head_csv, VALIDATION_MODE_HEAD_TO_HEAD),
    ],
)
def test_validation_modes_include_passing_seed_coverage_checks(
    tmp_path,
    writer,
    validation_mode,
):
    csv_path = tmp_path / f"{validation_mode}_results.csv"
    writer(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode=validation_mode,
    )
    seed_coverage_checks = [
        check
        for check in report.checks
        if check.category == "seed_coverage"
    ]

    assert seed_coverage_checks
    assert all(check.status == STATUS_PASS for check in seed_coverage_checks)


def test_validation_fails_when_matchups_have_only_one_seed(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)
    rows = pd.read_csv(csv_path)
    rows[rows["model_seed"] == 42].to_csv(csv_path, index=False)

    report = validate_checkpoint_results(csv_path)
    seed_coverage_checks = [
        check
        for check in report.checks
        if check.category == "seed_coverage"
    ]

    assert seed_coverage_checks
    assert all(check.status == STATUS_FAIL for check in seed_coverage_checks)
    assert all(check.observed_value == 1.0 for check in seed_coverage_checks)
    assert not report.passed


def test_custom_minimum_seed_threshold_is_enforced(tmp_path):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        thresholds=ValidationThresholds(min_seeds_per_matchup=3),
    )
    seed_coverage_checks = [
        check
        for check in report.checks
        if check.category == "seed_coverage"
    ]

    assert seed_coverage_checks
    assert all(check.status == STATUS_FAIL for check in seed_coverage_checks)
    assert all(check.observed_value == 2.0 for check in seed_coverage_checks)
    assert not report.passed

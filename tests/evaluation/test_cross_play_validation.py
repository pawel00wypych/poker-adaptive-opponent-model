import pandas as pd

from src.evaluation.algorithm_metadata import ALGORITHM_VALIDATION_SPECS
from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    VALIDATION_MODE_CROSS_PLAY,
    ValidationThresholds,
    validate_checkpoint_results,
    validate_required_matchups_present,
)
from src.evaluation.validation.cross_play_validation import (
    validate_cross_play_pair_reciprocity,
)
from src.experiments.validation.validate_checkpoint_evaluation import (
    build_thresholds,
    parse_args,
)
from tests.evaluation.test_experiment_validation import add_group


def write_sample_cross_play_csv(path):
    rows = []

    for first_index, first_spec in enumerate(ALGORITHM_VALIDATION_SPECS):
        for second_index, second_spec in enumerate(ALGORITHM_VALIDATION_SPECS):
            if first_spec == second_spec:
                continue

            mean_profit = float(second_index - first_index)
            add_group(
                rows,
                agent=first_spec.adaptive_agent,
                opponent=second_spec.adaptive_agent,
                profit_by_seed=(mean_profit - 0.1, mean_profit + 0.1),
                win_rate=60.0 if mean_profit > 0.0 else 40.0,
                bust_rate=10.0,
                classifier_coverage=90.0,
                policy_switches=1,
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def make_comparison_row(
    agent_name: str,
    opponent_name: str,
    checkpoint_episode: int,
    mean_profit_bb: float,
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "checkpoint_episode": checkpoint_episode,
        "mean_profit_bb": mean_profit_bb,
    }


def test_cross_play_mode_validates_required_adaptive_matrix(tmp_path):
    csv_path = tmp_path / "cross_play.csv"
    write_sample_cross_play_csv(csv_path)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode=VALIDATION_MODE_CROSS_PLAY,
    )
    coverage = next(
        check
        for check in report.checks
        if check.category == "cross_play_matchup_coverage"
    )
    categories = [check.category for check in report.checks]

    assert report.validation_mode == VALIDATION_MODE_CROSS_PLAY
    assert report.passed
    assert coverage.status == STATUS_PASS
    assert coverage.details["required_matchup_count"] == 12
    assert coverage.details["missing_matchup_count"] == 0
    assert categories.count("cross_play_pair_reciprocity") == 6
    assert categories.count("cross_play_classifier_coverage") == 12
    assert categories.count("seed_coverage") == 12


def test_cross_play_mode_fails_when_directed_matchup_is_missing(tmp_path):
    csv_path = tmp_path / "cross_play.csv"
    write_sample_cross_play_csv(csv_path)
    rows = pd.read_csv(csv_path)
    first_spec, second_spec = ALGORITHM_VALIDATION_SPECS[:2]
    rows = rows[
        ~(
            (rows["agent_name"] == first_spec.adaptive_agent)
            & (rows["opponent_name"] == second_spec.adaptive_agent)
        )
    ]
    rows.to_csv(csv_path, index=False)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode=VALIDATION_MODE_CROSS_PLAY,
    )
    coverage = next(
        check
        for check in report.checks
        if check.category == "cross_play_matchup_coverage"
    )

    assert coverage.status == STATUS_FAIL
    assert coverage.details["missing_matchup_count"] == 1
    assert coverage.details["missing_matchups"] == [
        {
            "agent_name": first_spec.adaptive_agent,
            "opponent_name": second_spec.adaptive_agent,
        }
    ]
    assert not report.passed


def test_selected_cross_play_coverage_can_warn_for_missing_matchup():
    first_spec, second_spec = ALGORITHM_VALIDATION_SPECS[:2]
    best_rows = pd.DataFrame(
        [
            {
                "agent_name": first_spec.adaptive_agent,
                "opponent_name": second_spec.adaptive_agent,
            }
        ]
    )

    check = validate_required_matchups_present(
        best_rows,
        (first_spec, second_spec),
        fail_when_missing=False,
        validation_mode=VALIDATION_MODE_CROSS_PLAY,
    )[0]

    assert check.status == STATUS_WARNING
    assert check.details["missing_matchups"] == [
        {
            "agent_name": second_spec.adaptive_agent,
            "opponent_name": first_spec.adaptive_agent,
        }
    ]


def test_cross_play_reciprocity_uses_latest_common_checkpoint():
    first_spec, second_spec = ALGORITHM_VALIDATION_SPECS[:2]
    rows = pd.DataFrame(
        [
            make_comparison_row(
                first_spec.adaptive_agent,
                second_spec.adaptive_agent,
                1000,
                8.0,
            ),
            make_comparison_row(
                second_spec.adaptive_agent,
                first_spec.adaptive_agent,
                1000,
                -8.0,
            ),
            make_comparison_row(
                first_spec.adaptive_agent,
                second_spec.adaptive_agent,
                2000,
                5.0,
            ),
            make_comparison_row(
                second_spec.adaptive_agent,
                first_spec.adaptive_agent,
                2000,
                1.0,
            ),
        ]
    )

    check = validate_cross_play_pair_reciprocity(
        rows,
        ValidationThresholds(
            max_cross_play_pair_sum_abs_profit_bb=2.0,
        ),
        (first_spec, second_spec),
    )[0]

    assert check.status == STATUS_WARNING
    assert check.checkpoint_episode == 2000
    assert check.observed_value == 6.0
    assert check.details["pair_sum_bb"] == 6.0


def test_cross_play_validates_optional_same_algorithm_general_pair(tmp_path):
    csv_path = tmp_path / "cross_play.csv"
    write_sample_cross_play_csv(csv_path)
    rows = pd.read_csv(csv_path).to_dict("records")
    spec = ALGORITHM_VALIDATION_SPECS[0]
    add_group(
        rows,
        agent=spec.adaptive_agent,
        opponent=spec.general_policy_agent,
        profit_by_seed=(2.9, 3.1),
        classifier_coverage=90.0,
    )
    add_group(
        rows,
        agent=spec.general_policy_agent,
        opponent=spec.adaptive_agent,
        profit_by_seed=(-3.1, -2.9),
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report = validate_checkpoint_results(
        csv_path,
        validation_mode=VALIDATION_MODE_CROSS_PLAY,
    )
    check = next(
        check
        for check in report.checks
        if check.check_name
        == (
            "Cross-play reciprocity for Monte Carlo adaptive and "
            "Monte Carlo fixed general"
        )
    )

    assert check.status == STATUS_PASS
    assert check.algorithm_name == "Monte Carlo"
    assert check.details["required_matchup"] is False


def test_cli_accepts_cross_play_mode_and_reciprocity_threshold():
    args = parse_args(
        [
            "--input-path",
            "results.csv",
            "--output-dir",
            "reports",
            "--validation-mode",
            VALIDATION_MODE_CROSS_PLAY,
            "--max-cross-play-pair-sum-abs-profit-bb",
            "3.5",
        ]
    )
    thresholds = build_thresholds(args)

    assert args.validation_mode == VALIDATION_MODE_CROSS_PLAY
    assert thresholds.max_cross_play_pair_sum_abs_profit_bb == 3.5

import pandas as pd
import pytest

from src.evaluation.algorithm_metadata import ALGORITHM_VALIDATION_SPECS
from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationCheckResult,
    ValidationThresholds,
    render_validation_markdown,
    validate_adaptive_beats_rule_based,
    validate_checkpoint_results,
    validate_oracle_not_worse_than_adaptive,
)
from src.evaluation.validation.common import ValidationReport
from src.evaluation.validation.generalization_validation import (
    validate_generalization_adaptive_beats_agent,
)
from tests.evaluation.test_experiment_validation import (
    write_sample_checkpoint_csv,
)

SPEC = ALGORITHM_VALIDATION_SPECS[0]


def aggregate_row(
    agent_name: str,
    mean_profit_bb: float,
    *,
    opponent_name: str = "calling",
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "checkpoint_episode": 1000,
        "mean_profit_bb": mean_profit_bb,
    }


def seed_row(
    agent_name: str,
    model_seed: int,
    mean_profit_bb: float,
    *,
    opponent_name: str = "calling",
) -> dict[str, object]:
    return {
        **aggregate_row(
            agent_name,
            mean_profit_bb,
            opponent_name=opponent_name,
        ),
        "model_seed": model_seed,
    }


def test_adaptive_rule_based_check_uses_ci_of_paired_seed_deltas():
    aggregated = pd.DataFrame(
        [
            aggregate_row(SPEC.adaptive_agent, 15.0),
            aggregate_row("rule_based", 13.0),
        ]
    )
    seed_rows = pd.DataFrame(
        [
            seed_row(SPEC.adaptive_agent, 1, 5.0),
            seed_row(SPEC.adaptive_agent, 2, 15.0),
            seed_row(SPEC.adaptive_agent, 3, 25.0),
            seed_row("rule_based", 1, 23.0),
            seed_row("rule_based", 2, 13.0),
            seed_row("rule_based", 3, 3.0),
        ]
    )

    check = validate_adaptive_beats_rule_based(
        aggregated,
        ValidationThresholds(),
        opponents=("calling",),
        algorithm_specs=(SPEC,),
        seed_rows=seed_rows,
    )[0]

    assert check.status == STATUS_WARNING
    assert check.observed_value == pytest.approx(2.0)
    assert check.sample_size == 3
    assert check.standard_error == pytest.approx(11.547005383792516)
    assert check.ci_lower < 0.0 < check.ci_upper
    assert check.details["paired_seed_statistics"]["deltas_by_seed"] == {
        "1": -18.0,
        "2": 2.0,
        "3": 22.0,
    }


def test_paired_comparison_fails_without_enough_common_seeds():
    aggregated = pd.DataFrame(
        [
            aggregate_row(SPEC.adaptive_agent, 5.0),
            aggregate_row("rule_based", 4.0),
        ]
    )
    seed_rows = pd.DataFrame(
        [
            seed_row(SPEC.adaptive_agent, 1, 5.0),
            seed_row(SPEC.adaptive_agent, 2, 5.0),
            seed_row("rule_based", 3, 4.0),
            seed_row("rule_based", 4, 4.0),
        ]
    )

    check = validate_adaptive_beats_rule_based(
        aggregated,
        ValidationThresholds(min_seeds_per_matchup=2),
        opponents=("calling",),
        algorithm_specs=(SPEC,),
        seed_rows=seed_rows,
    )[0]

    assert check.status == STATUS_FAIL
    assert check.sample_size == 0
    assert "0 common model seed(s)" in check.message
    paired = check.details["paired_seed_statistics"]
    assert paired["left_only_seeds"] == [1, 2]
    assert paired["right_only_seeds"] == [3, 4]


def test_paired_comparison_skips_when_one_agent_has_no_seed_rows():
    aggregated = pd.DataFrame(
        [
            aggregate_row(SPEC.adaptive_agent, 5.0),
            aggregate_row("rule_based", 4.0),
        ]
    )
    seed_rows = pd.DataFrame(
        [
            seed_row(SPEC.adaptive_agent, 1, 5.0),
            seed_row(SPEC.adaptive_agent, 2, 6.0),
        ]
    )

    check = validate_adaptive_beats_rule_based(
        aggregated,
        ValidationThresholds(),
        opponents=("calling",),
        algorithm_specs=(SPEC,),
        seed_rows=seed_rows,
    )[0]

    assert check.status == STATUS_SKIPPED
    assert check.sample_size == 0
    assert "Missing seed-level rows for rule_based" in check.message


def test_oracle_check_exports_zero_width_ci_for_constant_seed_deltas():
    aggregated = pd.DataFrame(
        [
            aggregate_row(SPEC.oracle_agent, 12.0),
            aggregate_row(SPEC.adaptive_agent, 10.0),
        ]
    )
    seed_rows = pd.DataFrame(
        [
            seed_row(SPEC.oracle_agent, seed, profit)
            for seed, profit in enumerate((7.0, 12.0, 17.0), start=1)
        ]
        + [
            seed_row(SPEC.adaptive_agent, seed, profit)
            for seed, profit in enumerate((5.0, 10.0, 15.0), start=1)
        ]
    )

    check = validate_oracle_not_worse_than_adaptive(
        aggregated,
        ValidationThresholds(),
        opponents=("calling",),
        algorithm_specs=(SPEC,),
        seed_rows=seed_rows,
    )[0]

    assert check.status == STATUS_PASS
    assert check.observed_value == 2.0
    assert check.standard_error == 0.0
    assert check.ci_lower == 2.0
    assert check.ci_upper == 2.0


def test_generalization_comparison_counts_only_ci_supported_wins():
    opponent_name = "calling_extreme"
    aggregated = pd.DataFrame(
        [
            aggregate_row(
                SPEC.adaptive_agent,
                4.0,
                opponent_name=opponent_name,
            ),
            aggregate_row(
                SPEC.general_policy_agent,
                2.0,
                opponent_name=opponent_name,
            ),
        ]
    )
    seed_rows = pd.DataFrame(
        [
            seed_row(
                SPEC.adaptive_agent,
                seed,
                value,
                opponent_name=opponent_name,
            )
            for seed, value in enumerate((3.0, 4.0, 5.0), start=1)
        ]
        + [
            seed_row(
                SPEC.general_policy_agent,
                seed,
                value,
                opponent_name=opponent_name,
            )
            for seed, value in enumerate((1.0, 2.0, 3.0), start=1)
        ]
    )

    check = validate_generalization_adaptive_beats_agent(
        aggregated,
        ValidationThresholds(),
        min_successful_variants=1,
        check_name="Adaptive beats fixed general",
        category="generalization_adaptive_delta_vs_general",
        opponents=(opponent_name,),
        algorithm_specs=(SPEC,),
        seed_rows=seed_rows,
    )[0]

    assert check.status == STATUS_PASS
    assert check.observed_value == 1.0
    assert check.details["successful_variants"] == [opponent_name]
    paired = check.details["paired_seed_statistics_by_variant"][opponent_name]
    assert paired["mean_delta"] == 2.0
    assert paired["ci_lower"] == 2.0
    assert paired["ci_upper"] == 2.0


def test_validation_markdown_includes_paired_statistical_columns():
    check = ValidationCheckResult(
        check_name="paired",
        status=STATUS_PASS,
        message="paired result",
        category="paired",
        observed_value=2.0,
        threshold=0.0,
        sample_size=3,
        standard_error=0.5,
        ci_lower=0.75,
        ci_upper=3.25,
    )
    report = ValidationReport(
        input_path="results.csv",
        thresholds=ValidationThresholds(),
        checks=[check],
    )

    markdown = render_validation_markdown(report)

    assert "sample_size" in markdown
    assert "standard_error" in markdown
    assert "ci_lower" in markdown
    assert "ci_upper" in markdown


def test_checkpoint_validation_pipeline_uses_selected_seed_level_rows(
    tmp_path,
):
    csv_path = tmp_path / "checkpoint_results.csv"
    write_sample_checkpoint_csv(csv_path)

    report = validate_checkpoint_results(csv_path)

    paired_checks = [
        check
        for check in report.checks
        if check.category in {
            "baseline_delta",
            "oracle_gap",
            "always_raise_sanity",
        }
        and check.status != STATUS_SKIPPED
        and check.details is not None
        and "paired_seed_statistics" in check.details
    ]
    assert paired_checks
    assert all(check.sample_size == 2 for check in paired_checks)
    assert all(check.standard_error is not None for check in paired_checks)
    assert all(check.ci_lower is not None for check in paired_checks)
    assert all(check.ci_upper is not None for check in paired_checks)

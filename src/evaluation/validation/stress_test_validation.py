from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.evaluation.algorithm_metadata import (
    AlgorithmValidationSpec,
    available_algorithm_specs,
)
from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.validation.common import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    ValidationCheckResult,
    ValidationThresholds,
    _checkpoint_episode,
    _find_row,
    _find_rows_at_latest_common_checkpoint,
    _format_float,
    _minimum_delta_status,
    _missing_common_checkpoint_result,
    _missing_row_result,
    _paired_seed_message,
    _paired_seed_statistics_for_check,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)

STRESS_TEST_OPPONENTS = (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)

STRESS_TEST_PROFITABILITY_OPPONENTS = (
    ALWAYS_CALL_AGENT,
    RULE_BASED_AGENT,
)


def _algorithm_specs(
    best_rows: pd.DataFrame,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> tuple[AlgorithmValidationSpec, ...]:
    return tuple(algorithm_specs or available_algorithm_specs(best_rows))


def _stress_test_agents(
    spec: AlgorithmValidationSpec,
) -> tuple[tuple[str, str], ...]:
    return (
        ("Adaptive", spec.adaptive_agent),
        ("Fixed general policy", spec.general_policy_agent),
    )


def validate_stress_test_profitability(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        for role_label, agent_name in _stress_test_agents(spec):
            for opponent_name in STRESS_TEST_PROFITABILITY_OPPONENTS:
                verb = "exploits" if opponent_name == ALWAYS_CALL_AGENT else "beats"
                opponent_label = (
                    "AlwaysCallPlayer"
                    if opponent_name == ALWAYS_CALL_AGENT
                    else "RuleBasedPlayer"
                )
                check_name = (
                    f"{spec.algorithm_name}: {role_label} {verb} "
                    f"{opponent_label}"
                )
                row = _find_row(best_rows, agent_name, opponent_name)
                if row is None:
                    results.append(
                        _missing_row_result(
                            check_name,
                            "stress_test_profitability",
                            agent_name,
                            opponent_name,
                            spec.algorithm_name,
                        )
                    )
                    continue

                mean_profit_bb = float(row["mean_profit_bb"])
                threshold = thresholds.min_head_to_head_mean_profit_bb
                results.append(
                    ValidationCheckResult(
                        check_name=check_name,
                        status=(
                            STATUS_PASS
                            if mean_profit_bb >= threshold
                            else STATUS_FAIL
                        ),
                        category="stress_test_profitability",
                        algorithm_name=spec.algorithm_name,
                        agent_name=agent_name,
                        opponent_name=opponent_name,
                        checkpoint_episode=_checkpoint_episode(row),
                        observed_value=mean_profit_bb,
                        threshold=threshold,
                        message=(
                            f"{agent_name} vs {opponent_name}: "
                            "mean_profit_bb="
                            f"{_format_float(mean_profit_bb)}."
                        ),
                        details={
                            "algorithm": spec.algorithm_name,
                            "role": role_label,
                            "mean_profit_bb": mean_profit_bb,
                            "win_rate": float(row["win_rate"]),
                            "bust_rate": float(row["bust_rate"]),
                        },
                    )
                )

    return results


def validate_stress_test_always_raise_resilience(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        for role_label, agent_name in _stress_test_agents(spec):
            check_name = (
                f"{spec.algorithm_name}: {role_label} resilience "
                "vs AlwaysRaisePlayer"
            )
            row = _find_row(best_rows, agent_name, ALWAYS_RAISE_AGENT)
            if row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "stress_test_always_raise",
                        agent_name,
                        ALWAYS_RAISE_AGENT,
                        spec.algorithm_name,
                    )
                )
                continue

            mean_profit_bb = float(row["mean_profit_bb"])
            bust_rate = float(row["bust_rate"])
            catastrophic_loss = (
                mean_profit_bb <= thresholds.always_raise_stress_loss_bb
                and bust_rate >= thresholds.always_raise_stress_bust_rate
            )
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=(
                        STATUS_WARNING if catastrophic_loss else STATUS_PASS
                    ),
                    category="stress_test_always_raise",
                    algorithm_name=spec.algorithm_name,
                    agent_name=agent_name,
                    opponent_name=ALWAYS_RAISE_AGENT,
                    checkpoint_episode=_checkpoint_episode(row),
                    observed_value=mean_profit_bb,
                    threshold=thresholds.always_raise_stress_loss_bb,
                    message=(
                        "Always-raise stress result: mean_profit_bb="
                        f"{_format_float(mean_profit_bb)}, "
                        f"bust_rate={_format_float(bust_rate)}%."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "role": role_label,
                        "mean_profit_bb": mean_profit_bb,
                        "bust_rate": bust_rate,
                        "stress_loss_threshold_bb": (
                            thresholds.always_raise_stress_loss_bb
                        ),
                        "stress_bust_rate_threshold": (
                            thresholds.always_raise_stress_bust_rate
                        ),
                    },
                )
            )

    return results


def validate_stress_test_adaptive_gap(
    comparison_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(comparison_rows, algorithm_specs):
        for opponent_name in STRESS_TEST_OPPONENTS:
            check_name = (
                f"{spec.algorithm_name}: Adaptive not significantly worse "
                f"than fixed general vs {opponent_name}"
            )
            matchups = (
                (spec.adaptive_agent, opponent_name),
                (spec.general_policy_agent, opponent_name),
            )
            checkpoint_episode, rows = (
                _find_rows_at_latest_common_checkpoint(
                    comparison_rows,
                    matchups,
                )
            )
            adaptive_row, general_row = rows
            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "stress_test_adaptive_gap",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue
            if general_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "stress_test_adaptive_gap",
                        spec.general_policy_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue
            if checkpoint_episode is None:
                results.append(
                    _missing_common_checkpoint_result(
                        check_name,
                        "stress_test_adaptive_gap",
                        comparison_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            paired_statistics, unavailable_result = (
                _paired_seed_statistics_for_check(
                    seed_rows,
                    left_agent_name=spec.adaptive_agent,
                    right_agent_name=spec.general_policy_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=checkpoint_episode,
                    thresholds=thresholds,
                    check_name=check_name,
                    category="stress_test_adaptive_gap",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                )
            )
            if unavailable_result is not None:
                results.append(unavailable_result)
                continue

            adaptive_profit = float(adaptive_row["mean_profit_bb"])
            general_profit = float(general_row["mean_profit_bb"])
            threshold = -thresholds.max_adaptive_underperformance_vs_general_bb
            if paired_statistics is None:
                adaptive_gap = adaptive_profit - general_profit
                status = (
                    STATUS_PASS
                    if adaptive_gap >= threshold
                    else STATUS_WARNING
                )
                message = (
                    "Adaptive minus fixed general mean profit is "
                    f"{_format_float(adaptive_gap)} BB/game."
                )
            else:
                adaptive_gap = float(paired_statistics.mean_delta)
                status = _minimum_delta_status(
                    paired_statistics,
                    threshold,
                    underperformance_status=STATUS_WARNING,
                )
                message = _paired_seed_message(
                    "Adaptive minus fixed general mean profit",
                    paired_statistics,
                )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    category="stress_test_adaptive_gap",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=checkpoint_episode,
                    observed_value=adaptive_gap,
                    threshold=threshold,
                    sample_size=(
                        paired_statistics.common_seed_count
                        if paired_statistics is not None
                        else None
                    ),
                    standard_error=(
                        paired_statistics.standard_error
                        if paired_statistics is not None
                        else None
                    ),
                    ci_lower=(
                        paired_statistics.ci_lower
                        if paired_statistics is not None
                        else None
                    ),
                    ci_upper=(
                        paired_statistics.ci_upper
                        if paired_statistics is not None
                        else None
                    ),
                    message=message,
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "general_policy_agent": spec.general_policy_agent,
                        "adaptive_mean_profit_bb": adaptive_profit,
                        "general_mean_profit_bb": general_profit,
                        **(
                            {
                                "paired_seed_statistics": (
                                    paired_statistics.to_details()
                                )
                            }
                            if paired_statistics is not None
                            else {}
                        ),
                    },
                )
            )

    return results


def validate_stress_test_classifier_coverage(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        for opponent_name in STRESS_TEST_OPPONENTS:
            check_name = (
                f"{spec.algorithm_name}: Stress-test classifier coverage "
                f"vs {opponent_name}"
            )
            row = _find_row(best_rows, spec.adaptive_agent, opponent_name)
            if row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "stress_test_classifier_coverage",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            coverage = float(row["global_classifier_coverage"])
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=(
                        STATUS_PASS
                        if coverage >= thresholds.min_classifier_coverage
                        else STATUS_WARNING
                    ),
                    category="stress_test_classifier_coverage",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=_checkpoint_episode(row),
                    observed_value=coverage,
                    threshold=thresholds.min_classifier_coverage,
                    message=(
                        "Adaptive classifier coverage on the OOD stress "
                        f"opponent is {_format_float(coverage)}%."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "coverage": coverage,
                        "accuracy_ignored_for_ood": True,
                    },
                )
            )

    return results


def validate_stress_test_results_from_best_rows(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    comparison_rows: pd.DataFrame | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    specs = tuple(algorithm_specs or available_algorithm_specs(best_rows))
    aligned_comparison_rows = (
        best_rows if comparison_rows is None else comparison_rows
    )
    checks: list[ValidationCheckResult] = []
    checks.extend(
        validate_stress_test_profitability(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_stress_test_always_raise_resilience(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_stress_test_adaptive_gap(
            aligned_comparison_rows,
            thresholds,
            algorithm_specs=specs,
            seed_rows=seed_rows,
        )
    )
    checks.extend(
        validate_stress_test_classifier_coverage(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(validate_minimum_seed_coverage(best_rows, thresholds))
    checks.extend(validate_seed_stability(best_rows, thresholds))
    checks.extend(validate_extreme_bb_per_100(best_rows, thresholds))
    return checks

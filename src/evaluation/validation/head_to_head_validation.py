from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ALGORITHM_MONTE_CARLO,
    AlgorithmValidationSpec,
    available_algorithm_specs,
)
from src.evaluation.validation.common import (
    HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
    HEAD_TO_HEAD_SPECIALIST_AGENTS,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationCheckResult,
    ValidationThresholds,
    _checkpoint_episode,
    _find_row,
    _find_rows_at_latest_common_checkpoint,
    _format_float,
    _missing_common_checkpoint_result,
    _missing_row_result,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)


def _algorithm_specs(
    best_rows: pd.DataFrame,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> tuple[AlgorithmValidationSpec, ...]:
    return tuple(algorithm_specs or available_algorithm_specs(best_rows))


def _profit_check_result(
    *,
    best_rows: pd.DataFrame,
    agent_name: str,
    opponent_name: str,
    check_name: str,
    category: str,
    thresholds: ValidationThresholds,
    algorithm_name: str | None = None,
    fail_on_underperformance: bool = True,
) -> ValidationCheckResult:
    row = _find_row(best_rows, agent_name, opponent_name)

    if row is None:
        return _missing_row_result(
            check_name,
            category,
            agent_name,
            opponent_name,
            algorithm_name,
        )

    mean_profit_bb = float(row["mean_profit_bb"])
    passed = mean_profit_bb >= thresholds.min_head_to_head_mean_profit_bb

    if passed:
        status = STATUS_PASS
    else:
        status = STATUS_FAIL if fail_on_underperformance else STATUS_WARNING

    return ValidationCheckResult(
        check_name=check_name,
        status=status,
        category=category,
        algorithm_name=algorithm_name,
        agent_name=agent_name,
        opponent_name=opponent_name,
        checkpoint_episode=_checkpoint_episode(row),
        observed_value=mean_profit_bb,
        threshold=thresholds.min_head_to_head_mean_profit_bb,
        message=(
            f"{agent_name} vs {opponent_name}: "
            f"mean_profit_bb={_format_float(mean_profit_bb)}, "
            f"win_rate={_format_float(float(row['win_rate']))}%."
        ),
        details={
            "algorithm": algorithm_name,
            "mean_profit_bb": mean_profit_bb,
            "win_rate": float(row["win_rate"]),
            "bust_rate": float(row["bust_rate"]),
        },
    )


def validate_head_to_head_rule_based_performance(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        results.extend(
            [
                _profit_check_result(
                    best_rows=best_rows,
                    agent_name=spec.general_policy_agent,
                    opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                    check_name=(
                        f"{spec.algorithm_name}: Fixed general policy "
                        "beats RuleBasedPlayer"
                    ),
                    category="head_to_head_rule_based",
                    thresholds=thresholds,
                    algorithm_name=spec.algorithm_name,
                ),
                _profit_check_result(
                    best_rows=best_rows,
                    agent_name=spec.adaptive_agent,
                    opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                    check_name=(
                        f"{spec.algorithm_name}: Adaptive beats "
                        "RuleBasedPlayer"
                    ),
                    category="head_to_head_rule_based",
                    thresholds=thresholds,
                    algorithm_name=spec.algorithm_name,
                ),
            ]
        )

    return results


def validate_head_to_head_specialist_rule_based_performance(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    check_name = "Monte Carlo: At least one specialist beats RuleBasedPlayer"
    available_rows: list[pd.Series] = []
    missing_agents: list[str] = []

    for agent_name in HEAD_TO_HEAD_SPECIALIST_AGENTS:
        row = _find_row(
            best_rows,
            agent_name,
            HEAD_TO_HEAD_RULE_BASED_OPPONENT,
        )
        if row is None:
            missing_agents.append(agent_name)
        else:
            available_rows.append(row)

    if not available_rows:
        return [
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_SKIPPED,
                category="head_to_head_rule_based",
                algorithm_name=ALGORITHM_MONTE_CARLO,
                opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                message="Missing all specialist rows vs RuleBasedPlayer.",
                details={"missing_agents": missing_agents},
            )
        ]

    best_row = max(
        available_rows,
        key=lambda row: float(row["mean_profit_bb"]),
    )
    best_profit = float(best_row["mean_profit_bb"])
    passed = best_profit >= thresholds.min_head_to_head_mean_profit_bb

    details = {
        "algorithm": ALGORITHM_MONTE_CARLO,
        "best_specialist_agent": str(best_row["agent_name"]),
        "best_specialist_mean_profit_bb": best_profit,
        "missing_agents": missing_agents,
        "note": (
            "Only Monte Carlo fixed specialists are public tested agents. "
            "TD specialists are used through adaptive/oracle agents."
        ),
    }

    for row in available_rows:
        details[f"{row['agent_name']}_mean_profit_bb"] = float(
            row["mean_profit_bb"]
        )

    return [
        ValidationCheckResult(
            check_name=check_name,
            status=STATUS_PASS if passed else STATUS_FAIL,
            category="head_to_head_rule_based",
            algorithm_name=ALGORITHM_MONTE_CARLO,
            agent_name=str(best_row["agent_name"]),
            opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
            checkpoint_episode=_checkpoint_episode(best_row),
            observed_value=best_profit,
            threshold=thresholds.min_head_to_head_mean_profit_bb,
            message=(
                "Best specialist vs RuleBasedPlayer is "
                f"{best_row['agent_name']} with "
                f"mean_profit_bb={_format_float(best_profit)}."
            ),
            details=details,
        )
    ]


def validate_adaptive_not_worse_than_general_rule_based(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        check_name = (
            f"{spec.algorithm_name}: Adaptive not significantly worse "
            "than fixed general vs RuleBasedPlayer"
        )
        matchups = (
            (spec.adaptive_agent, HEAD_TO_HEAD_RULE_BASED_OPPONENT),
            (spec.general_policy_agent, HEAD_TO_HEAD_RULE_BASED_OPPONENT),
        )
        checkpoint_episode, rows = _find_rows_at_latest_common_checkpoint(
            best_rows,
            matchups,
        )
        adaptive_row, general_row = rows

        if adaptive_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "head_to_head_adaptive_gap",
                    spec.adaptive_agent,
                    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                    spec.algorithm_name,
                )
            )
            continue

        if general_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "head_to_head_adaptive_gap",
                    spec.general_policy_agent,
                    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                    spec.algorithm_name,
                )
            )
            continue

        if checkpoint_episode is None:
            results.append(
                _missing_common_checkpoint_result(
                    check_name,
                    "head_to_head_adaptive_gap",
                    best_rows,
                    matchups,
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                )
            )
            continue

        adaptive_gap = float(
            adaptive_row["mean_profit_bb"] - general_row["mean_profit_bb"]
        )
        threshold = -thresholds.max_adaptive_underperformance_vs_general_bb
        status = STATUS_PASS if adaptive_gap >= threshold else STATUS_WARNING

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=status,
                category="head_to_head_adaptive_gap",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
                opponent_name=HEAD_TO_HEAD_RULE_BASED_OPPONENT,
                checkpoint_episode=checkpoint_episode,
                observed_value=adaptive_gap,
                threshold=threshold,
                message=(
                    "Adaptive minus fixed general mean profit vs "
                    "RuleBasedPlayer is "
                    f"{_format_float(adaptive_gap)} BB/game."
                ),
                details={
                    "algorithm": spec.algorithm_name,
                    "adaptive_agent": spec.adaptive_agent,
                    "general_policy_agent": spec.general_policy_agent,
                    "adaptive_mean_profit_bb": float(adaptive_row["mean_profit_bb"]),
                    "general_mean_profit_bb": float(general_row["mean_profit_bb"]),
                    "general_checkpoint_episode": checkpoint_episode,
                    "max_underperformance_bb": (
                        thresholds.max_adaptive_underperformance_vs_general_bb
                    ),
                },
            )
        )

    return results


def validate_head_to_head_ood_classifier_coverage(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = (
        HEAD_TO_HEAD_RULE_BASED_OPPONENT,
        HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    ),
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        for opponent_name in opponents:
            adaptive_row = _find_row(
                best_rows,
                spec.adaptive_agent,
                opponent_name,
            )
            check_name = (
                f"{spec.algorithm_name}: OOD classifier coverage "
                f"vs {opponent_name}"
            )

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "head_to_head_ood_classifier",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            coverage = float(adaptive_row["global_classifier_coverage"])
            status = (
                STATUS_PASS
                if coverage >= thresholds.min_classifier_coverage
                else STATUS_WARNING
            )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    category="head_to_head_ood_classifier",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=_checkpoint_episode(adaptive_row),
                    observed_value=coverage,
                    threshold=thresholds.min_classifier_coverage,
                    message=(
                        "Adaptive OOD classifier coverage is "
                        f"{_format_float(coverage)}%. Accuracy is "
                        "intentionally not validated for OOD opponents."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "coverage": coverage,
                        "accuracy_ignored_for_ood": True,
                        "global_classifier_accuracy": float(
                            adaptive_row["global_classifier_accuracy"]
                        ),
                    },
                )
            )

    return results


def validate_always_raise_head_to_head_stress_test(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    agents: Iterable[str] | None = None,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    specs = _algorithm_specs(best_rows, algorithm_specs)
    agent_to_algorithm = {
        spec.adaptive_agent: spec.algorithm_name
        for spec in specs
    }
    agent_to_algorithm.update(
        {
            spec.general_policy_agent: spec.algorithm_name
            for spec in specs
        }
    )

    if agents is None:
        evaluated_agents = list(agent_to_algorithm)
        available_agents = set(best_rows["agent_name"].dropna())
        evaluated_agents.extend(
            agent_name
            for agent_name in HEAD_TO_HEAD_SPECIALIST_AGENTS
            if agent_name in available_agents
        )
    else:
        evaluated_agents = list(agents)

    results: list[ValidationCheckResult] = []

    for agent_name in evaluated_agents:
        row = _find_row(
            best_rows,
            agent_name,
            HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
        )
        algorithm_name = agent_to_algorithm.get(
            agent_name,
            ALGORITHM_MONTE_CARLO
            if agent_name in HEAD_TO_HEAD_SPECIALIST_AGENTS
            else None,
        )
        check_name = (
            f"{algorithm_name + ': ' if algorithm_name else ''}"
            f"AlwaysRaise stress test vs {agent_name}"
        )

        if row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "head_to_head_stress_test",
                    agent_name,
                    HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
                    algorithm_name,
                )
            )
            continue

        mean_profit_bb = float(row["mean_profit_bb"])
        bust_rate = float(row["bust_rate"])
        stress_failure = (
            mean_profit_bb <= thresholds.always_raise_stress_loss_bb
            and bust_rate >= thresholds.always_raise_stress_bust_rate
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_WARNING if stress_failure else STATUS_PASS,
                category="head_to_head_stress_test",
                algorithm_name=algorithm_name,
                agent_name=agent_name,
                opponent_name=HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
                checkpoint_episode=_checkpoint_episode(row),
                observed_value=mean_profit_bb,
                threshold=thresholds.always_raise_stress_loss_bb,
                message=(
                    f"{agent_name} vs AlwaysRaisePlayer: "
                    f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                    f"bust_rate={_format_float(bust_rate)}%."
                ),
                details={
                    "algorithm": algorithm_name,
                    "mean_profit_bb": mean_profit_bb,
                    "bust_rate": bust_rate,
                    "stress_loss_threshold_bb": thresholds.always_raise_stress_loss_bb,
                    "stress_bust_rate_threshold": thresholds.always_raise_stress_bust_rate,
                },
            )
        )

    return results

def validate_head_to_head_results_from_best_rows(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    comparison_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    specs = tuple(algorithm_specs or available_algorithm_specs(best_rows))
    aligned_comparison_rows = (
        best_rows if comparison_rows is None else comparison_rows
    )

    checks: list[ValidationCheckResult] = []
    checks.extend(
        validate_head_to_head_rule_based_performance(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_head_to_head_specialist_rule_based_performance(
            best_rows,
            thresholds,
        )
    )
    checks.extend(
        validate_adaptive_not_worse_than_general_rule_based(
            aligned_comparison_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_head_to_head_ood_classifier_coverage(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_always_raise_head_to_head_stress_test(
            best_rows,
            thresholds,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_minimum_seed_coverage(
            best_rows,
            thresholds,
        )
    )
    checks.extend(
        validate_seed_stability(
            best_rows,
            thresholds,
        )
    )
    checks.extend(
        validate_extreme_bb_per_100(
            best_rows,
            thresholds,
        )
    )
    return checks

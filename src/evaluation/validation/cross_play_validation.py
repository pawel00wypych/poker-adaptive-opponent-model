from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import pandas as pd

from src.evaluation.algorithm_metadata import AlgorithmValidationSpec
from src.evaluation.metrics.paired_seed_statistics import (
    PAIRED_SEED_OPERATION_SUM,
)
from src.evaluation.validation.common import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    ValidationCheckResult,
    ValidationThresholds,
    _find_row,
    _find_rows_at_common_training_episode,
    _format_float,
    _missing_common_training_episode_result,
    _missing_row_result,
    _paired_seed_statistics_for_check,
    _training_episode,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)


def _required_adaptive_matchups(
    algorithm_specs: Iterable[AlgorithmValidationSpec],
) -> tuple[tuple[str, str], ...]:
    adaptive_agents = tuple(spec.adaptive_agent for spec in algorithm_specs)
    return tuple(
        (agent_name, opponent_name)
        for agent_name in adaptive_agents
        for opponent_name in adaptive_agents
        if agent_name != opponent_name
    )


def validate_cross_play_matchup_coverage(
    best_rows: pd.DataFrame,
    algorithm_specs: Iterable[AlgorithmValidationSpec],
    *,
    fail_when_missing: bool = True,
) -> list[ValidationCheckResult]:
    specs = tuple(algorithm_specs)
    available_matchups = (
        set(
            best_rows[["agent_name", "opponent_name"]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        if {"agent_name", "opponent_name"}.issubset(best_rows.columns)
        else set()
    )
    required_matchups = _required_adaptive_matchups(specs)
    missing_matchups = [
        {
            "agent_name": agent_name,
            "opponent_name": opponent_name,
        }
        for agent_name, opponent_name in required_matchups
        if (agent_name, opponent_name) not in available_matchups
    ]
    complete = not missing_matchups
    if complete:
        status = STATUS_PASS
    elif fail_when_missing:
        status = STATUS_FAIL
    else:
        status = STATUS_WARNING

    return [
        ValidationCheckResult(
            check_name=("Learned-agent cross-play: Required adaptive matchup coverage"),
            status=status,
            category="cross_play_matchup_coverage",
            message=(
                "All required directed adaptive-vs-adaptive matchups are present."
                if complete
                else (
                    f"Missing {len(missing_matchups)} of "
                    f"{len(required_matchups)} required directed "
                    "adaptive-vs-adaptive matchups."
                )
            ),
            details={
                "required_algorithms": [spec.algorithm_key for spec in specs],
                "required_agents": [spec.adaptive_agent for spec in specs],
                "required_matchup_count": len(required_matchups),
                "present_matchup_count": (
                    len(required_matchups) - len(missing_matchups)
                ),
                "missing_matchup_count": len(missing_matchups),
                "required_matchups": [
                    {
                        "agent_name": agent_name,
                        "opponent_name": opponent_name,
                    }
                    for agent_name, opponent_name in required_matchups
                ],
                "missing_matchups": missing_matchups,
                "present": complete,
            },
        )
    ]


def _cross_play_reciprocity_pairs(
    algorithm_specs: Iterable[AlgorithmValidationSpec],
) -> tuple[
    tuple[
        str,
        str,
        str,
        str,
        str | None,
        bool,
    ],
    ...,
]:
    specs = tuple(algorithm_specs)
    pairs: list[tuple[str, str, str, str, str | None, bool]] = []

    for first_spec, second_spec in combinations(specs, 2):
        pairs.append(
            (
                first_spec.adaptive_agent,
                second_spec.adaptive_agent,
                first_spec.algorithm_name,
                second_spec.algorithm_name,
                None,
                True,
            )
        )

    for first_spec, second_spec in combinations(specs, 2):
        pairs.append(
            (
                first_spec.general_policy_agent,
                second_spec.general_policy_agent,
                f"{first_spec.algorithm_name} fixed general",
                f"{second_spec.algorithm_name} fixed general",
                None,
                False,
            )
        )

    for spec in specs:
        pairs.append(
            (
                spec.adaptive_agent,
                spec.general_policy_agent,
                f"{spec.algorithm_name} adaptive",
                f"{spec.algorithm_name} fixed general",
                spec.algorithm_name,
                False,
            )
        )

    return tuple(pairs)


def validate_cross_play_pair_reciprocity(
    comparison_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec],
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []
    available_matchups = set(
        comparison_rows[["agent_name", "opponent_name"]]
        .dropna()
        .itertuples(index=False, name=None)
    )

    for (
        first_agent,
        second_agent,
        first_label,
        second_label,
        algorithm_name,
        required,
    ) in _cross_play_reciprocity_pairs(algorithm_specs):
        matchups = (
            (first_agent, second_agent),
            (second_agent, first_agent),
        )
        if not required and not any(
            matchup in available_matchups for matchup in matchups
        ):
            continue

        check_name = f"Cross-play reciprocity for {first_label} and {second_label}"
        training_episode, rows = _find_rows_at_common_training_episode(
            comparison_rows,
            matchups,
        )
        first_row, second_row = rows

        if first_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "cross_play_pair_reciprocity",
                    first_agent,
                    second_agent,
                    algorithm_name,
                )
            )
            continue
        if second_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "cross_play_pair_reciprocity",
                    second_agent,
                    first_agent,
                    algorithm_name,
                )
            )
            continue
        if training_episode is None:
            results.append(
                _missing_common_training_episode_result(
                    check_name,
                    "cross_play_pair_reciprocity",
                    comparison_rows,
                    matchups,
                    algorithm_name=algorithm_name,
                    agent_name=first_agent,
                    opponent_name=second_agent,
                )
            )
            continue

        threshold = thresholds.max_cross_play_pair_sum_abs_profit_bb
        paired_statistics, unavailable_result = _paired_seed_statistics_for_check(
            seed_rows,
            left_agent_name=first_agent,
            right_agent_name=second_agent,
            opponent_name=second_agent,
            right_opponent_name=first_agent,
            operation=PAIRED_SEED_OPERATION_SUM,
            training_episode=training_episode,
            thresholds=thresholds,
            check_name=check_name,
            category="cross_play_pair_reciprocity",
            algorithm_name=algorithm_name,
            agent_name=first_agent,
        )
        if unavailable_result is not None:
            results.append(unavailable_result)
            continue

        first_profit = float(first_row["mean_profit_bb"])
        second_profit = float(second_row["mean_profit_bb"])
        if paired_statistics is None:
            pair_sum = first_profit + second_profit
            status = STATUS_PASS if abs(pair_sum) <= threshold else STATUS_WARNING
            message = (
                "Opposite-direction mean profits sum to "
                f"{_format_float(pair_sum)} BB/game."
            )
        else:
            pair_sum = float(paired_statistics.mean_value)
            assert paired_statistics.ci_lower is not None
            assert paired_statistics.ci_upper is not None
            interval_within_bounds = (
                paired_statistics.ci_lower >= -threshold
                and paired_statistics.ci_upper <= threshold
            )
            status = STATUS_PASS if interval_within_bounds else STATUS_WARNING
            message = (
                "Opposite-direction per-seed profit sums have mean "
                f"{_format_float(pair_sum)} BB/game across "
                f"{paired_statistics.common_seed_count} paired seed(s); "
                "95% t-CI "
                f"[{_format_float(paired_statistics.ci_lower)}, "
                f"{_format_float(paired_statistics.ci_upper)}]."
            )

        absolute_pair_sum = abs(pair_sum)
        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=status,
                category="cross_play_pair_reciprocity",
                algorithm_name=algorithm_name,
                agent_name=first_agent,
                opponent_name=second_agent,
                training_episode=training_episode,
                observed_value=absolute_pair_sum,
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
                    "first_agent": first_agent,
                    "second_agent": second_agent,
                    "first_direction_mean_profit_bb": first_profit,
                    "second_direction_mean_profit_bb": second_profit,
                    "pair_sum_bb": pair_sum,
                    "absolute_pair_sum_bb": absolute_pair_sum,
                    "max_absolute_pair_sum_bb": threshold,
                    "required_matchup": required,
                    **(
                        {"paired_seed_statistics": (paired_statistics.to_details())}
                        if paired_statistics is not None
                        else {}
                    ),
                },
            )
        )

    return results


def validate_cross_play_classifier_coverage(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec],
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []
    specs = tuple(algorithm_specs)

    for spec in specs:
        for opponent_spec in specs:
            if spec == opponent_spec:
                continue

            check_name = (
                f"{spec.algorithm_name}: Cross-play classifier coverage "
                f"vs {opponent_spec.algorithm_name} adaptive"
            )
            row = _find_row(
                best_rows,
                spec.adaptive_agent,
                opponent_spec.adaptive_agent,
            )
            if row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "cross_play_classifier_coverage",
                        spec.adaptive_agent,
                        opponent_spec.adaptive_agent,
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
                    category="cross_play_classifier_coverage",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_spec.adaptive_agent,
                    training_episode=_training_episode(row),
                    observed_value=coverage,
                    threshold=thresholds.min_classifier_coverage,
                    message=(
                        "Adaptive classifier coverage against the learned "
                        f"policy is {_format_float(coverage)}%."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "opponent_algorithm": opponent_spec.algorithm_name,
                        "coverage": coverage,
                        "accuracy_ignored_for_ood": True,
                    },
                )
            )

    return results


def validate_cross_play_results_from_best_rows(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec],
    comparison_rows: pd.DataFrame | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    specs = tuple(algorithm_specs)
    aligned_comparison_rows = best_rows if comparison_rows is None else comparison_rows
    checks: list[ValidationCheckResult] = []
    checks.extend(validate_cross_play_matchup_coverage(best_rows, specs))
    checks.extend(
        validate_cross_play_pair_reciprocity(
            aligned_comparison_rows,
            thresholds,
            specs,
            seed_rows=seed_rows,
        )
    )
    checks.extend(
        validate_cross_play_classifier_coverage(
            best_rows,
            thresholds,
            specs,
        )
    )
    checks.extend(validate_minimum_seed_coverage(best_rows, thresholds))
    checks.extend(validate_seed_stability(best_rows, thresholds))
    checks.extend(validate_extreme_bb_per_100(best_rows, thresholds))
    return checks

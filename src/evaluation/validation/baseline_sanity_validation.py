from __future__ import annotations

from itertools import combinations

import pandas as pd

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
    _missing_common_checkpoint_result,
    _missing_row_result,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)

BASELINE_SANITY_AGENTS = (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)


def validate_baseline_matchup_coverage(
    best_rows: pd.DataFrame,
) -> list[ValidationCheckResult]:
    available_matchups = (
        set(
            best_rows[["agent_name", "opponent_name"]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        if {"agent_name", "opponent_name"}.issubset(best_rows.columns)
        else set()
    )
    required_matchups = [
        {
            "agent_name": agent_name,
            "opponent_name": opponent_name,
        }
        for agent_name in BASELINE_SANITY_AGENTS
        for opponent_name in BASELINE_SANITY_AGENTS
    ]
    missing_matchups = [
        matchup
        for matchup in required_matchups
        if (matchup["agent_name"], matchup["opponent_name"])
        not in available_matchups
    ]
    complete = not missing_matchups

    return [
        ValidationCheckResult(
            check_name="Baseline sanity: Required matchup coverage",
            status=STATUS_PASS if complete else STATUS_FAIL,
            category="baseline_matchup_coverage",
            message=(
                "All 9 baseline-vs-baseline matchups are present."
                if complete
                else (
                    f"Missing {len(missing_matchups)} of 9 required "
                    "baseline-vs-baseline matchups."
                )
            ),
            details={
                "required_matchup_count": len(required_matchups),
                "present_matchup_count": (
                    len(required_matchups) - len(missing_matchups)
                ),
                "missing_matchup_count": len(missing_matchups),
                "required_matchups": required_matchups,
                "missing_matchups": missing_matchups,
                "present": complete,
            },
        )
    ]


def validate_baseline_mirror_neutrality(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for agent_name in BASELINE_SANITY_AGENTS:
        check_name = f"Baseline mirror neutrality for {agent_name}"
        row = _find_row(best_rows, agent_name, agent_name)
        if row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_mirror_neutrality",
                    agent_name,
                    agent_name,
                )
            )
            continue

        mean_profit_bb = float(row["mean_profit_bb"])
        absolute_profit = abs(mean_profit_bb)
        threshold = thresholds.max_baseline_mirror_abs_profit_bb
        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=(
                    STATUS_PASS
                    if absolute_profit <= threshold
                    else STATUS_WARNING
                ),
                category="baseline_mirror_neutrality",
                agent_name=agent_name,
                opponent_name=agent_name,
                checkpoint_episode=_checkpoint_episode(row),
                observed_value=absolute_profit,
                threshold=threshold,
                message=(
                    f"Mirror mean profit is {_format_float(mean_profit_bb)} "
                    "BB/game; absolute deviation from zero is "
                    f"{_format_float(absolute_profit)} BB/game."
                ),
                details={
                    "mean_profit_bb": mean_profit_bb,
                    "absolute_mean_profit_bb": absolute_profit,
                    "max_absolute_profit_bb": threshold,
                },
            )
        )

    return results


def validate_baseline_pair_reciprocity(
    comparison_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for first_agent, second_agent in combinations(BASELINE_SANITY_AGENTS, 2):
        check_name = (
            f"Baseline pair reciprocity for {first_agent} and {second_agent}"
        )
        matchups = (
            (first_agent, second_agent),
            (second_agent, first_agent),
        )
        checkpoint_episode, rows = _find_rows_at_latest_common_checkpoint(
            comparison_rows,
            matchups,
        )
        first_row, second_row = rows

        if first_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_pair_reciprocity",
                    first_agent,
                    second_agent,
                )
            )
            continue
        if second_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_pair_reciprocity",
                    second_agent,
                    first_agent,
                )
            )
            continue
        if checkpoint_episode is None:
            results.append(
                _missing_common_checkpoint_result(
                    check_name,
                    "baseline_pair_reciprocity",
                    comparison_rows,
                    matchups,
                    agent_name=first_agent,
                    opponent_name=second_agent,
                )
            )
            continue

        first_profit = float(first_row["mean_profit_bb"])
        second_profit = float(second_row["mean_profit_bb"])
        pair_sum = first_profit + second_profit
        absolute_pair_sum = abs(pair_sum)
        threshold = thresholds.max_baseline_pair_sum_abs_profit_bb
        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=(
                    STATUS_PASS
                    if absolute_pair_sum <= threshold
                    else STATUS_WARNING
                ),
                category="baseline_pair_reciprocity",
                agent_name=first_agent,
                opponent_name=second_agent,
                checkpoint_episode=checkpoint_episode,
                observed_value=absolute_pair_sum,
                threshold=threshold,
                message=(
                    "Opposite-direction mean profits sum to "
                    f"{_format_float(pair_sum)} BB/game."
                ),
                details={
                    "first_agent": first_agent,
                    "second_agent": second_agent,
                    "first_direction_mean_profit_bb": first_profit,
                    "second_direction_mean_profit_bb": second_profit,
                    "pair_sum_bb": pair_sum,
                    "absolute_pair_sum_bb": absolute_pair_sum,
                    "max_absolute_pair_sum_bb": threshold,
                },
            )
        )

    return results


def validate_baseline_extreme_results(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for tested_agent in (ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT):
        for opponent_name in BASELINE_SANITY_AGENTS:
            if tested_agent == opponent_name:
                continue

            check_name = (
                f"Baseline extreme result for {tested_agent} "
                f"vs {opponent_name}"
            )
            row = _find_row(best_rows, tested_agent, opponent_name)
            if row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "baseline_extreme_result",
                        tested_agent,
                        opponent_name,
                    )
                )
                continue

            mean_profit_bb = float(row["mean_profit_bb"])
            win_rate = float(row["win_rate"])
            suspicious = (
                mean_profit_bb
                >= thresholds.high_always_raise_mean_profit_bb
                and win_rate >= thresholds.high_always_raise_win_rate
            )
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_WARNING if suspicious else STATUS_PASS,
                    category="baseline_extreme_result",
                    agent_name=tested_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=_checkpoint_episode(row),
                    observed_value=mean_profit_bb,
                    threshold=thresholds.high_always_raise_mean_profit_bb,
                    message=(
                        f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                        f"win_rate={_format_float(win_rate)}%."
                    ),
                    details={
                        "mean_profit_bb": mean_profit_bb,
                        "win_rate": win_rate,
                        "high_mean_profit_bb_threshold": (
                            thresholds.high_always_raise_mean_profit_bb
                        ),
                        "high_win_rate_threshold": (
                            thresholds.high_always_raise_win_rate
                        ),
                    },
                )
            )

    return results


def validate_baseline_sanity_results_from_best_rows(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    comparison_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    aligned_comparison_rows = (
        best_rows if comparison_rows is None else comparison_rows
    )
    checks: list[ValidationCheckResult] = []
    checks.extend(validate_baseline_matchup_coverage(best_rows))
    checks.extend(validate_baseline_mirror_neutrality(best_rows, thresholds))
    checks.extend(
        validate_baseline_pair_reciprocity(
            aligned_comparison_rows,
            thresholds,
        )
    )
    checks.extend(validate_baseline_extreme_results(best_rows, thresholds))
    checks.extend(validate_minimum_seed_coverage(best_rows, thresholds))
    checks.extend(validate_seed_stability(best_rows, thresholds))
    checks.extend(validate_extreme_bb_per_100(best_rows, thresholds))
    return checks

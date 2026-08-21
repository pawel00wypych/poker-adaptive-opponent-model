from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.metrics.baseline_metrics import (
    BASELINE_REPLICATE_STD_COLUMN,
)
from src.evaluation.validation.common import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationCheckResult,
    ValidationThresholds,
    _checkpoint_episode,
    _find_row,
    _format_float,
    _missing_row_result,
    validate_extreme_bb_per_100,
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
    replicate_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for first_agent, second_agent in combinations(BASELINE_SANITY_AGENTS, 2):
        check_name = (
            f"Baseline pair reciprocity for {first_agent} and {second_agent}"
        )
        first_direction = replicate_rows[
            (replicate_rows["agent_name"] == first_agent)
            & (replicate_rows["opponent_name"] == second_agent)
        ][["evaluation_replicate_id", "mean_profit_bb"]].rename(
            columns={"mean_profit_bb": "first_direction_profit_bb"}
        )
        second_direction = replicate_rows[
            (replicate_rows["agent_name"] == second_agent)
            & (replicate_rows["opponent_name"] == first_agent)
        ][["evaluation_replicate_id", "mean_profit_bb"]].rename(
            columns={"mean_profit_bb": "second_direction_profit_bb"}
        )

        if first_direction.empty:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_pair_reciprocity",
                    first_agent,
                    second_agent,
                )
            )
            continue
        if second_direction.empty:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_pair_reciprocity",
                    second_agent,
                    first_agent,
                )
            )
            continue
        paired = first_direction.merge(
            second_direction,
            on="evaluation_replicate_id",
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_FAIL,
                    category="baseline_pair_reciprocity",
                    agent_name=first_agent,
                    opponent_name=second_agent,
                    message=(
                        "The opposite matchup directions have no common "
                        "evaluation_replicate_id."
                    ),
                    details={
                        "first_direction_replicate_ids": sorted(
                            first_direction[
                                "evaluation_replicate_id"
                            ].astype(int).tolist()
                        ),
                        "second_direction_replicate_ids": sorted(
                            second_direction[
                                "evaluation_replicate_id"
                            ].astype(int).tolist()
                        ),
                    },
                )
            )
            continue

        paired["pair_sum_bb"] = (
            paired["first_direction_profit_bb"]
            + paired["second_direction_profit_bb"]
        )
        pair_sum = float(paired["pair_sum_bb"].mean())
        absolute_pair_sum = abs(pair_sum)
        sample_size = len(paired)
        pair_std = float(paired["pair_sum_bb"].std())
        standard_error = (
            pair_std / np.sqrt(sample_size)
            if sample_size >= 2 and not np.isnan(pair_std)
            else None
        )
        ci_margin = (
            float(student_t.ppf(0.975, sample_size - 1) * standard_error)
            if standard_error is not None
            else None
        )
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
                observed_value=absolute_pair_sum,
                threshold=threshold,
                sample_size=sample_size,
                standard_error=standard_error,
                ci_lower=(pair_sum - ci_margin if ci_margin is not None else None),
                ci_upper=(pair_sum + ci_margin if ci_margin is not None else None),
                message=(
                    "Mean paired sum of opposite-direction profits across "
                    f"{sample_size} evaluation replicate(s) is "
                    f"{_format_float(pair_sum)} BB/game."
                ),
                details={
                    "first_agent": first_agent,
                    "second_agent": second_agent,
                    "common_evaluation_replicate_ids": paired[
                        "evaluation_replicate_id"
                    ].astype(int).tolist(),
                    "paired_sum_mean_profit_bb": pair_sum,
                    "absolute_paired_sum_mean_profit_bb": absolute_pair_sum,
                    "paired_sum_std_profit_bb": (
                        None if np.isnan(pair_std) else pair_std
                    ),
                    "max_absolute_pair_sum_bb": threshold,
                },
            )
        )

    return results


def validate_minimum_evaluation_replicate_coverage(
    aggregated_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    minimum_replicates = thresholds.min_evaluation_replicates_per_matchup
    if minimum_replicates < 1:
        raise ValueError(
            "min_evaluation_replicates_per_matchup must be at least 1"
        )

    results: list[ValidationCheckResult] = []
    for _, row in aggregated_rows.iterrows():
        raw_count = row.get("evaluation_replicates", 0)
        replicate_count = 0 if pd.isna(raw_count) else int(raw_count)
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        results.append(
            ValidationCheckResult(
                check_name=(
                    "Minimum evaluation replicate coverage "
                    f"for {agent_name} vs {opponent_name}"
                ),
                status=(
                    STATUS_PASS
                    if replicate_count >= minimum_replicates
                    else STATUS_FAIL
                ),
                category="evaluation_replicate_coverage",
                agent_name=agent_name,
                opponent_name=opponent_name,
                observed_value=float(replicate_count),
                threshold=float(minimum_replicates),
                sample_size=replicate_count,
                message=(
                    f"Evaluation includes {replicate_count} distinct "
                    "simulation replicate(s); minimum required is "
                    f"{minimum_replicates}."
                ),
                details={
                    "evaluation_replicate_count": replicate_count,
                    "min_evaluation_replicates_per_matchup": (
                        minimum_replicates
                    ),
                },
            )
        )

    return results


def validate_simulation_stability(
    aggregated_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for _, row in aggregated_rows.iterrows():
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        raw_value = row.get(BASELINE_REPLICATE_STD_COLUMN, np.nan)
        replicate_count = int(row.get("evaluation_replicates", 0))
        check_name = (
            "Simulation stability across evaluation replicates "
            f"for {agent_name} vs {opponent_name}"
        )

        if pd.isna(raw_value) or replicate_count < 2:
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_SKIPPED,
                    category="simulation_stability",
                    agent_name=agent_name,
                    opponent_name=opponent_name,
                    sample_size=replicate_count,
                    message=(
                        "Simulation stability requires at least two "
                        "evaluation replicates."
                    ),
                )
            )
            continue

        value = float(raw_value)
        threshold = thresholds.max_std_across_evaluation_replicates_bb
        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_PASS if value <= threshold else STATUS_WARNING,
                category="simulation_stability",
                agent_name=agent_name,
                opponent_name=opponent_name,
                observed_value=value,
                threshold=threshold,
                sample_size=replicate_count,
                message=(
                    "Mean profit std across evaluation replicates is "
                    f"{_format_float(value)} BB/game."
                ),
                details={
                    "evaluation_replicate_count": replicate_count,
                    "std_across_evaluation_replicates_bb": value,
                    "max_std_across_evaluation_replicates_bb": threshold,
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
    replicate_rows: pd.DataFrame,
) -> list[ValidationCheckResult]:
    checks: list[ValidationCheckResult] = []
    checks.extend(validate_baseline_matchup_coverage(best_rows))
    checks.extend(validate_baseline_mirror_neutrality(best_rows, thresholds))
    checks.extend(
        validate_baseline_pair_reciprocity(
            replicate_rows,
            thresholds,
        )
    )
    checks.extend(validate_baseline_extreme_results(best_rows, thresholds))
    checks.extend(
        validate_minimum_evaluation_replicate_coverage(best_rows, thresholds)
    )
    checks.extend(validate_simulation_stability(best_rows, thresholds))
    checks.extend(validate_extreme_bb_per_100(best_rows, thresholds))
    return checks

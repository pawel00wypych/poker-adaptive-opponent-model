from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.evaluation.validation.common import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_RAISE_AGENT,
    DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS,
    DEFAULT_CLASSIFIER_OPPONENTS,
    DEFAULT_ORACLE_OPPONENTS,
    OPPONENT_TYPE_TIGHT,
    ORACLE_MC_AGENT,
    POLICY_TIGHT_AGENT,
    RULE_BASED_AGENT,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    VALIDATION_MODE_CHECKPOINT,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_HEAD_TO_HEAD,
    VALIDATION_MODES,
    ValidationCheckResult,
    ValidationReport,
    ValidationThresholds,
    _add_mean_hands_played,
    _best_rows_by_agent_and_opponent,
    _checkpoint_episode,
    _find_row,
    _format_float,
    _missing_row_result,
    aggregate_across_seeds,
    load_checkpoint_report_data,
    validate_always_raise_outperforms_adaptive,
    validate_always_raise_trivial_exploit,
    validate_extreme_bb_per_100,
    validate_seed_stability,
)
from src.evaluation.validation.generalization_validation import (
    validate_generalization_results_from_best_rows,
)
from src.evaluation.validation.head_to_head_validation import (
    validate_head_to_head_results_from_best_rows,
)


def validate_adaptive_beats_rule_based(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        adaptive_row = _find_row(
            best_rows,
            ADAPTIVE_MC_AGENT,
            opponent_name,
        )
        rule_based_row = _find_row(
            best_rows,
            RULE_BASED_AGENT,
            opponent_name,
        )
        check_name = (
            "Adaptive beats rule-based "
            f"vs {opponent_name}"
        )

        if adaptive_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_delta",
                    ADAPTIVE_MC_AGENT,
                    opponent_name,
                )
            )
            continue

        if rule_based_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "baseline_delta",
                    RULE_BASED_AGENT,
                    opponent_name,
                )
            )
            continue

        delta = float(
            adaptive_row["mean_profit_bb"]
            - rule_based_row["mean_profit_bb"]
        )
        status = (
            STATUS_PASS
            if delta >= thresholds.min_adaptive_delta_vs_rule_based_bb
            else STATUS_FAIL
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=status,
                category="baseline_delta",
                agent_name=ADAPTIVE_MC_AGENT,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(adaptive_row),
                observed_value=delta,
                threshold=(
                    thresholds.min_adaptive_delta_vs_rule_based_bb
                ),
                message=(
                    "Adaptive mean profit delta vs rule-based is "
                    f"{_format_float(delta)} BB/game."
                ),
                details={
                    "adaptive_mean_profit_bb": float(
                        adaptive_row["mean_profit_bb"]
                    ),
                    "rule_based_mean_profit_bb": float(
                        rule_based_row["mean_profit_bb"]
                    ),
                    "rule_based_checkpoint_episode": _checkpoint_episode(
                        rule_based_row
                    ),
                },
            )
        )

    return results

def validate_oracle_not_worse_than_adaptive(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_ORACLE_OPPONENTS,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        oracle_row = _find_row(
            best_rows,
            ORACLE_MC_AGENT,
            opponent_name,
        )
        adaptive_row = _find_row(
            best_rows,
            ADAPTIVE_MC_AGENT,
            opponent_name,
        )
        check_name = (
            "Oracle not significantly worse than adaptive "
            f"vs {opponent_name}"
        )

        if oracle_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "oracle_gap",
                    ORACLE_MC_AGENT,
                    opponent_name,
                )
            )
            continue

        if adaptive_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "oracle_gap",
                    ADAPTIVE_MC_AGENT,
                    opponent_name,
                )
            )
            continue

        oracle_gap = float(
            oracle_row["mean_profit_bb"]
            - adaptive_row["mean_profit_bb"]
        )
        status = (
            STATUS_PASS
            if oracle_gap >= -thresholds.max_oracle_underperformance_bb
            else STATUS_WARNING
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=status,
                category="oracle_gap",
                agent_name=ORACLE_MC_AGENT,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(oracle_row),
                observed_value=oracle_gap,
                threshold=-thresholds.max_oracle_underperformance_bb,
                message=(
                    "Oracle minus adaptive mean profit is "
                    f"{_format_float(oracle_gap)} BB/game."
                ),
                details={
                    "oracle_mean_profit_bb": float(
                        oracle_row["mean_profit_bb"]
                    ),
                    "adaptive_mean_profit_bb": float(
                        adaptive_row["mean_profit_bb"]
                    ),
                    "adaptive_checkpoint_episode": _checkpoint_episode(
                        adaptive_row
                    ),
                },
            )
        )

    return results

def validate_tight_exploitation(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    adaptive_row = _find_row(
        best_rows,
        ADAPTIVE_MC_AGENT,
        OPPONENT_TYPE_TIGHT,
    )
    check_name = "Adaptive exploits TightPlayer"

    if adaptive_row is None:
        return [
            _missing_row_result(
                check_name,
                "tight_exploitation",
                ADAPTIVE_MC_AGENT,
                OPPONENT_TYPE_TIGHT,
            )
        ]

    mean_profit_bb = float(adaptive_row["mean_profit_bb"])
    win_rate = float(adaptive_row["win_rate"])
    passed = (
        mean_profit_bb >= thresholds.min_tight_mean_profit_bb
        and win_rate >= thresholds.min_tight_win_rate
    )

    return [
        ValidationCheckResult(
            check_name=check_name,
            status=STATUS_PASS if passed else STATUS_FAIL,
            category="tight_exploitation",
            agent_name=ADAPTIVE_MC_AGENT,
            opponent_name=OPPONENT_TYPE_TIGHT,
            checkpoint_episode=_checkpoint_episode(adaptive_row),
            observed_value=mean_profit_bb,
            threshold=thresholds.min_tight_mean_profit_bb,
            message=(
                "Adaptive vs tight: "
                f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                f"win_rate={_format_float(win_rate)}%."
            ),
            details={
                "mean_profit_bb": mean_profit_bb,
                "min_mean_profit_bb": (
                    thresholds.min_tight_mean_profit_bb
                ),
                "win_rate": win_rate,
                "min_win_rate": thresholds.min_tight_win_rate,
            },
        )
    ]

def validate_classifier_quality(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_CLASSIFIER_OPPONENTS,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        adaptive_row = _find_row(
            best_rows,
            ADAPTIVE_MC_AGENT,
            opponent_name,
        )

        for metric_name, threshold in [
            (
                "global_classifier_accuracy",
                thresholds.min_classifier_accuracy,
            ),
            (
                "global_classifier_coverage",
                thresholds.min_classifier_coverage,
            ),
        ]:
            pretty_metric = metric_name.replace("global_classifier_", "")
            check_name = (
                f"Adaptive classifier {pretty_metric} "
                f"vs {opponent_name}"
            )

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "classifier_quality",
                        ADAPTIVE_MC_AGENT,
                        opponent_name,
                    )
                )
                continue

            value = float(adaptive_row[metric_name])
            status = (
                STATUS_PASS
                if value >= threshold
                else STATUS_WARNING
            )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    category="classifier_quality",
                    agent_name=ADAPTIVE_MC_AGENT,
                    opponent_name=opponent_name,
                    checkpoint_episode=_checkpoint_episode(adaptive_row),
                    observed_value=value,
                    threshold=threshold,
                    message=(
                        f"Adaptive classifier {pretty_metric} is "
                        f"{_format_float(value)}%."
                    ),
                    details={
                        "metric": metric_name,
                        "value": value,
                        "minimum": threshold,
                    },
                )
            )

    return results

def validate_tight_baseline_saturation(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    required_agents = (
        ADAPTIVE_MC_AGENT,
        RULE_BASED_AGENT,
        ALWAYS_RAISE_AGENT,
    )
    rows_by_agent = {
        agent_name: _find_row(
            best_rows,
            agent_name,
            OPPONENT_TYPE_TIGHT,
        )
        for agent_name in required_agents
    }
    check_name = "TightPlayer baseline saturation sanity check"

    for agent_name, row in rows_by_agent.items():
        if row is None:
            return [
                _missing_row_result(
                    check_name,
                    "always_raise_sanity",
                    agent_name,
                    OPPONENT_TYPE_TIGHT,
                )
            ]

    assert all(row is not None for row in rows_by_agent.values())

    saturated_agents: list[str] = []
    details: dict[str, object] = {}

    for agent_name, row in rows_by_agent.items():
        assert row is not None
        mean_profit_bb = float(row["mean_profit_bb"])
        win_rate = float(row["win_rate"])
        details[f"{agent_name}_mean_profit_bb"] = mean_profit_bb
        details[f"{agent_name}_win_rate"] = win_rate

        if (
            mean_profit_bb >= thresholds.min_tight_mean_profit_bb
            and win_rate >= thresholds.min_tight_win_rate
        ):
            saturated_agents.append(agent_name)

    is_saturated = len(saturated_agents) == len(required_agents)

    return [
        ValidationCheckResult(
            check_name=check_name,
            status=STATUS_WARNING if is_saturated else STATUS_PASS,
            category="always_raise_sanity",
            agent_name=ALWAYS_RAISE_AGENT,
            opponent_name=OPPONENT_TYPE_TIGHT,
            checkpoint_episode=_checkpoint_episode(
                rows_by_agent[ALWAYS_RAISE_AGENT]
            ),
            observed_value=float(
                rows_by_agent[ALWAYS_RAISE_AGENT]["mean_profit_bb"]
            ),
            threshold=thresholds.min_tight_mean_profit_bb,
            message=(
                "TightPlayer may be too weak to distinguish agent "
                "quality when adaptive, rule-based, and always-raise "
                "all reach the tight exploitation thresholds."
                if is_saturated
                else "TightPlayer still differentiates at least one "
                "baseline below the exploitation thresholds."
            ),
            details={
                **details,
                "saturated_agents": saturated_agents,
                "required_agents": list(required_agents),
                "min_tight_mean_profit_bb": (
                    thresholds.min_tight_mean_profit_bb
                ),
                "min_tight_win_rate": thresholds.min_tight_win_rate,
            },
        )
    ]

def validate_checkpoint_results(
    input_path: str | Path,
    thresholds: ValidationThresholds | None = None,
    validation_mode: str = VALIDATION_MODE_CHECKPOINT,
) -> ValidationReport:
    if validation_mode not in VALIDATION_MODES:
        raise ValueError(
            "Unsupported validation_mode "
            f"{validation_mode!r}. Expected one of {VALIDATION_MODES}."
        )

    thresholds = thresholds or ValidationThresholds()
    metrics = load_checkpoint_report_data(input_path)
    aggregated = aggregate_across_seeds(metrics)
    aggregated = _add_mean_hands_played(aggregated, metrics)
    best_rows = _best_rows_by_agent_and_opponent(aggregated)

    if validation_mode == VALIDATION_MODE_HEAD_TO_HEAD:
        checks = validate_head_to_head_results_from_best_rows(
            best_rows,
            thresholds,
        )
    elif validation_mode == VALIDATION_MODE_GENERALIZATION:
        checks = validate_generalization_results_from_best_rows(
            best_rows,
            thresholds,
        )
    else:
        checks: list[ValidationCheckResult] = []
        checks.extend(
            validate_adaptive_beats_rule_based(
                best_rows,
                thresholds,
            )
        )
        checks.extend(
            validate_oracle_not_worse_than_adaptive(
                best_rows,
                thresholds,
            )
        )
        checks.extend(
            validate_tight_exploitation(
                best_rows,
                thresholds,
            )
        )
        checks.extend(
            validate_classifier_quality(
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
        checks.extend(
            validate_always_raise_outperforms_adaptive(
                best_rows,
                thresholds,
            )
        )
        checks.extend(
            validate_always_raise_trivial_exploit(
                best_rows,
                thresholds,
            )
        )
        checks.extend(
            validate_tight_baseline_saturation(
                best_rows,
                thresholds,
            )
        )

    return ValidationReport(
        input_path=str(input_path),
        thresholds=thresholds,
        checks=checks,
        validation_mode=validation_mode,
    )

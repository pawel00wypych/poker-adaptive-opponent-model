from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ALGORITHM_VALIDATION_SPECS,
    AlgorithmValidationSpec,
    available_algorithm_specs,
)
from src.evaluation.constants import ALWAYS_RAISE_AGENT, RULE_BASED_AGENT
from src.evaluation.metrics.baseline_metrics import (
    aggregate_across_evaluation_replicates,
    calculate_baseline_replicate_metrics,
)
from src.evaluation.metrics.oracle_gap import (
    ORACLE_GAP_BB_COLUMN,
    calculate_oracle_gap_bb,
)
from src.evaluation.reporting.training_opponent_report import (
    aggregate_across_seeds,
    load_training_opponent_report_data,
)
from src.evaluation.validation.baseline_sanity_validation import (
    BASELINE_SANITY_AGENTS,
    validate_baseline_sanity_results_from_final_rows,
)
from src.evaluation.validation.common import (
    DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS,
    DEFAULT_CLASSIFIER_OPPONENTS,
    DEFAULT_ORACLE_OPPONENTS,
    HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    HEAD_TO_HEAD_RULE_BASED_OPPONENT,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    CheckKind,
    VALIDATION_MODE_BASELINE_SANITY,
    VALIDATION_MODE_CROSS_PLAY,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_HEAD_TO_HEAD,
    VALIDATION_MODE_STRESS_TEST,
    VALIDATION_MODE_TRAINING_OPPONENT,
    VALIDATION_MODES,
    ValidationCheckResult,
    ValidationReport,
    ValidationThresholds,
    _add_mean_hands_played,
    _find_row,
    _find_rows_at_common_training_episode,
    _format_float,
    _minimum_delta_status,
    _missing_common_training_episode_result,
    _missing_row_result,
    _paired_seed_message,
    _paired_seed_statistics_for_check,
    _training_episode,
    validate_always_raise_outperforms_adaptive,
    validate_always_raise_trivial_exploit,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)
from src.evaluation.validation.cross_play_validation import (
    validate_cross_play_matchup_coverage,
    validate_cross_play_results_from_final_rows,
)
from src.evaluation.validation.context import (
    EvaluationContext,
    EvaluationManifest,
)
from src.evaluation.validation.generalization_validation import (
    validate_generalization_results_from_final_rows,
)
from src.evaluation.validation.head_to_head_validation import (
    validate_head_to_head_results_from_final_rows,
)
from src.evaluation.validation.integrity import validate_raw_evaluation_integrity
from src.evaluation.validation.stress_test_validation import (
    STRESS_TEST_OPPONENTS,
    validate_stress_test_results_from_final_rows,
)
from src.evaluation.validation.suites import SUITES
from src.players.constants import GENERALIZATION_OPPONENTS
from src.poker.constants import OPPONENT_TYPE_TIGHT, TRAINING_OPPONENT_TYPES


def _algorithm_specs(
    final_rows: pd.DataFrame,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> tuple[AlgorithmValidationSpec, ...]:
    return tuple(algorithm_specs or available_algorithm_specs(final_rows))


_REQUIRED_ALGORITHM_ROLES_BY_MODE = {
    VALIDATION_MODE_TRAINING_OPPONENT: ("adaptive", "oracle", "policy_general"),
    VALIDATION_MODE_GENERALIZATION: ("adaptive", "oracle", "policy_general"),
    VALIDATION_MODE_HEAD_TO_HEAD: ("adaptive", "policy_general"),
    VALIDATION_MODE_STRESS_TEST: ("adaptive", "policy_general"),
    VALIDATION_MODE_BASELINE_SANITY: (),
    VALIDATION_MODE_CROSS_PLAY: ("adaptive",),
}

_REQUIRED_MATCHUP_OPPONENTS_BY_MODE = {
    VALIDATION_MODE_TRAINING_OPPONENT: TRAINING_OPPONENT_TYPES,
    VALIDATION_MODE_GENERALIZATION: GENERALIZATION_OPPONENTS,
    VALIDATION_MODE_HEAD_TO_HEAD: (
        HEAD_TO_HEAD_RULE_BASED_OPPONENT,
        HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT,
    ),
    VALIDATION_MODE_STRESS_TEST: STRESS_TEST_OPPONENTS,
    VALIDATION_MODE_BASELINE_SANITY: BASELINE_SANITY_AGENTS,
}


def _required_algorithm_agents(
    spec: AlgorithmValidationSpec,
    validation_mode: str,
) -> dict[str, str]:
    agents_by_role = {
        "adaptive": spec.adaptive_agent,
        "oracle": spec.oracle_agent,
        "policy_general": spec.general_policy_agent,
    }
    return {
        role: agents_by_role[role]
        for role in _REQUIRED_ALGORITHM_ROLES_BY_MODE[validation_mode]
    }


def validate_expected_algorithms_present(
    final_rows: pd.DataFrame,
    expected_specs: Iterable[AlgorithmValidationSpec],
    *,
    fail_when_missing: bool,
    validation_mode: str = VALIDATION_MODE_TRAINING_OPPONENT,
) -> list[ValidationCheckResult]:
    if validation_mode not in _REQUIRED_ALGORITHM_ROLES_BY_MODE:
        raise ValueError(
            "Unsupported validation_mode "
            f"{validation_mode!r}. Expected one of {VALIDATION_MODES}."
        )

    available_agents = (
        set(final_rows["agent_name"].dropna())
        if "agent_name" in final_rows.columns
        else set()
    )
    results: list[ValidationCheckResult] = []

    for spec in expected_specs:
        required_agents_by_role = _required_algorithm_agents(
            spec,
            validation_mode,
        )
        present_roles = [
            role
            for role, agent_name in required_agents_by_role.items()
            if agent_name in available_agents
        ]
        missing_roles = [
            role
            for role, agent_name in required_agents_by_role.items()
            if agent_name not in available_agents
        ]
        present_agents = [required_agents_by_role[role] for role in present_roles]
        missing_agents = [required_agents_by_role[role] for role in missing_roles]
        complete = not missing_agents
        if complete:
            status = STATUS_PASS
        elif fail_when_missing:
            status = STATUS_FAIL
        else:
            status = STATUS_WARNING

        missing_description = ", ".join(
            f"{role} ({required_agents_by_role[role]})" for role in missing_roles
        )
        results.append(
            ValidationCheckResult(
                check_name=(f"{spec.algorithm_name}: Algorithm result coverage"),
                status=status,
                check_type=CheckKind.INTEGRITY,
                category="algorithm_coverage",
                algorithm_name=spec.algorithm_name,
                message=(
                    "All required agent roles are present for this algorithm."
                    if complete
                    else f"Missing required agent roles: {missing_description}."
                ),
                details={
                    "algorithm_key": spec.algorithm_key,
                    "validation_mode": validation_mode,
                    "required_roles": list(required_agents_by_role),
                    "present_roles": present_roles,
                    "missing_roles": missing_roles,
                    "required_agents": list(required_agents_by_role.values()),
                    "present_agents": present_agents,
                    "missing_agents": missing_agents,
                    "adaptive_agent": spec.adaptive_agent,
                    "oracle_agent": spec.oracle_agent,
                    "general_policy_agent": spec.general_policy_agent,
                    "present": complete,
                },
            )
        )

    return results


def validate_required_matchups_present(
    final_rows: pd.DataFrame,
    expected_specs: Iterable[AlgorithmValidationSpec],
    *,
    fail_when_missing: bool,
    validation_mode: str = VALIDATION_MODE_TRAINING_OPPONENT,
) -> list[ValidationCheckResult]:
    if validation_mode == VALIDATION_MODE_CROSS_PLAY:
        return validate_cross_play_matchup_coverage(
            final_rows,
            expected_specs,
            fail_when_missing=fail_when_missing,
        )

    if validation_mode not in _REQUIRED_MATCHUP_OPPONENTS_BY_MODE:
        raise ValueError(
            "Unsupported validation_mode "
            f"{validation_mode!r}. Expected one of {VALIDATION_MODES}."
        )

    available_matchups = (
        set(
            final_rows[["agent_name", "opponent_name"]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        if {"agent_name", "opponent_name"}.issubset(final_rows.columns)
        else set()
    )
    required_opponents = _REQUIRED_MATCHUP_OPPONENTS_BY_MODE[validation_mode]
    results: list[ValidationCheckResult] = []

    for spec in expected_specs:
        required_agents_by_role = _required_algorithm_agents(
            spec,
            validation_mode,
        )
        required_matchups = [
            {
                "role": role,
                "agent_name": agent_name,
                "opponent_name": opponent_name,
            }
            for role, agent_name in required_agents_by_role.items()
            for opponent_name in required_opponents
        ]
        present_matchups = [
            matchup
            for matchup in required_matchups
            if (
                matchup["agent_name"],
                matchup["opponent_name"],
            )
            in available_matchups
        ]
        missing_matchups = [
            matchup
            for matchup in required_matchups
            if (
                matchup["agent_name"],
                matchup["opponent_name"],
            )
            not in available_matchups
        ]
        complete = not missing_matchups
        if complete:
            status = STATUS_PASS
        elif fail_when_missing:
            status = STATUS_FAIL
        else:
            status = STATUS_WARNING

        missing_description = ", ".join(
            f"{matchup['agent_name']} vs {matchup['opponent_name']}"
            for matchup in missing_matchups
        )
        results.append(
            ValidationCheckResult(
                check_name=(f"{spec.algorithm_name}: Required matchup coverage"),
                status=status,
                check_type=CheckKind.INTEGRITY,
                category="matchup_coverage",
                algorithm_name=spec.algorithm_name,
                message=(
                    "All required evaluation matchups are present for this algorithm."
                    if complete
                    else (
                        f"Missing {len(missing_matchups)} of "
                        f"{len(required_matchups)} required evaluation "
                        f"matchups: {missing_description}."
                    )
                ),
                details={
                    "algorithm_key": spec.algorithm_key,
                    "validation_mode": validation_mode,
                    "required_roles": list(required_agents_by_role),
                    "required_agents": list(required_agents_by_role.values()),
                    "required_opponents": list(required_opponents),
                    "required_matchup_count": len(required_matchups),
                    "present_matchup_count": len(present_matchups),
                    "missing_matchup_count": len(missing_matchups),
                    "required_matchups": required_matchups,
                    "present_matchups": present_matchups,
                    "missing_matchups": missing_matchups,
                    "present": complete,
                },
            )
        )

    return results


def validate_adaptive_beats_rule_based(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(final_rows, algorithm_specs):
        for opponent_name in opponents:
            matchups = (
                (spec.adaptive_agent, opponent_name),
                (RULE_BASED_AGENT, opponent_name),
            )
            training_episode, rows = _find_rows_at_common_training_episode(
                final_rows, matchups
            )
            adaptive_row, rule_based_row = rows
            check_name = (
                f"{spec.algorithm_name}: Adaptive beats rule-based vs {opponent_name}"
            )

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "baseline_delta",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
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
                        spec.algorithm_name,
                    )
                )
                continue

            if training_episode is None:
                results.append(
                    _missing_common_training_episode_result(
                        check_name,
                        "baseline_delta",
                        final_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            paired_statistics, unavailable_result = _paired_seed_statistics_for_check(
                seed_rows,
                left_agent_name=spec.adaptive_agent,
                right_agent_name=RULE_BASED_AGENT,
                opponent_name=opponent_name,
                training_episode=training_episode,
                thresholds=thresholds,
                check_name=check_name,
                category="baseline_delta",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
            )
            if unavailable_result is not None:
                results.append(unavailable_result)
                continue

            if paired_statistics is None:
                delta = float(
                    adaptive_row["mean_profit_bb"] - rule_based_row["mean_profit_bb"]
                )
                status = (
                    STATUS_PASS
                    if delta >= thresholds.min_adaptive_delta_vs_rule_based_bb
                    else STATUS_WARNING
                )
                message = (
                    "Adaptive mean profit delta vs rule-based is "
                    f"{_format_float(delta)} BB/game."
                )
            else:
                delta = float(paired_statistics.mean_delta)
                status = _minimum_delta_status(
                    paired_statistics,
                    thresholds.min_adaptive_delta_vs_rule_based_bb,
                )
                message = _paired_seed_message(
                    "Adaptive minus rule-based mean profit",
                    paired_statistics,
                )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    category="baseline_delta",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    training_episode=training_episode,
                    observed_value=delta,
                    threshold=(thresholds.min_adaptive_delta_vs_rule_based_bb),
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
                        "adaptive_mean_profit_bb": float(
                            adaptive_row["mean_profit_bb"]
                        ),
                        "rule_based_mean_profit_bb": float(
                            rule_based_row["mean_profit_bb"]
                        ),
                        "rule_based_training_episode": _training_episode(
                            rule_based_row
                        ),
                        **(
                            {"paired_seed_statistics": (paired_statistics.to_details())}
                            if paired_statistics is not None
                            else {}
                        ),
                    },
                )
            )

    return results


def validate_adaptation_gain_training(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = TRAINING_OPPONENT_TYPES,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    """Report H1-linked adaptive-minus-general effects without gating validity."""

    results: list[ValidationCheckResult] = []
    for spec in _algorithm_specs(final_rows, algorithm_specs):
        for opponent_name in opponents:
            matchups = (
                (spec.adaptive_agent, opponent_name),
                (spec.general_policy_agent, opponent_name),
            )
            training_episode, rows = _find_rows_at_common_training_episode(
                final_rows,
                matchups,
            )
            adaptive_row, general_row = rows
            check_name = (
                f"{spec.algorithm_name}: Adaptive gain vs fixed general "
                f"vs {opponent_name}"
            )
            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "adaptation_gain_training",
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
                        "adaptation_gain_training",
                        spec.general_policy_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue
            if training_episode is None:
                results.append(
                    _missing_common_training_episode_result(
                        check_name,
                        "adaptation_gain_training",
                        final_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            paired_statistics, unavailable_result = _paired_seed_statistics_for_check(
                seed_rows,
                left_agent_name=spec.adaptive_agent,
                right_agent_name=spec.general_policy_agent,
                opponent_name=opponent_name,
                training_episode=training_episode,
                thresholds=thresholds,
                check_name=check_name,
                category="adaptation_gain_training",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
            )
            if unavailable_result is not None:
                results.append(unavailable_result)
                continue

            if paired_statistics is None:
                delta = float(
                    adaptive_row["mean_profit_bb"] - general_row["mean_profit_bb"]
                )
                status = STATUS_PASS if delta > 0.0 else STATUS_WARNING
                message = (
                    "Adaptive minus fixed general mean profit is "
                    f"{_format_float(delta)} BB/game."
                )
            else:
                delta = float(paired_statistics.mean_delta)
                status = _minimum_delta_status(
                    paired_statistics,
                    0.0,
                    underperformance_status=STATUS_WARNING,
                )
                message = _paired_seed_message(
                    "Adaptive minus fixed general mean profit",
                    paired_statistics,
                )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    check_id=(
                        f"h1_adaptation_gain_{spec.algorithm_key}_{opponent_name}"
                    ),
                    hypothesis_id="H1",
                    status=status,
                    category="adaptation_gain_training",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    training_episode=training_episode,
                    observed_value=delta,
                    threshold=0.0,
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
                        "adaptive_mean_profit_bb": float(
                            adaptive_row["mean_profit_bb"]
                        ),
                        "general_mean_profit_bb": float(
                            general_row["mean_profit_bb"]
                        ),
                        **(
                            {"paired_seed_statistics": paired_statistics.to_details()}
                            if paired_statistics is not None
                            else {}
                        ),
                    },
                )
            )
    return results


def validate_oracle_not_worse_than_adaptive(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_ORACLE_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(final_rows, algorithm_specs):
        for opponent_name in opponents:
            matchups = (
                (spec.oracle_agent, opponent_name),
                (spec.adaptive_agent, opponent_name),
            )
            training_episode, rows = _find_rows_at_common_training_episode(
                final_rows, matchups
            )
            oracle_row, adaptive_row = rows
            check_name = (
                f"{spec.algorithm_name}: Oracle not significantly worse "
                f"than adaptive vs {opponent_name}"
            )

            if oracle_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "oracle_gap",
                        spec.oracle_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "oracle_gap",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            if training_episode is None:
                results.append(
                    _missing_common_training_episode_result(
                        check_name,
                        "oracle_gap",
                        final_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.oracle_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            paired_statistics, unavailable_result = _paired_seed_statistics_for_check(
                seed_rows,
                left_agent_name=spec.oracle_agent,
                right_agent_name=spec.adaptive_agent,
                opponent_name=opponent_name,
                training_episode=training_episode,
                thresholds=thresholds,
                check_name=check_name,
                category="oracle_gap",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.oracle_agent,
            )
            if unavailable_result is not None:
                results.append(unavailable_result)
                continue

            threshold = -thresholds.max_oracle_underperformance_bb
            if paired_statistics is None:
                oracle_gap_bb = float(
                    calculate_oracle_gap_bb(
                        oracle_row["mean_profit_bb"],
                        adaptive_row["mean_profit_bb"],
                    )
                )
                status = STATUS_PASS if oracle_gap_bb >= threshold else STATUS_WARNING
                message = (
                    "Oracle minus adaptive mean profit is "
                    f"{_format_float(oracle_gap_bb)} BB/game."
                )
            else:
                oracle_gap_bb = float(paired_statistics.mean_delta)
                status = _minimum_delta_status(
                    paired_statistics,
                    threshold,
                    underperformance_status=STATUS_WARNING,
                )
                message = _paired_seed_message(
                    "Oracle minus adaptive mean profit",
                    paired_statistics,
                )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    hypothesis_id="H3",
                    category="oracle_gap",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.oracle_agent,
                    opponent_name=opponent_name,
                    training_episode=training_episode,
                    observed_value=oracle_gap_bb,
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
                        "oracle_agent": spec.oracle_agent,
                        "adaptive_agent": spec.adaptive_agent,
                        "oracle_mean_profit_bb": float(oracle_row["mean_profit_bb"]),
                        "adaptive_mean_profit_bb": float(
                            adaptive_row["mean_profit_bb"]
                        ),
                        ORACLE_GAP_BB_COLUMN: oracle_gap_bb,
                        "adaptive_training_episode": _training_episode(adaptive_row),
                        **(
                            {"paired_seed_statistics": (paired_statistics.to_details())}
                            if paired_statistics is not None
                            else {}
                        ),
                    },
                )
            )

    return results


def validate_tight_exploitation(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(final_rows, algorithm_specs):
        adaptive_row = _find_row(
            final_rows,
            spec.adaptive_agent,
            OPPONENT_TYPE_TIGHT,
        )
        check_name = f"{spec.algorithm_name}: Adaptive exploits TightPlayer"

        if adaptive_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "tight_exploitation",
                    spec.adaptive_agent,
                    OPPONENT_TYPE_TIGHT,
                    spec.algorithm_name,
                )
            )
            continue

        mean_profit_bb = float(adaptive_row["mean_profit_bb"])
        win_rate = float(adaptive_row["win_rate"])
        passed = (
            mean_profit_bb >= thresholds.min_tight_mean_profit_bb
            and win_rate >= thresholds.min_tight_win_rate
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_PASS if passed else STATUS_WARNING,
                category="tight_exploitation",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
                opponent_name=OPPONENT_TYPE_TIGHT,
                training_episode=_training_episode(adaptive_row),
                observed_value=mean_profit_bb,
                threshold=thresholds.min_tight_mean_profit_bb,
                message=(
                    "Adaptive vs tight: "
                    f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                    f"win_rate={_format_float(win_rate)}%."
                ),
                details={
                    "algorithm": spec.algorithm_name,
                    "adaptive_agent": spec.adaptive_agent,
                    "mean_profit_bb": mean_profit_bb,
                    "min_mean_profit_bb": thresholds.min_tight_mean_profit_bb,
                    "win_rate": win_rate,
                    "min_win_rate": thresholds.min_tight_win_rate,
                },
            )
        )

    return results


def validate_classifier_quality(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = DEFAULT_CLASSIFIER_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(final_rows, algorithm_specs):
        for opponent_name in opponents:
            adaptive_row = _find_row(
                final_rows,
                spec.adaptive_agent,
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
                    f"{spec.algorithm_name}: Adaptive classifier "
                    f"{pretty_metric} vs {opponent_name}"
                )

                if adaptive_row is None:
                    results.append(
                        _missing_row_result(
                            check_name,
                            "classifier_quality",
                            spec.adaptive_agent,
                            opponent_name,
                            spec.algorithm_name,
                        )
                    )
                    continue

                value = float(adaptive_row[metric_name])
                status = STATUS_PASS if value >= threshold else STATUS_WARNING

                results.append(
                    ValidationCheckResult(
                        check_name=check_name,
                        status=status,
                        category="classifier_quality",
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                        training_episode=_training_episode(adaptive_row),
                        observed_value=value,
                        threshold=threshold,
                        message=(
                            f"Adaptive classifier {pretty_metric} is "
                            f"{_format_float(value)}%."
                        ),
                        details={
                            "algorithm": spec.algorithm_name,
                            "adaptive_agent": spec.adaptive_agent,
                            "metric": metric_name,
                            "value": value,
                            "minimum": threshold,
                        },
                    )
                )

    return results


def validate_tight_baseline_saturation(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(final_rows, algorithm_specs):
        required_agents = (
            spec.adaptive_agent,
            RULE_BASED_AGENT,
            ALWAYS_RAISE_AGENT,
        )
        matchups = tuple(
            (agent_name, OPPONENT_TYPE_TIGHT) for agent_name in required_agents
        )
        training_episode, rows = _find_rows_at_common_training_episode(
            final_rows,
            matchups,
        )
        rows_by_agent = dict(zip(required_agents, rows, strict=True))
        check_name = (
            f"{spec.algorithm_name}: TightPlayer baseline saturation sanity check"
        )

        missing = False
        for agent_name, row in rows_by_agent.items():
            if row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "always_raise_sanity",
                        agent_name,
                        OPPONENT_TYPE_TIGHT,
                        spec.algorithm_name,
                    )
                )
                missing = True
                break
        if missing:
            continue

        if training_episode is None:
            results.append(
                _missing_common_training_episode_result(
                    check_name,
                    "always_raise_sanity",
                    final_rows,
                    matchups,
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=OPPONENT_TYPE_TIGHT,
                )
            )
            continue

        assert all(row is not None for row in rows_by_agent.values())

        saturated_agents: list[str] = []
        details: dict[str, object] = {
            "algorithm": spec.algorithm_name,
            "adaptive_agent": spec.adaptive_agent,
        }

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
        always_raise_row = rows_by_agent[ALWAYS_RAISE_AGENT]
        assert always_raise_row is not None

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_WARNING if is_saturated else STATUS_PASS,
                category="always_raise_sanity",
                algorithm_name=spec.algorithm_name,
                agent_name=ALWAYS_RAISE_AGENT,
                opponent_name=OPPONENT_TYPE_TIGHT,
                training_episode=training_episode,
                observed_value=float(always_raise_row["mean_profit_bb"]),
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
                    "min_tight_mean_profit_bb": thresholds.min_tight_mean_profit_bb,
                    "min_tight_win_rate": thresholds.min_tight_win_rate,
                },
            )
        )

    return results


def _select_final_model_rows(
    aggregated: pd.DataFrame,
) -> tuple[pd.DataFrame, int, str]:
    if aggregated.empty or "training_episode" not in aggregated.columns:
        raise ValueError("No final-model evaluation rows were found.")

    available_training_episodes = sorted(
        {int(value) for value in aggregated["training_episode"].dropna()}
    )
    if not available_training_episodes:
        raise ValueError("No training episodes were found in evaluation data.")

    if len(available_training_episodes) != 1:
        raise ValueError(
            "Final evaluation data must contain exactly one training episode. "
            "Evaluate checkpoints separately with the learning-curve workflow. "
            f"Found training episodes: {available_training_episodes}."
        )

    return (
        aggregated.reset_index(drop=True),
        available_training_episodes[0],
        "final",
    )


def _integrity_failure(
    *,
    check_id: str,
    message: str,
    details: dict[str, object] | None = None,
) -> ValidationCheckResult:
    return ValidationCheckResult(
        check_id=check_id,
        check_name=message,
        check_type=CheckKind.INTEGRITY,
        status=STATUS_FAIL,
        category="assessment_pipeline",
        message=message,
        details=details,
    )


def _build_evaluation_context(
    *,
    input_path: Path,
    validation_mode: str,
    raw_games: pd.DataFrame,
    manifest: EvaluationManifest | None,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None,
    require_all_algorithms: bool,
) -> EvaluationContext:
    if SUITES[validation_mode].baseline_only:
        replicate_metrics = calculate_baseline_replicate_metrics(input_path)
        aggregated = aggregate_across_evaluation_replicates(replicate_metrics)
        return EvaluationContext(
            input_path=input_path,
            validation_mode=validation_mode,
            raw_games=raw_games,
            aggregated=aggregated,
            seed_rows=None,
            replicate_rows=replicate_metrics,
            manifest=manifest,
            algorithm_specs=(),
            selected_training_episode=None,
            model_selection="not_applicable",
        )

    metrics = load_training_opponent_report_data(input_path)
    aggregated = aggregate_across_seeds(metrics)
    aggregated = _add_mean_hands_played(aggregated, metrics)
    validation_rows, selected_training_episode, model_selection = (
        _select_final_model_rows(aggregated)
    )
    seed_rows = (
        metrics[metrics["training_episode"] == selected_training_episode]
        .reset_index(drop=True)
        if "training_episode" in metrics.columns
        else None
    )
    expected_specs = tuple(
        algorithm_specs
        if algorithm_specs is not None
        else (
            ALGORITHM_VALIDATION_SPECS
            if require_all_algorithms or validation_mode == VALIDATION_MODE_CROSS_PLAY
            else available_algorithm_specs(aggregated)
        )
    )
    return EvaluationContext(
        input_path=input_path,
        validation_mode=validation_mode,
        raw_games=raw_games,
        aggregated=validation_rows,
        seed_rows=seed_rows,
        replicate_rows=None,
        manifest=manifest,
        algorithm_specs=expected_specs,
        selected_training_episode=selected_training_episode,
        model_selection=model_selection,
    )


def _coverage_checks(
    context: EvaluationContext,
    *,
    algorithm_specs_were_selected: bool,
    require_all_algorithms: bool,
) -> list[ValidationCheckResult]:
    suite = SUITES[context.validation_mode]
    if not (
        require_all_algorithms
        or algorithm_specs_were_selected
        or context.validation_mode == VALIDATION_MODE_CROSS_PLAY
    ):
        return []

    checks: list[ValidationCheckResult] = []
    if suite.requires_algorithm_coverage:
        checks.extend(
            validate_expected_algorithms_present(
                context.aggregated,
                context.algorithm_specs,
                fail_when_missing=require_all_algorithms,
                validation_mode=context.validation_mode,
            )
        )
    if suite.requires_matchup_coverage:
        checks.extend(
            validate_required_matchups_present(
                context.aggregated,
                context.algorithm_specs,
                fail_when_missing=require_all_algorithms,
                validation_mode=context.validation_mode,
            )
        )
    return checks


def _run_training_opponent_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    rows = context.aggregated
    specs = context.algorithm_specs
    checks: list[ValidationCheckResult] = []
    checks.extend(
        validate_adaptation_gain_training(
            rows,
            thresholds,
            algorithm_specs=specs,
            seed_rows=context.seed_rows,
        )
    )
    checks.extend(
        validate_adaptive_beats_rule_based(
            rows,
            thresholds,
            algorithm_specs=specs,
            seed_rows=context.seed_rows,
        )
    )
    checks.extend(
        validate_oracle_not_worse_than_adaptive(
            rows,
            thresholds,
            algorithm_specs=specs,
            seed_rows=context.seed_rows,
        )
    )
    checks.extend(validate_tight_exploitation(rows, thresholds, specs))
    checks.extend(validate_classifier_quality(rows, thresholds, algorithm_specs=specs))
    checks.extend(validate_minimum_seed_coverage(rows, thresholds))
    checks.extend(validate_seed_stability(rows, thresholds))
    checks.extend(validate_extreme_bb_per_100(rows, thresholds))
    checks.extend(
        validate_always_raise_outperforms_adaptive(
            rows,
            thresholds,
            algorithm_specs=specs,
            seed_rows=context.seed_rows,
        )
    )
    checks.extend(validate_always_raise_trivial_exploit(rows, thresholds))
    checks.extend(validate_tight_baseline_saturation(rows, thresholds, specs))
    return checks


def _run_generalization_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    return validate_generalization_results_from_final_rows(
        context.aggregated,
        thresholds,
        algorithm_specs=context.algorithm_specs,
        comparison_rows=context.aggregated,
        seed_rows=context.seed_rows,
    )


def _run_stress_test_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    return validate_stress_test_results_from_final_rows(
        context.aggregated,
        thresholds,
        algorithm_specs=context.algorithm_specs,
        comparison_rows=context.aggregated,
        seed_rows=context.seed_rows,
    )


def _run_head_to_head_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    return validate_head_to_head_results_from_final_rows(
        context.aggregated,
        thresholds,
        algorithm_specs=context.algorithm_specs,
        comparison_rows=context.aggregated,
        seed_rows=context.seed_rows,
    )


def _run_cross_play_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    return validate_cross_play_results_from_final_rows(
        context.aggregated,
        thresholds,
        algorithm_specs=context.algorithm_specs,
        comparison_rows=context.aggregated,
        seed_rows=context.seed_rows,
    )


def _run_baseline_sanity_suite(
    context: EvaluationContext,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    assert context.replicate_rows is not None
    return validate_baseline_sanity_results_from_final_rows(
        context.aggregated,
        thresholds,
        replicate_rows=context.replicate_rows,
    )


SUITE_RUNNERS = {
    VALIDATION_MODE_TRAINING_OPPONENT: _run_training_opponent_suite,
    VALIDATION_MODE_GENERALIZATION: _run_generalization_suite,
    VALIDATION_MODE_STRESS_TEST: _run_stress_test_suite,
    VALIDATION_MODE_HEAD_TO_HEAD: _run_head_to_head_suite,
    VALIDATION_MODE_CROSS_PLAY: _run_cross_play_suite,
    VALIDATION_MODE_BASELINE_SANITY: _run_baseline_sanity_suite,
}


def validate_evaluation_results(
    input_path: str | Path,
    thresholds: ValidationThresholds | None = None,
    validation_mode: str = VALIDATION_MODE_TRAINING_OPPONENT,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    require_all_algorithms: bool = False,
) -> ValidationReport:
    if validation_mode not in SUITES:
        raise ValueError(
            "Unsupported validation_mode "
            f"{validation_mode!r}. Expected one of {VALIDATION_MODES}."
        )

    thresholds = thresholds or ValidationThresholds()
    path = Path(input_path)
    raw_games = pd.read_csv(path)
    try:
        manifest = EvaluationManifest.load_for_csv(path)
    except (OSError, TypeError, ValueError) as error:
        manifest = None
        manifest_error = _integrity_failure(
            check_id="manifest_load",
            message="Cannot load the evaluation summary manifest.",
            details={"error": str(error)},
        )
    else:
        manifest_error = None

    integrity_checks = validate_raw_evaluation_integrity(
        raw_games,
        validation_mode=validation_mode,
        requirements=thresholds.integrity_requirements,
        manifest=manifest,
    )
    if manifest_error is not None:
        integrity_checks.append(manifest_error)

    if any(check.blocking for check in integrity_checks):
        return ValidationReport(
            input_path=str(path),
            thresholds=thresholds,
            checks=integrity_checks,
            validation_mode=validation_mode,
            training_episode=None,
            model_selection=None,
        )

    try:
        context = _build_evaluation_context(
            input_path=path,
            validation_mode=validation_mode,
            raw_games=raw_games,
            manifest=manifest,
            algorithm_specs=algorithm_specs,
            require_all_algorithms=require_all_algorithms,
        )
    except (KeyError, TypeError, ValueError) as error:
        integrity_checks.append(
            _integrity_failure(
                check_id="aggregation_pipeline",
                message="Cannot build final validation aggregates.",
                details={"error": str(error)},
            )
        )
        return ValidationReport(
            input_path=str(path),
            thresholds=thresholds,
            checks=integrity_checks,
            validation_mode=validation_mode,
            training_episode=None,
            model_selection=None,
        )

    checks = list(integrity_checks)
    checks.extend(
        _coverage_checks(
            context,
            algorithm_specs_were_selected=algorithm_specs is not None,
            require_all_algorithms=require_all_algorithms,
        )
    )
    checks.extend(SUITE_RUNNERS[validation_mode](context, thresholds))

    return ValidationReport(
        input_path=str(path),
        thresholds=thresholds,
        checks=checks,
        validation_mode=validation_mode,
        training_episode=context.selected_training_episode,
        model_selection=context.model_selection,
    )

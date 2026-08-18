from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ALGORITHM_MONTE_CARLO,
    AlgorithmValidationSpec,
    available_algorithm_specs,
)
from src.evaluation.constants import (
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_TIGHT_AGENT,
    RULE_BASED_AGENT,
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
    _find_rows_at_latest_common_checkpoint,
    _format_float,
    _missing_common_checkpoint_result,
    _missing_row_result,
    validate_always_raise_outperforms_adaptive,
    validate_always_raise_trivial_exploit,
    validate_extreme_bb_per_100,
    validate_minimum_seed_coverage,
    validate_seed_stability,
)
from src.players.constants import (
    GENERALIZATION_OPPONENT_TO_BASE_TYPE,
    GENERALIZATION_OPPONENTS,
    OPPONENT_AGGRESSIVE_EXTREME,
)
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
)


def _algorithm_specs(
    best_rows: pd.DataFrame,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> tuple[AlgorithmValidationSpec, ...]:
    return tuple(algorithm_specs or available_algorithm_specs(best_rows))


def _existing_generalization_opponents(
    best_rows: pd.DataFrame,
    opponents: Iterable[str] = GENERALIZATION_OPPONENTS,
) -> tuple[str, ...]:
    available_opponents = set(best_rows["opponent_name"].unique())
    return tuple(
        opponent
        for opponent in opponents
        if opponent in available_opponents
    )


def _collect_agent_profit_rows(
    best_rows: pd.DataFrame,
    *,
    agent_name: str,
    opponents: Iterable[str],
) -> tuple[list[pd.Series], list[str]]:
    rows: list[pd.Series] = []
    missing_opponents: list[str] = []

    for opponent_name in opponents:
        row = _find_row(
            best_rows,
            agent_name,
            opponent_name,
        )

        if row is None:
            missing_opponents.append(opponent_name)
        else:
            rows.append(row)

    return rows, missing_opponents


def validate_generalization_adaptive_positive_variants(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = GENERALIZATION_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        check_name = (
            f"{spec.algorithm_name}: Adaptive positive on "
            "generalization variants"
        )
        rows, missing_opponents = _collect_agent_profit_rows(
            best_rows,
            agent_name=spec.adaptive_agent,
            opponents=opponents,
        )

        if not rows:
            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_SKIPPED,
                    category="generalization_profitability",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    message=(
                        "Missing adaptive rows for all generalization "
                        "variants."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "missing_opponents": missing_opponents,
                    },
                )
            )
            continue

        positive_variants = [
            str(row["opponent_name"])
            for row in rows
            if float(row["mean_profit_bb"]) >= 0.0
        ]
        non_positive_variants = [
            str(row["opponent_name"])
            for row in rows
            if float(row["mean_profit_bb"]) < 0.0
        ]
        observed = len(positive_variants)
        threshold = thresholds.min_generalization_positive_variants

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_PASS if observed >= threshold else STATUS_FAIL,
                category="generalization_profitability",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
                observed_value=float(observed),
                threshold=float(threshold),
                message=(
                    "Adaptive has non-negative mean profit on "
                    f"{observed}/{len(rows)} available variants."
                ),
                details={
                    "algorithm": spec.algorithm_name,
                    "adaptive_agent": spec.adaptive_agent,
                    "positive_variants": positive_variants,
                    "non_positive_variants": non_positive_variants,
                    "missing_opponents": missing_opponents,
                    "profits_by_variant": {
                        str(row["opponent_name"]): float(row["mean_profit_bb"])
                        for row in rows
                    },
                },
            )
        )

    return results


def validate_generalization_adaptive_beats_agent(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    *,
    baseline_agent_name: str | None = None,
    min_successful_variants: int,
    check_name: str,
    category: str,
    fail_on_underperformance: bool = True,
    opponents: Iterable[str] = GENERALIZATION_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        compared_baseline_agent = baseline_agent_name or spec.general_policy_agent
        full_check_name = f"{spec.algorithm_name}: {check_name}"
        successful_variants: list[str] = []
        failing_variants: list[str] = []
        missing_pairs: list[dict[str, str]] = []
        unaligned_variants: list[str] = []
        deltas_by_variant: dict[str, float] = {}
        checkpoint_episodes_by_variant: dict[str, int] = {}

        for opponent_name in opponents:
            matchups = (
                (spec.adaptive_agent, opponent_name),
                (compared_baseline_agent, opponent_name),
            )
            checkpoint_episode, rows = (
                _find_rows_at_latest_common_checkpoint(best_rows, matchups)
            )
            adaptive_row, baseline_row = rows

            if adaptive_row is None:
                missing_pairs.append(
                    {
                        "agent_name": spec.adaptive_agent,
                        "opponent_name": opponent_name,
                    }
                )
                continue

            if baseline_row is None:
                missing_pairs.append(
                    {
                        "agent_name": compared_baseline_agent,
                        "opponent_name": opponent_name,
                    }
                )
                continue

            if checkpoint_episode is None:
                unaligned_variants.append(opponent_name)
                continue

            delta = float(
                adaptive_row["mean_profit_bb"]
                - baseline_row["mean_profit_bb"]
            )
            deltas_by_variant[opponent_name] = delta
            checkpoint_episodes_by_variant[opponent_name] = checkpoint_episode

            if delta >= 0.0:
                successful_variants.append(opponent_name)
            else:
                failing_variants.append(opponent_name)

        compared_variants = len(deltas_by_variant)

        if compared_variants == 0:
            results.append(
                ValidationCheckResult(
                    check_name=full_check_name,
                    status=STATUS_SKIPPED,
                    category=category,
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    message=(
                        "Missing comparable adaptive/baseline rows for all "
                        "generalization variants."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "baseline_agent_name": compared_baseline_agent,
                        "missing_pairs": missing_pairs,
                        "unaligned_variants": unaligned_variants,
                    },
                )
            )
            continue

        observed = len(successful_variants)
        passed = observed >= min_successful_variants

        if passed:
            status = STATUS_PASS
        else:
            status = STATUS_FAIL if fail_on_underperformance else STATUS_WARNING

        results.append(
            ValidationCheckResult(
                check_name=full_check_name,
                status=status,
                category=category,
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
                observed_value=float(observed),
                threshold=float(min_successful_variants),
                message=(
                    f"Adaptive beats {compared_baseline_agent} on "
                    f"{observed}/{compared_variants} comparable variants."
                ),
                details={
                    "algorithm": spec.algorithm_name,
                    "adaptive_agent": spec.adaptive_agent,
                    "baseline_agent_name": compared_baseline_agent,
                    "successful_variants": successful_variants,
                    "failing_variants": failing_variants,
                    "missing_pairs": missing_pairs,
                    "unaligned_variants": unaligned_variants,
                    "deltas_by_variant": deltas_by_variant,
                    "checkpoint_episodes_by_variant": (
                        checkpoint_episodes_by_variant
                    ),
                },
            )
        )

    return results


def validate_generalization_oracle_gap(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = GENERALIZATION_OPPONENTS,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        for opponent_name in opponents:
            matchups = (
                (spec.adaptive_agent, opponent_name),
                (spec.oracle_agent, opponent_name),
            )
            checkpoint_episode, rows = (
                _find_rows_at_latest_common_checkpoint(best_rows, matchups)
            )
            adaptive_row, oracle_row = rows
            check_name = (
                f"{spec.algorithm_name}: Generalization oracle gap "
                f"vs {opponent_name}"
            )

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "generalization_oracle_gap",
                        spec.adaptive_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            if oracle_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "generalization_oracle_gap",
                        spec.oracle_agent,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            if checkpoint_episode is None:
                results.append(
                    _missing_common_checkpoint_result(
                        check_name,
                        "generalization_oracle_gap",
                        best_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            oracle_gap = float(
                oracle_row["mean_profit_bb"]
                - adaptive_row["mean_profit_bb"]
            )
            large_gap = oracle_gap > thresholds.max_generalization_oracle_gap_bb

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_WARNING if large_gap else STATUS_PASS,
                    category="generalization_oracle_gap",
                    algorithm_name=spec.algorithm_name,
                    agent_name=spec.adaptive_agent,
                    opponent_name=opponent_name,
                    checkpoint_episode=checkpoint_episode,
                    observed_value=oracle_gap,
                    threshold=thresholds.max_generalization_oracle_gap_bb,
                    message=(
                        "Oracle minus adaptive mean profit is "
                        f"{_format_float(oracle_gap)} BB/game."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "oracle_agent": spec.oracle_agent,
                        "adaptive_mean_profit_bb": float(
                            adaptive_row["mean_profit_bb"]
                        ),
                        "oracle_mean_profit_bb": float(
                            oracle_row["mean_profit_bb"]
                        ),
                        "max_oracle_gap_bb": thresholds.max_generalization_oracle_gap_bb,
                    },
                )
            )

    return results


def validate_generalization_classifier_quality(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = GENERALIZATION_OPPONENTS,
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
                    f"{spec.algorithm_name}: Generalization classifier "
                    f"{pretty_metric} vs {opponent_name}"
                )

                if adaptive_row is None:
                    results.append(
                        _missing_row_result(
                            check_name,
                            "generalization_classifier_quality",
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
                        category="generalization_classifier_quality",
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                        checkpoint_episode=_checkpoint_episode(adaptive_row),
                        observed_value=value,
                        threshold=threshold,
                        message=(
                            "Adaptive classifier "
                            f"{pretty_metric} on unseen variant is "
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


def validate_generalization_aggressive_extreme_robustness(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for spec in _algorithm_specs(best_rows, algorithm_specs):
        check_name = (
            f"{spec.algorithm_name}: Aggressive extreme robustness check"
        )
        row = _find_row(
            best_rows,
            spec.adaptive_agent,
            OPPONENT_AGGRESSIVE_EXTREME,
        )

        if row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "generalization_extreme_robustness",
                    spec.adaptive_agent,
                    OPPONENT_AGGRESSIVE_EXTREME,
                    spec.algorithm_name,
                )
            )
            continue

        mean_profit_bb = float(row["mean_profit_bb"])
        bust_rate = float(row["bust_rate"])
        robustness_warning = (
            mean_profit_bb
            < thresholds.generalization_extreme_aggressive_min_profit_bb
            or bust_rate
            > thresholds.generalization_extreme_aggressive_max_bust_rate
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=STATUS_WARNING if robustness_warning else STATUS_PASS,
                category="generalization_extreme_robustness",
                algorithm_name=spec.algorithm_name,
                agent_name=spec.adaptive_agent,
                opponent_name=OPPONENT_AGGRESSIVE_EXTREME,
                checkpoint_episode=_checkpoint_episode(row),
                observed_value=mean_profit_bb,
                threshold=thresholds.generalization_extreme_aggressive_min_profit_bb,
                message=(
                    "Adaptive vs aggressive_extreme: "
                    f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                    f"bust_rate={_format_float(bust_rate)}%."
                ),
                details={
                    "algorithm": spec.algorithm_name,
                    "adaptive_agent": spec.adaptive_agent,
                    "mean_profit_bb": mean_profit_bb,
                    "bust_rate": bust_rate,
                    "min_profit_threshold_bb": (
                        thresholds.generalization_extreme_aggressive_min_profit_bb
                    ),
                    "max_bust_rate_threshold": (
                        thresholds.generalization_extreme_aggressive_max_bust_rate
                    ),
                },
            )
        )

    return results


def validate_generalization_matching_specialists(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []
    base_type_to_specialist = {
        OPPONENT_TYPE_CALLING: POLICY_CALLING_AGENT,
        OPPONENT_TYPE_AGGRESSIVE: POLICY_AGGRESSIVE_AGENT,
        OPPONENT_TYPE_TIGHT: POLICY_TIGHT_AGENT,
    }
    variant_to_specialist = {
        opponent_name: base_type_to_specialist[base_type]
        for opponent_name, base_type in GENERALIZATION_OPPONENT_TO_BASE_TYPE.items()
    }

    for opponent_name, specialist_agent in variant_to_specialist.items():
        row = _find_row(
            best_rows,
            specialist_agent,
            opponent_name,
        )
        check_name = (
            "Monte Carlo: Matching specialist transfer "
            f"for {specialist_agent} vs {opponent_name}"
        )

        if row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "generalization_specialist_transfer",
                    specialist_agent,
                    opponent_name,
                    ALGORITHM_MONTE_CARLO,
                )
            )
            continue

        mean_profit_bb = float(row["mean_profit_bb"])
        status = (
            STATUS_PASS
            if mean_profit_bb >= thresholds.min_head_to_head_mean_profit_bb
            else STATUS_WARNING
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=status,
                category="generalization_specialist_transfer",
                algorithm_name=ALGORITHM_MONTE_CARLO,
                agent_name=specialist_agent,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(row),
                observed_value=mean_profit_bb,
                threshold=thresholds.min_head_to_head_mean_profit_bb,
                message=(
                    f"{specialist_agent} vs {opponent_name}: "
                    f"mean_profit_bb={_format_float(mean_profit_bb)}."
                ),
                details={
                    "algorithm": ALGORITHM_MONTE_CARLO,
                    "mean_profit_bb": mean_profit_bb,
                    "expected_family_specialist": True,
                    "note": (
                        "Only Monte Carlo fixed specialists are public "
                        "tested agents. TD specialists are used through "
                        "adaptive/oracle agents."
                    ),
                },
            )
        )

    return results


def validate_generalization_results_from_best_rows(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    comparison_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    opponents = _existing_generalization_opponents(best_rows)
    specs = tuple(algorithm_specs or available_algorithm_specs(best_rows))
    aligned_comparison_rows = (
        best_rows if comparison_rows is None else comparison_rows
    )

    checks: list[ValidationCheckResult] = []
    checks.extend(
        validate_generalization_adaptive_positive_variants(
            best_rows,
            thresholds,
            opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_generalization_adaptive_beats_agent(
            aligned_comparison_rows,
            thresholds,
            min_successful_variants=(
                thresholds.min_generalization_adaptive_beats_general_variants
            ),
            check_name=(
                "Adaptive beats fixed general on generalization variants"
            ),
            category="generalization_adaptive_delta_vs_general",
            fail_on_underperformance=True,
            opponents=opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_generalization_adaptive_beats_agent(
            aligned_comparison_rows,
            thresholds,
            baseline_agent_name=RULE_BASED_AGENT,
            min_successful_variants=(
                thresholds.min_generalization_adaptive_beats_rule_based_variants
            ),
            check_name=(
                "Adaptive beats rule-based on generalization variants"
            ),
            category="generalization_adaptive_delta_vs_rule_based",
            fail_on_underperformance=False,
            opponents=opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_generalization_oracle_gap(
            aligned_comparison_rows,
            thresholds,
            opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_generalization_classifier_quality(
            best_rows,
            thresholds,
            opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_generalization_matching_specialists(
            best_rows,
            thresholds,
        )
    )
    checks.extend(
        validate_always_raise_outperforms_adaptive(
            aligned_comparison_rows,
            thresholds,
            opponents,
            algorithm_specs=specs,
        )
    )
    checks.extend(
        validate_always_raise_trivial_exploit(
            best_rows,
            thresholds,
            opponents,
        )
    )
    checks.extend(
        validate_generalization_aggressive_extreme_robustness(
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

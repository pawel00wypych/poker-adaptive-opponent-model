from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import cast

import pandas as pd

from src.evaluation.algorithm_metadata import (
    ADAPTIVE_AGENTS,
    GENERAL_POLICY_AGENTS,
    ORACLE_ALGORITHM_AGENTS,
    AlgorithmValidationSpec,
    algorithm_name_for_agent,
    available_algorithm_specs,
)
from src.evaluation.constants import (
    ALWAYS_RAISE_AGENT,
    FIXED_SPECIALIST_AGENTS,
    RULE_BASED_AGENT,
)
from src.evaluation.metrics.paired_seed_statistics import (
    PAIRED_SEED_OPERATION_DIFFERENCE,
    PairedSeedStatistics,
    PairedSeedStatisticsError,
    calculate_paired_seed_statistics,
)
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    TRAINING_OPPONENT_TYPES,
)

STATUS_PASS = "PASS"

STATUS_WARNING = "WARNING"

STATUS_FAIL = "FAIL"

STATUS_SKIPPED = "SKIPPED"

VALIDATION_STATUSES = (
    STATUS_PASS,
    STATUS_WARNING,
    STATUS_FAIL,
    STATUS_SKIPPED,
)

VALIDATION_MODE_TRAINING_OPPONENT = "training-opponent"

VALIDATION_MODE_HEAD_TO_HEAD = "head-to-head"

VALIDATION_MODE_GENERALIZATION = "generalization"

VALIDATION_MODE_STRESS_TEST = "stress-test"

VALIDATION_MODE_BASELINE_SANITY = "baseline-sanity"

VALIDATION_MODE_CROSS_PLAY = "cross-play"

VALIDATION_MODES = (
    VALIDATION_MODE_TRAINING_OPPONENT,
    VALIDATION_MODE_HEAD_TO_HEAD,
    VALIDATION_MODE_GENERALIZATION,
    VALIDATION_MODE_STRESS_TEST,
    VALIDATION_MODE_BASELINE_SANITY,
    VALIDATION_MODE_CROSS_PLAY,
)

DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS = (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)

DEFAULT_CLASSIFIER_OPPONENTS = TRAINING_OPPONENT_TYPES

DEFAULT_ORACLE_OPPONENTS = TRAINING_OPPONENT_TYPES

HEAD_TO_HEAD_RULE_BASED_OPPONENT = RULE_BASED_AGENT

HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT = ALWAYS_RAISE_AGENT

HEAD_TO_HEAD_SPECIALIST_AGENTS = FIXED_SPECIALIST_AGENTS

HEAD_TO_HEAD_LEARNED_AGENTS = (
    *GENERAL_POLICY_AGENTS,
    *ADAPTIVE_AGENTS,
    *FIXED_SPECIALIST_AGENTS,
)

GENERALIZATION_CORE_AGENTS = (
    *ADAPTIVE_AGENTS,
    *ORACLE_ALGORITHM_AGENTS,
    *GENERAL_POLICY_AGENTS,
    RULE_BASED_AGENT,
    ALWAYS_RAISE_AGENT,
)

GENERALIZATION_SPECIALIST_AGENTS = FIXED_SPECIALIST_AGENTS


def _missing_values_as_none(value: object) -> object:
    if isinstance(value, dict):
        return {key: _missing_values_as_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_missing_values_as_none(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_missing_values_as_none(item) for item in value)
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return None
    return value


@dataclass(frozen=True)
class ValidationThresholds:
    min_adaptive_delta_vs_rule_based_bb: float = 0.0
    max_oracle_underperformance_bb: float = 1.0
    min_tight_win_rate: float = 95.0
    min_tight_mean_profit_bb: float = 15.0
    min_classifier_accuracy: float = 80.0
    min_classifier_coverage: float = 80.0
    max_std_across_seeds_bb: float = 5.0
    extreme_bb_per_100_threshold: float = 300.0
    low_mean_hands_played_threshold: float = 5.0
    always_raise_adaptive_warning_gap_bb: float = 3.0
    high_always_raise_mean_profit_bb: float = 18.0
    high_always_raise_win_rate: float = 95.0
    min_head_to_head_mean_profit_bb: float = 0.0
    max_adaptive_underperformance_vs_general_bb: float = 1.0
    always_raise_stress_loss_bb: float = -15.0
    always_raise_stress_bust_rate: float = 80.0
    min_generalization_positive_variants: int = 3
    min_generalization_adaptive_beats_general_variants: int = 3
    min_generalization_adaptive_beats_rule_based_variants: int = 3
    max_generalization_oracle_gap_bb: float = 3.0
    generalization_extreme_aggressive_min_profit_bb: float = -5.0
    generalization_extreme_aggressive_max_bust_rate: float = 85.0
    min_seeds_per_matchup: int = 2
    min_evaluation_replicates_per_matchup: int = 2
    max_std_across_evaluation_replicates_bb: float = 5.0
    max_baseline_mirror_abs_profit_bb: float = 1.0
    max_baseline_pair_sum_abs_profit_bb: float = 2.0
    max_cross_play_pair_sum_abs_profit_bb: float = 2.0


@dataclass(frozen=True)
class ValidationCheckResult:
    check_name: str
    status: str
    message: str
    category: str
    algorithm_name: str | None = None
    agent_name: str | None = None
    opponent_name: str | None = None
    training_episode: int | None = None
    observed_value: float | None = None
    threshold: float | None = None
    sample_size: int | None = None
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            _missing_values_as_none(asdict(self)),
        )


@dataclass(frozen=True)
class ValidationReport:
    input_path: str
    thresholds: ValidationThresholds
    checks: list[ValidationCheckResult]
    validation_mode: str = VALIDATION_MODE_TRAINING_OPPONENT
    training_episode: int | None = None
    model_selection: str | None = None

    @property
    def passed(self) -> bool:
        return not any(check.status == STATUS_FAIL for check in self.checks)

    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in VALIDATION_STATUSES
        }

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            _missing_values_as_none(
                {
                    "input_path": self.input_path,
                    "validation_mode": self.validation_mode,
                    "training_episode": self.training_episode,
                    "model_selection": self.model_selection,
                    "passed": self.passed,
                    "status_counts": self.status_counts(),
                    "thresholds": asdict(self.thresholds),
                    "checks": [check.to_dict() for check in self.checks],
                }
            ),
        )


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _paired_seed_statistics_for_check(
    seed_rows: pd.DataFrame | None,
    *,
    left_agent_name: str,
    right_agent_name: str,
    opponent_name: str,
    right_opponent_name: str | None = None,
    operation: str = PAIRED_SEED_OPERATION_DIFFERENCE,
    training_episode: int,
    thresholds: ValidationThresholds,
    check_name: str,
    category: str,
    algorithm_name: str | None,
    agent_name: str | None,
) -> tuple[PairedSeedStatistics | None, ValidationCheckResult | None]:
    """Return paired statistics or a validation result explaining no result.

    ``None`` seed rows retain compatibility for callers that only have legacy
    aggregate rows. The main validation pipeline always supplies seed rows.
    """

    if seed_rows is None:
        return None, None

    try:
        statistics = calculate_paired_seed_statistics(
            seed_rows,
            left_agent_name=left_agent_name,
            right_agent_name=right_agent_name,
            opponent_name=opponent_name,
            right_opponent_name=right_opponent_name,
            training_episode=training_episode,
            operation=operation,
        )
    except PairedSeedStatisticsError as error:
        return None, ValidationCheckResult(
            check_name=check_name,
            status=STATUS_FAIL,
            category=category,
            algorithm_name=algorithm_name,
            agent_name=agent_name,
            opponent_name=opponent_name,
            training_episode=training_episode,
            message=f"Cannot calculate paired seed statistics: {error}",
            details={"paired_seed_error": str(error)},
        )

    details = {"paired_seed_statistics": statistics.to_details()}
    if statistics.left_seed_count == 0 or statistics.right_seed_count == 0:
        if statistics.left_seed_count == 0:
            missing_agent = left_agent_name
            missing_opponent = opponent_name
        else:
            missing_agent = right_agent_name
            missing_opponent = statistics.right_opponent_name
        return None, ValidationCheckResult(
            check_name=check_name,
            status=STATUS_SKIPPED,
            category=category,
            algorithm_name=algorithm_name,
            agent_name=agent_name,
            opponent_name=opponent_name,
            training_episode=training_episode,
            sample_size=statistics.common_seed_count,
            message=(
                f"Missing seed-level rows for {missing_agent} vs {missing_opponent}."
            ),
            details=details,
        )

    minimum_common_seeds = max(2, thresholds.min_seeds_per_matchup)
    if statistics.common_seed_count < minimum_common_seeds:
        details["minimum_common_seeds"] = minimum_common_seeds
        return None, ValidationCheckResult(
            check_name=check_name,
            status=STATUS_FAIL,
            category=category,
            algorithm_name=algorithm_name,
            agent_name=agent_name,
            opponent_name=opponent_name,
            training_episode=training_episode,
            observed_value=statistics.mean_delta,
            sample_size=statistics.common_seed_count,
            standard_error=statistics.standard_error,
            ci_lower=statistics.ci_lower,
            ci_upper=statistics.ci_upper,
            message=(
                "Paired comparison has "
                f"{statistics.common_seed_count} common model seed(s); "
                f"at least {minimum_common_seeds} are required."
            ),
            details=details,
        )

    return statistics, None


def _minimum_delta_status(
    statistics: PairedSeedStatistics,
    threshold: float,
    *,
    underperformance_status: str = STATUS_FAIL,
) -> str:
    if statistics.ci_lower is None or statistics.ci_upper is None:
        raise ValueError("Paired confidence interval is unavailable.")
    if statistics.ci_lower >= threshold:
        return STATUS_PASS
    if statistics.ci_upper < threshold:
        return underperformance_status
    return STATUS_WARNING


def _maximum_delta_status(
    statistics: PairedSeedStatistics,
    threshold: float,
    *,
    exceedance_status: str = STATUS_WARNING,
) -> str:
    if statistics.ci_lower is None or statistics.ci_upper is None:
        raise ValueError("Paired confidence interval is unavailable.")
    if statistics.ci_upper <= threshold:
        return STATUS_PASS
    return exceedance_status


def _paired_seed_message(
    label: str,
    statistics: PairedSeedStatistics,
) -> str:
    return (
        f"{label} is {_format_float(statistics.mean_delta)} BB/game "
        f"across {statistics.common_seed_count} paired seed(s); 95% t-CI "
        f"[{_format_float(statistics.ci_lower)}, "
        f"{_format_float(statistics.ci_upper)}]."
    )


def _add_mean_hands_played(
    aggregated: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if aggregated.empty or "mean_hands_played" in aggregated.columns:
        return aggregated

    required_columns = {
        "training_run",
        "agent_name",
        "opponent_name",
        "training_episode",
        "total_hands",
        "games",
    }

    if not required_columns.issubset(metrics.columns):
        aggregated = aggregated.copy()
        aggregated["mean_hands_played"] = 0.0
        return aggregated

    working = metrics.copy()
    working["mean_hands_played"] = working["total_hands"] / working["games"]

    hand_means = (
        working.groupby(
            [
                "training_run",
                "agent_name",
                "opponent_name",
                "training_episode",
            ]
        )["mean_hands_played"]
        .mean()
        .reset_index()
    )

    return aggregated.merge(
        hand_means,
        on=[
            "training_run",
            "agent_name",
            "opponent_name",
            "training_episode",
        ],
        how="left",
    )


ROW_IDENTITY_COLUMNS = (
    "training_run",
    "training_episode",
    "model_seed",
    "model_source",
    "experiment_name",
)


class AmbiguousValidationRowError(ValueError):
    """Raised when validation data does not identify exactly one row.

    Validators must never choose between candidate rows: picking the
    best-scoring one would select a result using the evaluation data itself.
    """


def _describe_ambiguity(matching: pd.DataFrame) -> str:
    differing = []

    for column in ROW_IDENTITY_COLUMNS:
        if column not in matching.columns:
            continue

        values = sorted(
            {str(value) for value in matching[column].dropna().tolist()}
        )
        if len(values) > 1:
            differing.append(f"{column}={values}")

    if not differing:
        return "The duplicate rows share the same identifying columns."

    return "Rows differ by " + ", ".join(differing) + "."


def _find_row(
    df: pd.DataFrame,
    agent_name: str,
    opponent_name: str,
    training_episode: int | None = None,
) -> pd.Series | None:
    """Return the single aggregated row for one matchup, or None if absent.

    Ambiguity is an error rather than something to resolve by ranking: the
    validators must not pick a row based on how well it scored.
    """
    matching = df[
        (df["agent_name"] == agent_name) & (df["opponent_name"] == opponent_name)
    ]

    if training_episode is not None:
        matching = matching[matching["training_episode"] == training_episode]

    if matching.empty:
        return None

    if len(matching) > 1:
        raise AmbiguousValidationRowError(
            f"Expected exactly one aggregated row for {agent_name!r} vs "
            f"{opponent_name!r}, found {len(matching)}. "
            f"{_describe_ambiguity(matching)} "
            "Validate a single training run and training episode at a time."
        )

    return matching.iloc[0]


def _find_rows_at_common_training_episode(
    df: pd.DataFrame,
    matchups: Iterable[tuple[str, str]],
) -> tuple[int | None, tuple[pd.Series | None, ...]]:
    matchup_list = tuple(matchups)
    available_rows: list[pd.Series | None] = []
    training_episode_sets: list[set[int]] = []

    for agent_name, opponent_name in matchup_list:
        matching = df[
            (df["agent_name"] == agent_name) & (df["opponent_name"] == opponent_name)
        ]
        if matching.empty:
            available_rows.append(None)
            continue

        available_rows.append(matching.iloc[0])
        training_episodes = {
            int(training_episode)
            for training_episode in matching["training_episode"].dropna()
        }
        training_episode_sets.append(training_episodes)

    if any(row is None for row in available_rows):
        return None, tuple(available_rows)

    if not training_episode_sets or any(
        not training_episodes for training_episodes in training_episode_sets
    ):
        return None, tuple(available_rows)

    common_training_episodes = set.intersection(*training_episode_sets)
    if not common_training_episodes:
        return None, tuple(available_rows)

    training_episode = max(common_training_episodes)
    aligned_rows = tuple(
        _find_row(
            df,
            agent_name,
            opponent_name,
            training_episode=training_episode,
        )
        for agent_name, opponent_name in matchup_list
    )
    return training_episode, aligned_rows


def _training_episode(row: pd.Series | None) -> int | None:
    if row is None or "training_episode" not in row:
        return None
    return int(row["training_episode"])


def _missing_row_result(
    check_name: str,
    category: str,
    agent_name: str,
    opponent_name: str,
    algorithm_name: str | None = None,
) -> ValidationCheckResult:
    return ValidationCheckResult(
        check_name=check_name,
        status=STATUS_SKIPPED,
        category=category,
        algorithm_name=algorithm_name or algorithm_name_for_agent(agent_name),
        agent_name=agent_name,
        opponent_name=opponent_name,
        message=(f"Missing row for {agent_name} vs {opponent_name}."),
    )


def _missing_common_training_episode_result(
    check_name: str,
    category: str,
    df: pd.DataFrame,
    matchups: Iterable[tuple[str, str]],
    *,
    algorithm_name: str | None = None,
    agent_name: str | None = None,
    opponent_name: str | None = None,
) -> ValidationCheckResult:
    matchup_list = tuple(matchups)
    training_episodes_by_matchup = {
        f"{matchup_agent} vs {matchup_opponent}": sorted(
            {
                int(training_episode)
                for training_episode in df.loc[
                    (df["agent_name"] == matchup_agent)
                    & (df["opponent_name"] == matchup_opponent),
                    "training_episode",
                ].dropna()
            }
        )
        for matchup_agent, matchup_opponent in matchup_list
    }
    return ValidationCheckResult(
        check_name=check_name,
        status=STATUS_SKIPPED,
        category=category,
        algorithm_name=algorithm_name,
        agent_name=agent_name,
        opponent_name=opponent_name,
        message=(
            "No common training_episode is available for all rows in this comparison."
        ),
        details={
            "required_matchups": [
                {
                    "agent_name": matchup_agent,
                    "opponent_name": matchup_opponent,
                }
                for matchup_agent, matchup_opponent in matchup_list
            ],
            "training_episodes_by_matchup": training_episodes_by_matchup,
        },
    )


def validate_minimum_seed_coverage(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    minimum_seeds = thresholds.min_seeds_per_matchup
    if minimum_seeds < 1:
        raise ValueError("min_seeds_per_matchup must be at least 1")

    results: list[ValidationCheckResult] = []

    for _, row in final_rows.iterrows():
        raw_seed_count = row.get("seeds", 0)
        seed_count = 0 if pd.isna(raw_seed_count) else int(raw_seed_count)
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        sufficient = seed_count >= minimum_seeds

        results.append(
            ValidationCheckResult(
                check_name=(
                    f"Minimum seed coverage for {agent_name} vs {opponent_name}"
                ),
                status=STATUS_PASS if sufficient else STATUS_FAIL,
                category="seed_coverage",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                training_episode=_training_episode(row),
                observed_value=float(seed_count),
                threshold=float(minimum_seeds),
                message=(
                    f"Evaluation includes {seed_count} distinct model "
                    f"seed(s); minimum required is {minimum_seeds}."
                ),
                details={
                    "seed_count": seed_count,
                    "min_seeds_per_matchup": minimum_seeds,
                    "missing_seed_count": max(minimum_seeds - seed_count, 0),
                },
            )
        )

    return results


def validate_seed_stability(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for _, row in final_rows.iterrows():
        value = float(row["mean_profit_bb_std_across_seeds"])
        status = (
            STATUS_PASS
            if value <= thresholds.max_std_across_seeds_bb
            else STATUS_WARNING
        )
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])

        results.append(
            ValidationCheckResult(
                check_name=(f"Seed stability for {agent_name} vs {opponent_name}"),
                status=status,
                category="seed_stability",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                training_episode=_training_episode(row),
                observed_value=value,
                threshold=thresholds.max_std_across_seeds_bb,
                message=(
                    f"Mean profit std across seeds is {_format_float(value)} BB/game."
                ),
            )
        )

    return results


def validate_extreme_bb_per_100(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for _, row in final_rows.iterrows():
        bb_per_100 = float(row["bb_per_100"])
        mean_hands_played = float(row.get("mean_hands_played", 0.0))
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        is_extreme = (
            abs(bb_per_100) > thresholds.extreme_bb_per_100_threshold
            and mean_hands_played < thresholds.low_mean_hands_played_threshold
        )

        results.append(
            ValidationCheckResult(
                check_name=(
                    f"Extreme BB/100 sanity check for {agent_name} vs {opponent_name}"
                ),
                status=STATUS_WARNING if is_extreme else STATUS_PASS,
                category="bb_per_100_sanity",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                training_episode=_training_episode(row),
                observed_value=bb_per_100,
                threshold=thresholds.extreme_bb_per_100_threshold,
                message=(
                    "bb_per_100="
                    f"{_format_float(bb_per_100)}, "
                    "mean_hands_played="
                    f"{_format_float(mean_hands_played)}."
                ),
                details={
                    "bb_per_100": bb_per_100,
                    "mean_hands_played": mean_hands_played,
                    "bb_per_100_threshold": (thresholds.extreme_bb_per_100_threshold),
                    "low_mean_hands_played_threshold": (
                        thresholds.low_mean_hands_played_threshold
                    ),
                },
            )
        )

    return results


def validate_always_raise_outperforms_adaptive(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = TRAINING_OPPONENT_TYPES,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
    seed_rows: pd.DataFrame | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []
    specs = tuple(algorithm_specs or available_algorithm_specs(final_rows))

    for spec in specs:
        for opponent_name in opponents:
            matchups = (
                (ALWAYS_RAISE_AGENT, opponent_name),
                (spec.adaptive_agent, opponent_name),
            )
            training_episode, rows = _find_rows_at_common_training_episode(
                final_rows, matchups
            )
            always_raise_row, adaptive_row = rows
            check_name = (
                f"{spec.algorithm_name}: Always-raise dominance "
                f"sanity check vs {opponent_name}"
            )

            if always_raise_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "always_raise_sanity",
                        ALWAYS_RAISE_AGENT,
                        opponent_name,
                        spec.algorithm_name,
                    )
                )
                continue

            if adaptive_row is None:
                results.append(
                    _missing_row_result(
                        check_name,
                        "always_raise_sanity",
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
                        "always_raise_sanity",
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
                left_agent_name=ALWAYS_RAISE_AGENT,
                right_agent_name=spec.adaptive_agent,
                opponent_name=opponent_name,
                training_episode=training_episode,
                thresholds=thresholds,
                check_name=check_name,
                category="always_raise_sanity",
                algorithm_name=spec.algorithm_name,
                agent_name=ALWAYS_RAISE_AGENT,
            )
            if unavailable_result is not None:
                results.append(unavailable_result)
                continue

            if paired_statistics is None:
                delta = float(
                    always_raise_row["mean_profit_bb"] - adaptive_row["mean_profit_bb"]
                )
                is_large_gap = delta >= thresholds.always_raise_adaptive_warning_gap_bb
                status = STATUS_WARNING if is_large_gap else STATUS_PASS
                message = (
                    "Always-raise minus adaptive mean profit is "
                    f"{_format_float(delta)} BB/game."
                )
            else:
                delta = float(paired_statistics.mean_delta)
                status = _maximum_delta_status(
                    paired_statistics,
                    thresholds.always_raise_adaptive_warning_gap_bb,
                )
                message = _paired_seed_message(
                    "Always-raise minus adaptive mean profit",
                    paired_statistics,
                )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=status,
                    category="always_raise_sanity",
                    algorithm_name=spec.algorithm_name,
                    agent_name=ALWAYS_RAISE_AGENT,
                    opponent_name=opponent_name,
                    training_episode=training_episode,
                    observed_value=delta,
                    threshold=(thresholds.always_raise_adaptive_warning_gap_bb),
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
                        "always_raise_mean_profit_bb": float(
                            always_raise_row["mean_profit_bb"]
                        ),
                        "adaptive_mean_profit_bb": float(
                            adaptive_row["mean_profit_bb"]
                        ),
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


def validate_always_raise_trivial_exploit(
    final_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = TRAINING_OPPONENT_TYPES,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        always_raise_row = _find_row(
            final_rows,
            ALWAYS_RAISE_AGENT,
            opponent_name,
        )
        check_name = f"Always-raise trivial exploit sanity check vs {opponent_name}"

        if always_raise_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "always_raise_sanity",
                    ALWAYS_RAISE_AGENT,
                    opponent_name,
                )
            )
            continue

        mean_profit_bb = float(always_raise_row["mean_profit_bb"])
        win_rate = float(always_raise_row["win_rate"])
        is_trivial_exploit = (
            mean_profit_bb >= thresholds.high_always_raise_mean_profit_bb
            and win_rate >= thresholds.high_always_raise_win_rate
        )

        results.append(
            ValidationCheckResult(
                check_name=check_name,
                status=(STATUS_WARNING if is_trivial_exploit else STATUS_PASS),
                category="always_raise_sanity",
                agent_name=ALWAYS_RAISE_AGENT,
                opponent_name=opponent_name,
                training_episode=_training_episode(always_raise_row),
                observed_value=mean_profit_bb,
                threshold=thresholds.high_always_raise_mean_profit_bb,
                message=(
                    "Always-raise vs opponent: "
                    f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                    f"win_rate={_format_float(win_rate)}%."
                ),
                details={
                    "mean_profit_bb": mean_profit_bb,
                    "win_rate": win_rate,
                    "high_mean_profit_bb_threshold": (
                        thresholds.high_always_raise_mean_profit_bb
                    ),
                    "high_win_rate_threshold": (thresholds.high_always_raise_win_rate),
                },
            )
        )

    return results

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

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
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_TIGHT_AGENT,
    RULE_BASED_AGENT,
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

VALIDATION_MODE_CHECKPOINT = "checkpoint"

VALIDATION_MODE_HEAD_TO_HEAD = "head-to-head"

VALIDATION_MODE_GENERALIZATION = "generalization"

VALIDATION_MODES = (
    VALIDATION_MODE_CHECKPOINT,
    VALIDATION_MODE_HEAD_TO_HEAD,
    VALIDATION_MODE_GENERALIZATION,
)

DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS = (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)

DEFAULT_CLASSIFIER_OPPONENTS = TRAINING_OPPONENT_TYPES

DEFAULT_ORACLE_OPPONENTS = TRAINING_OPPONENT_TYPES

HEAD_TO_HEAD_RULE_BASED_OPPONENT = RULE_BASED_AGENT

HEAD_TO_HEAD_ALWAYS_RAISE_OPPONENT = ALWAYS_RAISE_AGENT

HEAD_TO_HEAD_SPECIALIST_AGENTS = (
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
)

HEAD_TO_HEAD_LEARNED_AGENTS = (
    *GENERAL_POLICY_AGENTS,
    *ADAPTIVE_AGENTS,
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
)

GENERALIZATION_CORE_AGENTS = (
    *ADAPTIVE_AGENTS,
    *ORACLE_ALGORITHM_AGENTS,
    *GENERAL_POLICY_AGENTS,
    RULE_BASED_AGENT,
    ALWAYS_RAISE_AGENT,
)

GENERALIZATION_SPECIALIST_AGENTS = (
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
)


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

@dataclass(frozen=True)
class ValidationCheckResult:
    check_name: str
    status: str
    message: str
    category: str
    algorithm_name: str | None = None
    agent_name: str | None = None
    opponent_name: str | None = None
    checkpoint_episode: int | None = None
    observed_value: float | None = None
    threshold: float | None = None
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class ValidationReport:
    input_path: str
    thresholds: ValidationThresholds
    checks: list[ValidationCheckResult]
    validation_mode: str = VALIDATION_MODE_CHECKPOINT

    @property
    def passed(self) -> bool:
        return not any(
            check.status == STATUS_FAIL
            for check in self.checks
        )

    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(
                check.status == status
                for check in self.checks
            )
            for status in VALIDATION_STATUSES
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "validation_mode": self.validation_mode,
            "passed": self.passed,
            "status_counts": self.status_counts(),
            "thresholds": asdict(self.thresholds),
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }

def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"

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
        "checkpoint_episode",
        "total_hands",
        "games",
    }

    if not required_columns.issubset(metrics.columns):
        aggregated = aggregated.copy()
        aggregated["mean_hands_played"] = 0.0
        return aggregated

    working = metrics.copy()
    working["mean_hands_played"] = (
        working["total_hands"] / working["games"]
    )

    hand_means = (
        working.groupby(
            [
                "training_run",
                "agent_name",
                "opponent_name",
                "checkpoint_episode",
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
            "checkpoint_episode",
        ],
        how="left",
    )

def _best_rows_by_agent_and_opponent(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated.copy()

    indexes = aggregated.groupby(
        [
            "agent_name",
            "opponent_name",
        ]
    )["mean_profit_bb"].idxmax()

    return aggregated.loc[indexes].reset_index(drop=True)

def _find_row(
    df: pd.DataFrame,
    agent_name: str,
    opponent_name: str,
    checkpoint_episode: int | None = None,
) -> pd.Series | None:
    matching = df[
        (df["agent_name"] == agent_name)
        & (df["opponent_name"] == opponent_name)
    ]

    if checkpoint_episode is not None:
        matching = matching[
            matching["checkpoint_episode"] == checkpoint_episode
        ]

    if matching.empty:
        return None

    if "mean_profit_bb" in matching.columns:
        return matching.loc[matching["mean_profit_bb"].idxmax()]

    return matching.iloc[0]


def _find_rows_at_latest_common_checkpoint(
    df: pd.DataFrame,
    matchups: Iterable[tuple[str, str]],
) -> tuple[int | None, tuple[pd.Series | None, ...]]:
    matchup_list = tuple(matchups)
    available_rows: list[pd.Series | None] = []
    checkpoint_sets: list[set[int]] = []

    for agent_name, opponent_name in matchup_list:
        matching = df[
            (df["agent_name"] == agent_name)
            & (df["opponent_name"] == opponent_name)
        ]
        if matching.empty:
            available_rows.append(None)
            continue

        available_rows.append(matching.iloc[0])
        checkpoints = {
            int(checkpoint)
            for checkpoint in matching["checkpoint_episode"].dropna()
        }
        checkpoint_sets.append(checkpoints)

    if any(row is None for row in available_rows):
        return None, tuple(available_rows)

    if not checkpoint_sets or any(not checkpoints for checkpoints in checkpoint_sets):
        return None, tuple(available_rows)

    common_checkpoints = set.intersection(*checkpoint_sets)
    if not common_checkpoints:
        return None, tuple(available_rows)

    checkpoint_episode = max(common_checkpoints)
    aligned_rows = tuple(
        _find_row(
            df,
            agent_name,
            opponent_name,
            checkpoint_episode=checkpoint_episode,
        )
        for agent_name, opponent_name in matchup_list
    )
    return checkpoint_episode, aligned_rows

def _checkpoint_episode(row: pd.Series | None) -> int | None:
    if row is None or "checkpoint_episode" not in row:
        return None
    return int(row["checkpoint_episode"])

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
        message=(
            f"Missing row for {agent_name} vs {opponent_name}."
        ),
    )


def _missing_common_checkpoint_result(
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
    checkpoints_by_matchup = {
        f"{matchup_agent} vs {matchup_opponent}": sorted(
            {
                int(checkpoint)
                for checkpoint in df.loc[
                    (df["agent_name"] == matchup_agent)
                    & (df["opponent_name"] == matchup_opponent),
                    "checkpoint_episode",
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
            "No common checkpoint_episode is available for all rows in "
            "this comparison."
        ),
        details={
            "required_matchups": [
                {
                    "agent_name": matchup_agent,
                    "opponent_name": matchup_opponent,
                }
                for matchup_agent, matchup_opponent in matchup_list
            ],
            "checkpoints_by_matchup": checkpoints_by_matchup,
        },
    )


def validate_minimum_seed_coverage(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    minimum_seeds = thresholds.min_seeds_per_matchup
    if minimum_seeds < 1:
        raise ValueError("min_seeds_per_matchup must be at least 1")

    results: list[ValidationCheckResult] = []

    for _, row in best_rows.iterrows():
        raw_seed_count = row.get("seeds", 0)
        seed_count = 0 if pd.isna(raw_seed_count) else int(raw_seed_count)
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        sufficient = seed_count >= minimum_seeds

        results.append(
            ValidationCheckResult(
                check_name=(
                    "Minimum seed coverage "
                    f"for {agent_name} vs {opponent_name}"
                ),
                status=STATUS_PASS if sufficient else STATUS_FAIL,
                category="seed_coverage",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(row),
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
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for _, row in best_rows.iterrows():
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
                check_name=(
                    "Seed stability "
                    f"for {agent_name} vs {opponent_name}"
                ),
                status=status,
                category="seed_stability",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(row),
                observed_value=value,
                threshold=thresholds.max_std_across_seeds_bb,
                message=(
                    "Mean profit std across seeds is "
                    f"{_format_float(value)} BB/game."
                ),
            )
        )

    return results

def validate_extreme_bb_per_100(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for _, row in best_rows.iterrows():
        bb_per_100 = float(row["bb_per_100"])
        mean_hands_played = float(row.get("mean_hands_played", 0.0))
        agent_name = str(row["agent_name"])
        opponent_name = str(row["opponent_name"])
        is_extreme = (
            abs(bb_per_100)
            > thresholds.extreme_bb_per_100_threshold
            and mean_hands_played
            < thresholds.low_mean_hands_played_threshold
        )

        results.append(
            ValidationCheckResult(
                check_name=(
                    "Extreme BB/100 sanity check "
                    f"for {agent_name} vs {opponent_name}"
                ),
                status=STATUS_WARNING if is_extreme else STATUS_PASS,
                category="bb_per_100_sanity",
                algorithm_name=algorithm_name_for_agent(agent_name),
                agent_name=agent_name,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(row),
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
                    "bb_per_100_threshold": (
                        thresholds.extreme_bb_per_100_threshold
                    ),
                    "low_mean_hands_played_threshold": (
                        thresholds.low_mean_hands_played_threshold
                    ),
                },
            )
        )

    return results

def validate_always_raise_outperforms_adaptive(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = TRAINING_OPPONENT_TYPES,
    algorithm_specs: Iterable[AlgorithmValidationSpec] | None = None,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []
    specs = tuple(algorithm_specs or available_algorithm_specs(best_rows))

    for spec in specs:
        for opponent_name in opponents:
            matchups = (
                (ALWAYS_RAISE_AGENT, opponent_name),
                (spec.adaptive_agent, opponent_name),
            )
            checkpoint_episode, rows = (
                _find_rows_at_latest_common_checkpoint(best_rows, matchups)
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

            if checkpoint_episode is None:
                results.append(
                    _missing_common_checkpoint_result(
                        check_name,
                        "always_raise_sanity",
                        best_rows,
                        matchups,
                        algorithm_name=spec.algorithm_name,
                        agent_name=spec.adaptive_agent,
                        opponent_name=opponent_name,
                    )
                )
                continue

            delta = float(
                always_raise_row["mean_profit_bb"]
                - adaptive_row["mean_profit_bb"]
            )
            is_large_gap = (
                delta >= thresholds.always_raise_adaptive_warning_gap_bb
            )

            results.append(
                ValidationCheckResult(
                    check_name=check_name,
                    status=STATUS_WARNING if is_large_gap else STATUS_PASS,
                    category="always_raise_sanity",
                    algorithm_name=spec.algorithm_name,
                    agent_name=ALWAYS_RAISE_AGENT,
                    opponent_name=opponent_name,
                    checkpoint_episode=checkpoint_episode,
                    observed_value=delta,
                    threshold=(
                        thresholds.always_raise_adaptive_warning_gap_bb
                    ),
                    message=(
                        "Always-raise minus adaptive mean profit is "
                        f"{_format_float(delta)} BB/game."
                    ),
                    details={
                        "algorithm": spec.algorithm_name,
                        "adaptive_agent": spec.adaptive_agent,
                        "always_raise_mean_profit_bb": float(
                            always_raise_row["mean_profit_bb"]
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

def validate_always_raise_trivial_exploit(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
    opponents: Iterable[str] = TRAINING_OPPONENT_TYPES,
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        always_raise_row = _find_row(
            best_rows,
            ALWAYS_RAISE_AGENT,
            opponent_name,
        )
        check_name = (
            "Always-raise trivial exploit sanity check "
            f"vs {opponent_name}"
        )

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
                status=(
                    STATUS_WARNING
                    if is_trivial_exploit
                    else STATUS_PASS
                ),
                category="always_raise_sanity",
                agent_name=ALWAYS_RAISE_AGENT,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(always_raise_row),
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
                    "high_win_rate_threshold": (
                        thresholds.high_always_raise_win_rate
                    ),
                },
            )
        )

    return results

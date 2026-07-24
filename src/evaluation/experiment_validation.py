from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.evaluation.checkpoint_report import (
    aggregate_across_seeds,
    display_agent_name,
    load_checkpoint_report_data,
)
from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_RAISE_AGENT,
    ORACLE_ADAPTIVE_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.html_utils import write_text
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
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

DEFAULT_ADAPTIVE_RULE_BASED_OPPONENTS = (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)

DEFAULT_CLASSIFIER_OPPONENTS = TRAINING_OPPONENT_TYPES
DEFAULT_ORACLE_OPPONENTS = TRAINING_OPPONENT_TYPES


@dataclass(frozen=True)
class ValidationThresholds:
    min_adaptive_delta_vs_rule_based_bb: float = 0.0
    max_oracle_underperformance_bb: float = 1.0
    min_fish_win_rate: float = 95.0
    min_fish_mean_profit_bb: float = 15.0
    min_classifier_accuracy: float = 80.0
    min_classifier_coverage: float = 80.0
    max_std_across_seeds_bb: float = 5.0
    extreme_bb_per_100_threshold: float = 300.0
    low_mean_hands_played_threshold: float = 5.0
    always_raise_adaptive_warning_gap_bb: float = 3.0
    high_always_raise_mean_profit_bb: float = 18.0
    high_always_raise_win_rate: float = 95.0


@dataclass(frozen=True)
class ValidationCheckResult:
    check_name: str
    status: str
    message: str
    category: str
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
) -> pd.Series | None:
    matching = df[
        (df["agent_name"] == agent_name)
        & (df["opponent_name"] == opponent_name)
    ]

    if matching.empty:
        return None

    return matching.iloc[0]


def _checkpoint_episode(row: pd.Series | None) -> int | None:
    if row is None or "checkpoint_episode" not in row:
        return None
    return int(row["checkpoint_episode"])


def _missing_row_result(
    check_name: str,
    category: str,
    agent_name: str,
    opponent_name: str,
) -> ValidationCheckResult:
    return ValidationCheckResult(
        check_name=check_name,
        status=STATUS_SKIPPED,
        category=category,
        agent_name=agent_name,
        opponent_name=opponent_name,
        message=(
            f"Missing row for {agent_name} vs {opponent_name}."
        ),
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
            ORACLE_ADAPTIVE_AGENT,
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
                    ORACLE_ADAPTIVE_AGENT,
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
                agent_name=ORACLE_ADAPTIVE_AGENT,
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


def validate_fish_exploitation(
    best_rows: pd.DataFrame,
    thresholds: ValidationThresholds,
) -> list[ValidationCheckResult]:
    adaptive_row = _find_row(
        best_rows,
        ADAPTIVE_MC_AGENT,
        OPPONENT_TYPE_FISH,
    )
    check_name = "Adaptive exploits FishPlayer"

    if adaptive_row is None:
        return [
            _missing_row_result(
                check_name,
                "fish_exploitation",
                ADAPTIVE_MC_AGENT,
                OPPONENT_TYPE_FISH,
            )
        ]

    mean_profit_bb = float(adaptive_row["mean_profit_bb"])
    win_rate = float(adaptive_row["win_rate"])
    passed = (
        mean_profit_bb >= thresholds.min_fish_mean_profit_bb
        and win_rate >= thresholds.min_fish_win_rate
    )

    return [
        ValidationCheckResult(
            check_name=check_name,
            status=STATUS_PASS if passed else STATUS_FAIL,
            category="fish_exploitation",
            agent_name=ADAPTIVE_MC_AGENT,
            opponent_name=OPPONENT_TYPE_FISH,
            checkpoint_episode=_checkpoint_episode(adaptive_row),
            observed_value=mean_profit_bb,
            threshold=thresholds.min_fish_mean_profit_bb,
            message=(
                "Adaptive vs fish: "
                f"mean_profit_bb={_format_float(mean_profit_bb)}, "
                f"win_rate={_format_float(win_rate)}%."
            ),
            details={
                "mean_profit_bb": mean_profit_bb,
                "min_mean_profit_bb": (
                    thresholds.min_fish_mean_profit_bb
                ),
                "win_rate": win_rate,
                "min_win_rate": thresholds.min_fish_win_rate,
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
) -> list[ValidationCheckResult]:
    results: list[ValidationCheckResult] = []

    for opponent_name in opponents:
        always_raise_row = _find_row(
            best_rows,
            ALWAYS_RAISE_AGENT,
            opponent_name,
        )
        adaptive_row = _find_row(
            best_rows,
            ADAPTIVE_MC_AGENT,
            opponent_name,
        )
        check_name = (
            "Always-raise dominance sanity check "
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

        if adaptive_row is None:
            results.append(
                _missing_row_result(
                    check_name,
                    "always_raise_sanity",
                    ADAPTIVE_MC_AGENT,
                    opponent_name,
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
                agent_name=ALWAYS_RAISE_AGENT,
                opponent_name=opponent_name,
                checkpoint_episode=_checkpoint_episode(always_raise_row),
                observed_value=delta,
                threshold=(
                    thresholds.always_raise_adaptive_warning_gap_bb
                ),
                message=(
                    "Always-raise minus adaptive mean profit is "
                    f"{_format_float(delta)} BB/game."
                ),
                details={
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


def validate_fish_baseline_saturation(
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
            OPPONENT_TYPE_FISH,
        )
        for agent_name in required_agents
    }
    check_name = "FishPlayer baseline saturation sanity check"

    for agent_name, row in rows_by_agent.items():
        if row is None:
            return [
                _missing_row_result(
                    check_name,
                    "always_raise_sanity",
                    agent_name,
                    OPPONENT_TYPE_FISH,
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
            mean_profit_bb >= thresholds.min_fish_mean_profit_bb
            and win_rate >= thresholds.min_fish_win_rate
        ):
            saturated_agents.append(agent_name)

    is_saturated = len(saturated_agents) == len(required_agents)

    return [
        ValidationCheckResult(
            check_name=check_name,
            status=STATUS_WARNING if is_saturated else STATUS_PASS,
            category="always_raise_sanity",
            agent_name=ALWAYS_RAISE_AGENT,
            opponent_name=OPPONENT_TYPE_FISH,
            checkpoint_episode=_checkpoint_episode(
                rows_by_agent[ALWAYS_RAISE_AGENT]
            ),
            observed_value=float(
                rows_by_agent[ALWAYS_RAISE_AGENT]["mean_profit_bb"]
            ),
            threshold=thresholds.min_fish_mean_profit_bb,
            message=(
                "FishPlayer may be too weak to distinguish agent "
                "quality when adaptive, rule-based, and always-raise "
                "all reach the fish exploitation thresholds."
                if is_saturated
                else "FishPlayer still differentiates at least one "
                "baseline below the exploitation thresholds."
            ),
            details={
                **details,
                "saturated_agents": saturated_agents,
                "required_agents": list(required_agents),
                "min_fish_mean_profit_bb": (
                    thresholds.min_fish_mean_profit_bb
                ),
                "min_fish_win_rate": thresholds.min_fish_win_rate,
            },
        )
    ]


def validate_checkpoint_results(
    input_path: str | Path,
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    thresholds = thresholds or ValidationThresholds()
    metrics = load_checkpoint_report_data(input_path)
    aggregated = aggregate_across_seeds(metrics)
    aggregated = _add_mean_hands_played(aggregated, metrics)
    best_rows = _best_rows_by_agent_and_opponent(aggregated)

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
        validate_fish_exploitation(
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
        validate_fish_baseline_saturation(
            best_rows,
            thresholds,
        )
    )

    return ValidationReport(
        input_path=str(input_path),
        thresholds=thresholds,
        checks=checks,
    )


def validation_checks_to_dataframe(
    checks: Iterable[ValidationCheckResult],
) -> pd.DataFrame:
    rows = [
        check.to_dict()
        for check in checks
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "check_name",
                "status",
                "category",
                "agent_name",
                "opponent_name",
                "checkpoint_episode",
                "observed_value",
                "threshold",
                "message",
            ]
        )

    return pd.DataFrame(rows)


def _format_report_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[
        [
            "status",
            "category",
            "check_name",
            "agent_name",
            "opponent_name",
            "checkpoint_episode",
            "observed_value",
            "threshold",
            "message",
        ]
    ].copy()

    for column in ["observed_value", "threshold"]:
        table[column] = table[column].map(_format_float)

    table["agent_name"] = table["agent_name"].map(
        lambda value: display_agent_name(value)
        if isinstance(value, str)
        else value
    )

    return table


def render_validation_markdown(report: ValidationReport) -> str:
    checks_df = validation_checks_to_dataframe(report.checks)
    counts = report.status_counts()
    status_table = pd.DataFrame(
        [
            {
                "status": status,
                "count": counts[status],
            }
            for status in VALIDATION_STATUSES
        ]
    )

    lines = [
        "# Experiment validation report",
        "",
        "This report runs automated sanity checks on checkpoint "
        "evaluation results.",
        "",
        "## Input",
        "",
        f"- **Evaluation file:** `{report.input_path}`",
        f"- **Overall status:** `{'PASS' if report.passed else 'FAIL'}`",
        "",
        "## Status summary",
        "",
        status_table.to_markdown(index=False),
        "",
        "## Thresholds",
        "",
        pd.DataFrame(
            [
                {
                    "threshold": key,
                    "value": value,
                }
                for key, value in asdict(report.thresholds).items()
            ]
        ).to_markdown(index=False),
        "",
        "## Checks",
        "",
    ]

    if checks_df.empty:
        lines.append("No checks were generated.")
    else:
        lines.append(
            _format_report_table(checks_df).to_markdown(index=False)
        )

    lines.append("")
    return "\n".join(lines)


def write_validation_markdown_report(
    report: ValidationReport,
    output_dir: str | Path,
    filename: str = "experiment_validation.md",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    write_text(
        output_path,
        render_validation_markdown(report),
    )
    return output_path


def write_validation_json_report(
    report: ValidationReport,
    output_dir: str | Path,
    filename: str = "experiment_validation.json",
) -> Path:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    write_text(
        output_path,
        json.dumps(report.to_dict(), indent=2),
    )
    return output_path

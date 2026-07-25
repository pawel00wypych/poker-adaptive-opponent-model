from __future__ import annotations

import json
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
from src.poker.constants import OPPONENT_TYPE_FISH

QUALITY_OK = "OK"
QUALITY_WARNING = "WARNING"
QUALITY_FAIL = "FAIL"

QUALITY_STATUSES = (
    QUALITY_OK,
    QUALITY_WARNING,
    QUALITY_FAIL,
)

SUMMARY_GROUP_COLUMNS = [
    "training_run",
    "opponent_name",
    "checkpoint_episode",
]

RANKING_SORT_COLUMNS = [
    "training_run",
    "opponent_name",
    "checkpoint_episode",
    "mean_profit_bb",
    "win_rate",
    "bust_rate",
    "mean_profit_bb_std_across_seeds",
]

SUMMARY_TABLE_COLUMNS = [
    "training_run",
    "checkpoint_episode",
    "opponent_name",
    "rank",
    "agent_name",
    "mean_profit_bb",
    "mean_profit_bb_std_across_seeds",
    "bb_per_100",
    "win_rate",
    "bust_rate",
    "delta_vs_rule_based",
    "delta_vs_oracle",
    "quality_status",
    "quality_reason",
]

DELTA_TABLE_COLUMNS = [
    "training_run",
    "checkpoint_episode",
    "opponent_name",
    "agent_name",
    "mean_profit_bb",
    "delta_vs_rule_based",
    "delta_vs_oracle",
]

QUALITY_TABLE_COLUMNS = [
    "training_run",
    "checkpoint_episode",
    "opponent_name",
    "agent_name",
    "quality_status",
    "quality_reason",
    "mean_profit_bb",
    "mean_profit_bb_std_across_seeds",
    "win_rate",
    "bust_rate",
]


@dataclass(frozen=True)
class SummaryThresholds:
    max_std_across_seeds_bb: float = 5.0
    min_warning_win_rate: float = 55.0
    high_always_raise_mean_profit_bb: float = 18.0
    high_always_raise_win_rate: float = 95.0
    fish_saturation_win_rate: float = 95.0
    fish_saturation_mean_profit_bb: float = 15.0


@dataclass(frozen=True)
class ExperimentSummary:
    input_path: str
    thresholds: SummaryThresholds
    overview: dict[str, object]
    main_findings: list[str]
    ranking: list[dict[str, object]]
    deltas: list[dict[str, object]]
    quality_flags: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "thresholds": asdict(self.thresholds),
            "overview": self.overview,
            "main_findings": self.main_findings,
            "ranking": self.ranking,
            "deltas": self.deltas,
            "quality_flags": self.quality_flags,
        }


def _round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = result.select_dtypes(include="number").columns

    for column in numeric_columns:
        if column in {"checkpoint_episode", "rank", "games", "seeds"}:
            continue
        result[column] = result[column].round(3)

    return result


def _display_table(df: pd.DataFrame) -> pd.DataFrame:
    result = _round_numeric_columns(df)

    if "agent_name" in result.columns:
        result["agent_name"] = result["agent_name"].map(display_agent_name)

    return result


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _add_mean_hands_played(
    aggregated: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if aggregated.empty or "mean_hands_played" in aggregated.columns:
        return aggregated.copy()

    required_columns = {
        "training_run",
        "agent_name",
        "opponent_name",
        "checkpoint_episode",
        "total_hands",
        "games",
    }

    if not required_columns.issubset(metrics.columns):
        result = aggregated.copy()
        result["mean_hands_played"] = 0.0
        return result

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


def load_experiment_summary_data(
    input_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = load_checkpoint_report_data(input_path)
    aggregated = aggregate_across_seeds(metrics)
    aggregated = _add_mean_hands_played(aggregated, metrics)
    return metrics, aggregated


def build_agent_ranking(aggregated: pd.DataFrame) -> pd.DataFrame:
    if aggregated.empty:
        result = aggregated.copy()
        result["rank"] = pd.Series(dtype="int64")
        return result

    result = aggregated.sort_values(
        RANKING_SORT_COLUMNS,
        ascending=[True, True, True, False, False, True, True],
    ).reset_index(drop=True)

    result["rank"] = (
        result.groupby(SUMMARY_GROUP_COLUMNS).cumcount() + 1
    )

    front_columns = SUMMARY_GROUP_COLUMNS + ["rank", "agent_name"]
    remaining_columns = [
        column for column in result.columns
        if column not in front_columns
    ]
    return result[front_columns + remaining_columns]


def add_baseline_deltas(ranking: pd.DataFrame) -> pd.DataFrame:
    result = ranking.copy()

    for baseline_agent, column_name in [
        (RULE_BASED_AGENT, "delta_vs_rule_based"),
        (ORACLE_ADAPTIVE_AGENT, "delta_vs_oracle"),
    ]:
        baseline = result[
            result["agent_name"] == baseline_agent
        ][SUMMARY_GROUP_COLUMNS + ["mean_profit_bb"]].rename(
            columns={"mean_profit_bb": f"{baseline_agent}_mean_profit_bb"}
        )

        result = result.merge(
            baseline,
            on=SUMMARY_GROUP_COLUMNS,
            how="left",
        )
        result[column_name] = (
            result["mean_profit_bb"]
            - result[f"{baseline_agent}_mean_profit_bb"]
        )
        result = result.drop(columns=[f"{baseline_agent}_mean_profit_bb"])

    return result


def _quality_status_and_reason(
    row: pd.Series,
    thresholds: SummaryThresholds,
) -> tuple[str, str]:
    mean_profit_bb = float(row.get("mean_profit_bb", 0.0))
    std_across_seeds = float(
        row.get("mean_profit_bb_std_across_seeds", 0.0)
    )
    win_rate = float(row.get("win_rate", 0.0))
    agent_name = str(row.get("agent_name", ""))

    warnings: list[str] = []

    if mean_profit_bb < 0.0:
        return (
            QUALITY_FAIL,
            f"Negative mean profit ({mean_profit_bb:.3f} BB/game).",
        )

    if std_across_seeds > thresholds.max_std_across_seeds_bb:
        warnings.append(
            "High seed variance "
            f"({std_across_seeds:.3f} > "
            f"{thresholds.max_std_across_seeds_bb:.3f} BB/game)."
        )

    if win_rate < thresholds.min_warning_win_rate:
        warnings.append(
            "Low win rate "
            f"({win_rate:.3f}% < "
            f"{thresholds.min_warning_win_rate:.3f}%)."
        )

    if (
        agent_name == ALWAYS_RAISE_AGENT
        and mean_profit_bb >= thresholds.high_always_raise_mean_profit_bb
        and win_rate >= thresholds.high_always_raise_win_rate
    ):
        warnings.append(
            "Always-raise reaches very high profit and win rate; "
            "opponent may be vulnerable to trivial aggression."
        )

    if warnings:
        return QUALITY_WARNING, " ".join(warnings)

    return QUALITY_OK, "Meets default quality thresholds."


def add_quality_flags(
    ranking_with_deltas: pd.DataFrame,
    thresholds: SummaryThresholds,
) -> pd.DataFrame:
    result = ranking_with_deltas.copy()

    statuses_and_reasons = result.apply(
        lambda row: _quality_status_and_reason(row, thresholds),
        axis=1,
    )
    result["quality_status"] = [item[0] for item in statuses_and_reasons]
    result["quality_reason"] = [item[1] for item in statuses_and_reasons]

    return result


def build_overview(
    metrics: pd.DataFrame,
    summary_table: pd.DataFrame,
) -> dict[str, object]:
    if metrics.empty:
        return {
            "training_runs": [],
            "checkpoints": [],
            "seeds": [],
            "opponents": [],
            "agents": [],
            "raw_games": 0,
            "summary_rows": 0,
        }

    return {
        "training_runs": sorted(metrics["training_run"].dropna().unique()),
        "checkpoints": sorted(
            int(value)
            for value in metrics["checkpoint_episode"].dropna().unique()
        ),
        "seeds": sorted(
            int(value)
            for value in metrics["model_seed"].dropna().unique()
        ),
        "opponents": sorted(metrics["opponent_name"].dropna().unique()),
        "agents": sorted(metrics["agent_name"].dropna().unique()),
        "raw_games": int(metrics["games"].sum())
        if "games" in metrics.columns
        else 0,
        "summary_rows": int(len(summary_table)),
    }


def _format_signed(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.3f}"


def _average_mean_profit(summary_table: pd.DataFrame) -> pd.DataFrame:
    if summary_table.empty:
        return pd.DataFrame(columns=["agent_name", "avg_mean_profit_bb"])

    return (
        summary_table.groupby("agent_name")["mean_profit_bb"]
        .mean()
        .reset_index(name="avg_mean_profit_bb")
        .sort_values("avg_mean_profit_bb", ascending=False)
        .reset_index(drop=True)
    )


def _adaptive_rule_based_findings(summary_table: pd.DataFrame) -> str | None:
    adaptive_rows = summary_table[
        summary_table["agent_name"] == ADAPTIVE_MC_AGENT
    ]
    rule_based_rows = summary_table[
        summary_table["agent_name"] == RULE_BASED_AGENT
    ]

    if adaptive_rows.empty or rule_based_rows.empty:
        return None

    merged = adaptive_rows[SUMMARY_GROUP_COLUMNS + ["mean_profit_bb"]].merge(
        rule_based_rows[SUMMARY_GROUP_COLUMNS + ["mean_profit_bb"]],
        on=SUMMARY_GROUP_COLUMNS,
        suffixes=("_adaptive", "_rule_based"),
    )

    if merged.empty:
        return None

    merged["delta"] = (
        merged["mean_profit_bb_adaptive"]
        - merged["mean_profit_bb_rule_based"]
    )
    wins = int((merged["delta"] > 0).sum())
    total = int(len(merged))
    avg_delta = float(merged["delta"].mean())

    return (
        "Adaptive Monte Carlo beats the rule-based baseline in "
        f"{wins}/{total} comparable matchup groups "
        f"(average delta {_format_signed(avg_delta)} BB/game)."
    )


def _oracle_gap_finding(summary_table: pd.DataFrame) -> str | None:
    if "delta_vs_oracle" not in summary_table.columns:
        return None

    comparable = summary_table[
        summary_table["delta_vs_oracle"].notna()
        & (summary_table["agent_name"] != ORACLE_ADAPTIVE_AGENT)
    ]

    if comparable.empty:
        return None

    adaptive = comparable[comparable["agent_name"] == ADAPTIVE_MC_AGENT]
    source = adaptive if not adaptive.empty else comparable
    avg_gap = float(source["delta_vs_oracle"].mean())

    return (
        "Average delta vs oracle for "
        f"{'Adaptive Monte Carlo' if not adaptive.empty else 'non-oracle agents'} "
        f"is {_format_signed(avg_gap)} BB/game."
    )


def _high_seed_variance_finding(
    summary_table: pd.DataFrame,
    thresholds: SummaryThresholds,
) -> str | None:
    unstable = summary_table[
        summary_table["mean_profit_bb_std_across_seeds"]
        > thresholds.max_std_across_seeds_bb
    ]

    if unstable.empty:
        return None

    examples = unstable.sort_values(
        "mean_profit_bb_std_across_seeds",
        ascending=False,
    ).head(3)
    labels = [
        f"{display_agent_name(row.agent_name)} vs {row.opponent_name} "
        f"({row.mean_profit_bb_std_across_seeds:.3f})"
        for row in examples.itertuples(index=False)
    ]

    return (
        f"High seed variance detected in {len(unstable)} summary rows; "
        "largest examples: " + "; ".join(labels) + "."
    )


def _always_raise_finding(
    summary_table: pd.DataFrame,
    thresholds: SummaryThresholds,
) -> str | None:
    rows = summary_table[
        (summary_table["agent_name"] == ALWAYS_RAISE_AGENT)
        & (
            summary_table["mean_profit_bb"]
            >= thresholds.high_always_raise_mean_profit_bb
        )
        & (summary_table["win_rate"] >= thresholds.high_always_raise_win_rate)
    ]

    if rows.empty:
        return None

    opponents = ", ".join(sorted(rows["opponent_name"].unique()))
    return (
        "Always-raise reaches very high profit and win rate against "
        f"{opponents}, suggesting those opponents may be vulnerable "
        "to trivial aggression."
    )


def _fish_saturation_finding(
    summary_table: pd.DataFrame,
    thresholds: SummaryThresholds,
) -> str | None:
    fish_rows = summary_table[
        summary_table["opponent_name"] == OPPONENT_TYPE_FISH
    ]

    if len(fish_rows) < 2:
        return None

    saturated = fish_rows[
        (fish_rows["mean_profit_bb"] >= thresholds.fish_saturation_mean_profit_bb)
        & (fish_rows["win_rate"] >= thresholds.fish_saturation_win_rate)
    ]

    if len(saturated) < 2:
        return None

    return (
        f"FishPlayer appears saturated: {len(saturated)} agents reach "
        f"at least {thresholds.fish_saturation_mean_profit_bb:.1f} "
        "BB/game and "
        f"{thresholds.fish_saturation_win_rate:.1f}% win rate."
    )


def generate_main_findings(
    summary_table: pd.DataFrame,
    thresholds: SummaryThresholds,
) -> list[str]:
    findings: list[str] = []

    average_profit = _average_mean_profit(summary_table)
    if not average_profit.empty:
        best = average_profit.iloc[0]
        findings.append(
            "Best average agent across evaluated groups is "
            f"{display_agent_name(str(best['agent_name']))} "
            f"with {best['avg_mean_profit_bb']:.3f} BB/game."
        )

    for finding in [
        _adaptive_rule_based_findings(summary_table),
        _oracle_gap_finding(summary_table),
        _high_seed_variance_finding(summary_table, thresholds),
        _always_raise_finding(summary_table, thresholds),
        _fish_saturation_finding(summary_table, thresholds),
    ]:
        if finding is not None:
            findings.append(finding)

    failures = summary_table[
        summary_table["quality_status"] == QUALITY_FAIL
    ]
    warnings = summary_table[
        summary_table["quality_status"] == QUALITY_WARNING
    ]

    findings.append(
        "Traffic-light summary: "
        f"{len(summary_table) - len(failures) - len(warnings)} OK, "
        f"{len(warnings)} WARNING, {len(failures)} FAIL."
    )

    return findings


def build_experiment_summary(
    input_path: str | Path,
    thresholds: SummaryThresholds | None = None,
) -> tuple[ExperimentSummary, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = thresholds or SummaryThresholds()
    metrics, aggregated = load_experiment_summary_data(input_path)
    ranking = build_agent_ranking(aggregated)
    ranking = add_baseline_deltas(ranking)
    summary_table = add_quality_flags(ranking, thresholds)

    deltas = summary_table[_existing_columns(summary_table, DELTA_TABLE_COLUMNS)]
    quality_flags = summary_table[
        _existing_columns(summary_table, QUALITY_TABLE_COLUMNS)
    ]

    overview = build_overview(metrics, summary_table)
    main_findings = generate_main_findings(summary_table, thresholds)

    report = ExperimentSummary(
        input_path=str(input_path),
        thresholds=thresholds,
        overview=overview,
        main_findings=main_findings,
        ranking=summary_table.to_dict(orient="records"),
        deltas=deltas.to_dict(orient="records"),
        quality_flags=quality_flags.to_dict(orient="records"),
    )

    return report, summary_table, deltas, quality_flags


def _overview_markdown(overview: dict[str, object]) -> str:
    rows = [
        {"field": key, "value": value}
        for key, value in overview.items()
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def render_experiment_summary_markdown(
    report: ExperimentSummary,
    ranking: pd.DataFrame,
    deltas: pd.DataFrame,
    quality_flags: pd.DataFrame,
) -> str:
    ranking_table = ranking[
        _existing_columns(ranking, SUMMARY_TABLE_COLUMNS)
    ]
    delta_table = deltas[
        _existing_columns(deltas, DELTA_TABLE_COLUMNS)
    ]
    quality_table = quality_flags[
        _existing_columns(quality_flags, QUALITY_TABLE_COLUMNS)
    ]

    findings = "\n".join(
        f"{index + 1}. {finding}"
        for index, finding in enumerate(report.main_findings)
    )

    return "\n".join(
        [
            "# Experiment summary",
            "",
            "This report summarizes checkpoint evaluation results "
            "across agents, opponents, checkpoints, and seeds.",
            "",
            "## Overview",
            "",
            _overview_markdown(report.overview),
            "",
            "## Main findings",
            "",
            findings,
            "",
            "## Agent ranking by opponent and checkpoint",
            "",
            _display_table(ranking_table).to_markdown(index=False),
            "",
            "## Delta vs baselines",
            "",
            _display_table(delta_table).to_markdown(index=False),
            "",
            "## Traffic-light quality flags",
            "",
            _display_table(quality_table).to_markdown(index=False),
            "",
        ]
    )


def write_dataframe_csv(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def write_dataframe_latex(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = _display_table(df).to_latex(
        index=False,
        escape=True,
        float_format="%.3f",
    )
    write_text(output_path, text)
    return output_path


def write_experiment_summary_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    thresholds: SummaryThresholds | None = None,
    report_format: str = "all",
    export_latex: bool = True,
) -> list[Path]:
    if report_format not in {"markdown", "json", "both", "all"}:
        raise ValueError(
            "Unsupported report_format. Expected one of: "
            "markdown, json, both, all."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, ranking, deltas, quality_flags = build_experiment_summary(
        input_path=input_path,
        thresholds=thresholds,
    )

    created_paths: list[Path] = []

    if report_format in {"markdown", "both", "all"}:
        markdown_path = output_dir / "experiment_summary.md"
        write_text(
            markdown_path,
            render_experiment_summary_markdown(
                report,
                ranking,
                deltas,
                quality_flags,
            ),
        )
        created_paths.append(markdown_path)

    if report_format in {"json", "both", "all"}:
        json_path = output_dir / "experiment_summary.json"
        write_text(json_path, json.dumps(report.to_dict(), indent=2))
        created_paths.append(json_path)

    csv_exports = [
        ("agent_ranking.csv", ranking),
        ("deltas.csv", deltas),
        ("quality_flags.csv", quality_flags),
    ]

    for filename, df in csv_exports:
        created_paths.append(
            write_dataframe_csv(df, output_dir / filename)
        )

    if export_latex:
        latex_exports = [
            ("agent_ranking.tex", ranking),
            ("deltas.tex", deltas),
            ("quality_flags.tex", quality_flags),
        ]
        for filename, df in latex_exports:
            created_paths.append(
                write_dataframe_latex(df, output_dir / filename)
            )

    return created_paths

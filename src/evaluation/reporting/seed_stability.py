from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_MAX_COLUMN,
    SEED_MIN_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
)
from src.evaluation.reporting.experiment_summary import (
    dataframe_records_with_missing_as_none,
    write_dataframe_csv,
    write_dataframe_latex,
)
from src.evaluation.reporting.html_utils import write_text
from src.evaluation.reporting.training_opponent_report import (
    aggregate_across_seeds,
    display_agent_name,
    load_training_opponent_report_data,
)

RANKING_STABILITY_HIGH = "high"
RANKING_STABILITY_MODERATE = "moderate"
RANKING_STABILITY_LOW = "low"
RANKING_STABILITY_INSUFFICIENT = "insufficient_data"

RANKING_CONTEXT_COLUMNS = [
    "training_run",
    "opponent_name",
    "training_episode",
]

RANKING_SEED_COLUMNS = [
    *RANKING_CONTEXT_COLUMNS,
    "model_seed",
]

SEED_SUMMARY_GROUP_COLUMNS = [
    "training_run",
    "agent_name",
    "opponent_name",
    "training_episode",
]

SEED_PERFORMANCE_COLUMNS = [
    "training_run",
    "training_episode",
    "opponent_name",
    "model_seed",
    "agent_name",
    "games",
    "mean_profit_bb",
    "bb_per_100",
    "win_rate",
    "bust_rate",
    "rank",
    "is_seed_winner",
    "ranking_complete",
]

SEED_RANKING_COLUMNS = [
    "training_run",
    "training_episode",
    "opponent_name",
    "model_seed",
    "rank",
    "agent_name",
    "mean_profit_bb",
    "is_seed_winner",
    "ranking_complete",
]

SEED_STABILITY_SUMMARY_COLUMNS = [
    "training_run",
    "training_episode",
    "opponent_name",
    "agent_name",
    "seeds",
    "games",
    "mean_profit_bb",
    "mean_profit_bb_std_across_seeds",
    SEED_STANDARD_ERROR_COLUMN,
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    "best_seed",
    "best_seed_mean_profit_bb",
    "worst_seed",
    "worst_seed_mean_profit_bb",
    SEED_MIN_COLUMN,
    SEED_MAX_COLUMN,
    SEED_SPREAD_COLUMN,
    "rank_seed_count",
    "mean_rank",
    "rank_std",
    "best_rank",
    "worst_rank",
    "rank_spread",
    "first_place_count",
    "first_place_rate",
]

RANKING_STABILITY_COLUMNS = [
    "training_run",
    "training_episode",
    "opponent_name",
    "seed_count",
    "complete_seed_count",
    "excluded_seed_count",
    "agent_count",
    "kendalls_w",
    "ranking_stability",
    "most_frequent_winner",
    "winner_consistency",
]


@dataclass(frozen=True)
class SeedStabilityConfig:
    min_complete_seeds_for_ranking: int = 2

    def __post_init__(self) -> None:
        if self.min_complete_seeds_for_ranking < 2:
            raise ValueError("min_complete_seeds_for_ranking must be at least 2")


@dataclass(frozen=True)
class SeedStabilityReport:
    input_path: str
    config: SeedStabilityConfig
    overview: dict[str, object]
    main_findings: list[str]
    seed_performance: list[dict[str, object]]
    seed_stability_summary: list[dict[str, object]]
    seed_rankings: list[dict[str, object]]
    ranking_stability: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "config": asdict(self.config),
            "overview": self.overview,
            "main_findings": self.main_findings,
            "seed_performance": self.seed_performance,
            "seed_stability_summary": self.seed_stability_summary,
            "seed_rankings": self.seed_rankings,
            "ranking_stability": self.ranking_stability,
        }


def _existing_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def select_final_model_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return all final-model metrics without checkpoint-based selection."""
    if metrics.empty:
        return metrics.copy()

    return metrics.sort_values(
        [
            "training_run",
            "training_episode",
            "opponent_name",
            "model_seed",
            "agent_name",
        ]
    ).reset_index(drop=True)


def build_seed_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add per-seed agent rankings and ranking-completeness markers."""
    required_columns = {
        "training_run",
        "training_episode",
        "opponent_name",
        "model_seed",
        "agent_name",
        "mean_profit_bb",
    }
    missing_columns = sorted(required_columns.difference(metrics.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot build seed performance without columns: {missing_columns}."
        )

    if metrics.empty:
        return pd.DataFrame(columns=SEED_PERFORMANCE_COLUMNS)

    result = metrics.copy()
    duplicate_keys = [*RANKING_SEED_COLUMNS, "agent_name"]
    duplicated = result.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        duplicate_rows = result.loc[duplicated, duplicate_keys]
        raise ValueError(
            "Expected one seed-level result per agent and matchup; "
            f"found duplicates: {duplicate_rows.to_dict(orient='records')}."
        )

    result["mean_profit_bb"] = pd.to_numeric(
        result["mean_profit_bb"],
        errors="raise",
    )
    seed_groups = result.groupby(RANKING_SEED_COLUMNS, dropna=False)
    result["rank"] = seed_groups["mean_profit_bb"].rank(
        method="average",
        ascending=False,
    )
    best_profit = seed_groups["mean_profit_bb"].transform("max")
    result["is_seed_winner"] = result["mean_profit_bb"].eq(best_profit)

    expected_agent_count = result.groupby(
        RANKING_CONTEXT_COLUMNS,
        dropna=False,
    )["agent_name"].transform("nunique")
    evaluated_agent_count = seed_groups["agent_name"].transform("nunique")
    result["ranking_complete"] = evaluated_agent_count.eq(expected_agent_count)

    columns = _existing_columns(result, SEED_PERFORMANCE_COLUMNS)
    return (
        result[columns]
        .sort_values([*RANKING_SEED_COLUMNS, "rank", "agent_name"])
        .reset_index(drop=True)
    )


def build_seed_rankings(seed_performance: pd.DataFrame) -> pd.DataFrame:
    columns = _existing_columns(seed_performance, SEED_RANKING_COLUMNS)
    return seed_performance[columns].copy()


def _extreme_seed_rows(
    seed_performance: pd.DataFrame,
    *,
    best: bool,
) -> pd.DataFrame:
    ascending = [True] * len(SEED_SUMMARY_GROUP_COLUMNS) + [not best, True]
    ordered = seed_performance.sort_values(
        [
            *SEED_SUMMARY_GROUP_COLUMNS,
            "mean_profit_bb",
            "model_seed",
        ],
        ascending=ascending,
    )
    extreme = ordered.drop_duplicates(SEED_SUMMARY_GROUP_COLUMNS)
    prefix = "best" if best else "worst"
    return extreme[
        [*SEED_SUMMARY_GROUP_COLUMNS, "model_seed", "mean_profit_bb"]
    ].rename(
        columns={
            "model_seed": f"{prefix}_seed",
            "mean_profit_bb": f"{prefix}_seed_mean_profit_bb",
        }
    )


def build_seed_stability_summary(
    metrics: pd.DataFrame,
    seed_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Combine PR51 statistics with best/worst seed and rank summaries."""
    if metrics.empty:
        return pd.DataFrame(columns=SEED_STABILITY_SUMMARY_COLUMNS)

    summary = aggregate_across_seeds(metrics)
    summary = summary.merge(
        _extreme_seed_rows(seed_performance, best=True),
        on=SEED_SUMMARY_GROUP_COLUMNS,
        how="left",
    )
    summary = summary.merge(
        _extreme_seed_rows(seed_performance, best=False),
        on=SEED_SUMMARY_GROUP_COLUMNS,
        how="left",
    )

    complete_rankings = seed_performance[seed_performance["ranking_complete"]]
    if complete_rankings.empty:
        rank_summary = pd.DataFrame(
            columns=[
                *SEED_SUMMARY_GROUP_COLUMNS,
                "rank_seed_count",
                "mean_rank",
                "rank_std",
                "best_rank",
                "worst_rank",
                "first_place_count",
                "first_place_rate",
                "rank_spread",
            ]
        )
    else:
        rank_summary = (
            complete_rankings.groupby(SEED_SUMMARY_GROUP_COLUMNS)
            .agg(
                rank_seed_count=("model_seed", "nunique"),
                mean_rank=("rank", "mean"),
                rank_std=("rank", "std"),
                best_rank=("rank", "min"),
                worst_rank=("rank", "max"),
                first_place_count=("is_seed_winner", "sum"),
                first_place_rate=("is_seed_winner", "mean"),
            )
            .reset_index()
        )
        rank_summary["rank_spread"] = (
            rank_summary["worst_rank"] - rank_summary["best_rank"]
        )

    summary = summary.merge(
        rank_summary,
        on=SEED_SUMMARY_GROUP_COLUMNS,
        how="left",
    )
    for column in ("rank_seed_count", "first_place_count"):
        if column in summary.columns:
            summary[column] = (
                pd.to_numeric(summary[column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

    columns = _existing_columns(summary, SEED_STABILITY_SUMMARY_COLUMNS)
    return (
        summary[columns]
        .sort_values(
            [
                "training_run",
                "training_episode",
                "opponent_name",
                "mean_profit_bb",
                "agent_name",
            ],
            ascending=[True, True, True, False, True],
        )
        .reset_index(drop=True)
    )


def calculate_kendalls_w(rank_matrix: pd.DataFrame) -> float | None:
    """Calculate Kendall's coefficient of concordance for seed rankings."""
    if rank_matrix.shape[0] < 2 or rank_matrix.shape[1] < 2:
        return None
    if rank_matrix.isna().any().any():
        return None

    ranks = rank_matrix.to_numpy(dtype="float64")
    seed_count, agent_count = ranks.shape
    rank_sums = ranks.sum(axis=0)
    expected_rank_sum = seed_count * (agent_count + 1) / 2.0
    squared_deviation_sum = float(np.square(rank_sums - expected_rank_sum).sum())

    tie_correction = 0.0
    for seed_ranks in ranks:
        _, tie_counts = np.unique(seed_ranks, return_counts=True)
        tie_correction += float(
            sum(count**3 - count for count in tie_counts if count > 1)
        )

    denominator = (
        seed_count**2 * (agent_count**3 - agent_count) - seed_count * tie_correction
    )
    if denominator <= 0.0:
        return None

    coefficient = 12.0 * squared_deviation_sum / denominator
    return float(np.clip(coefficient, 0.0, 1.0))


def describe_ranking_stability(kendalls_w: float | None) -> str:
    if kendalls_w is None:
        return RANKING_STABILITY_INSUFFICIENT
    if kendalls_w >= 0.8:
        return RANKING_STABILITY_HIGH
    if kendalls_w >= 0.5:
        return RANKING_STABILITY_MODERATE
    return RANKING_STABILITY_LOW


def build_ranking_stability(
    seed_performance: pd.DataFrame,
    *,
    min_complete_seeds: int = 2,
) -> pd.DataFrame:
    """Summarize agreement between complete per-seed agent rankings."""
    if min_complete_seeds < 2:
        raise ValueError("min_complete_seeds must be at least 2")
    if seed_performance.empty:
        return pd.DataFrame(columns=RANKING_STABILITY_COLUMNS)

    rows: list[dict[str, object]] = []
    grouped = seed_performance.groupby(
        RANKING_CONTEXT_COLUMNS,
        dropna=False,
    )
    for group_key, group in grouped:
        training_run, opponent_name, training_episode = group_key
        seed_count = int(group["model_seed"].nunique())
        agent_count = int(group["agent_name"].nunique())
        complete = group[group["ranking_complete"]]
        complete_seed_count = int(complete["model_seed"].nunique())

        kendalls_w: float | None = None
        if complete_seed_count >= min_complete_seeds and agent_count >= 2:
            rank_matrix = complete.pivot(
                index="model_seed",
                columns="agent_name",
                values="rank",
            )
            kendalls_w = calculate_kendalls_w(rank_matrix)

        winner_counts = (
            complete[complete["is_seed_winner"]]
            .groupby("agent_name")["model_seed"]
            .nunique()
        )
        if winner_counts.empty or complete_seed_count == 0:
            most_frequent_winner = None
            winner_consistency = None
        else:
            highest_count = int(winner_counts.max())
            most_frequent_winner = ", ".join(
                sorted(
                    str(agent_name)
                    for agent_name in winner_counts[
                        winner_counts == highest_count
                    ].index
                )
            )
            winner_consistency = highest_count / complete_seed_count

        rows.append(
            {
                "training_run": training_run,
                "training_episode": training_episode,
                "opponent_name": opponent_name,
                "seed_count": seed_count,
                "complete_seed_count": complete_seed_count,
                "excluded_seed_count": seed_count - complete_seed_count,
                "agent_count": agent_count,
                "kendalls_w": kendalls_w,
                "ranking_stability": describe_ranking_stability(kendalls_w),
                "most_frequent_winner": most_frequent_winner,
                "winner_consistency": winner_consistency,
            }
        )

    return (
        pd.DataFrame(rows, columns=RANKING_STABILITY_COLUMNS)
        .sort_values(["training_run", "training_episode", "opponent_name"])
        .reset_index(drop=True)
    )


def build_seed_stability_overview(
    seed_performance: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> dict[str, object]:
    if seed_performance.empty:
        return {
            "training_runs": [],
            "final_training_episodes": [],
            "model_seeds": [],
            "opponents": [],
            "agents": [],
            "seed_performance_rows": 0,
            "ranking_groups": 0,
            "ranking_groups_with_stability": 0,
        }

    available_rankings = 0
    if not ranking_stability.empty:
        available_rankings = int(ranking_stability["kendalls_w"].notna().sum())

    return {
        "training_runs": sorted(
            str(value) for value in seed_performance["training_run"].unique()
        ),
        "final_training_episodes": sorted(
            int(value) for value in seed_performance["training_episode"].unique()
        ),
        "model_seeds": sorted(
            int(value) for value in seed_performance["model_seed"].unique()
        ),
        "opponents": sorted(
            str(value) for value in seed_performance["opponent_name"].unique()
        ),
        "agents": sorted(
            str(value) for value in seed_performance["agent_name"].unique()
        ),
        "seed_performance_rows": len(seed_performance),
        "ranking_groups": len(ranking_stability),
        "ranking_groups_with_stability": available_rankings,
    }


def generate_seed_stability_findings(
    seed_summary: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> list[str]:
    if seed_summary.empty:
        return ["No seed-level evaluation results were available."]

    findings: list[str] = []
    widest = seed_summary.sort_values(
        SEED_SPREAD_COLUMN,
        ascending=False,
    ).iloc[0]
    findings.append(
        "Largest seed spread: "
        f"{widest['agent_name']} vs {widest['opponent_name']} "
        f"at final training episode {int(widest['training_episode'])} "
        f"({float(widest[SEED_SPREAD_COLUMN]):.3f} BB/game; "
        f"best seed {widest['best_seed']}, "
        f"worst seed {widest['worst_seed']})."
    )

    available = ranking_stability.dropna(subset=["kendalls_w"])
    if available.empty:
        findings.append(
            "Ranking stability could not be calculated because fewer than "
            "the required number of complete seed rankings were available."
        )
    else:
        least_stable = available.sort_values("kendalls_w").iloc[0]
        findings.append(
            "Least stable agent ranking: "
            f"{least_stable['opponent_name']} at final training episode "
            f"{int(least_stable['training_episode'])} "
            f"(Kendall's W={float(least_stable['kendalls_w']):.3f}, "
            f"{least_stable['ranking_stability']})."
        )

    excluded = int(ranking_stability["excluded_seed_count"].sum())
    if excluded:
        findings.append(
            f"Excluded {excluded} incomplete seed ranking(s) from "
            "ranking-agreement calculations."
        )

    return findings


def build_seed_stability_report(
    input_path: str | Path,
    config: SeedStabilityConfig | None = None,
) -> tuple[
    SeedStabilityReport,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = config or SeedStabilityConfig()
    metrics = load_training_opponent_report_data(input_path)
    selected = select_final_model_rows(metrics)
    seed_performance = build_seed_performance(selected)
    seed_rankings = build_seed_rankings(seed_performance)
    seed_summary = build_seed_stability_summary(
        selected,
        seed_performance,
    )
    ranking_stability = build_ranking_stability(
        seed_performance,
        min_complete_seeds=config.min_complete_seeds_for_ranking,
    )
    overview = build_seed_stability_overview(
        seed_performance,
        ranking_stability,
    )
    findings = generate_seed_stability_findings(
        seed_summary,
        ranking_stability,
    )

    report = SeedStabilityReport(
        input_path=str(input_path),
        config=config,
        overview=overview,
        main_findings=findings,
        seed_performance=dataframe_records_with_missing_as_none(seed_performance),
        seed_stability_summary=dataframe_records_with_missing_as_none(seed_summary),
        seed_rankings=dataframe_records_with_missing_as_none(seed_rankings),
        ranking_stability=dataframe_records_with_missing_as_none(ranking_stability),
    )
    return (
        report,
        seed_performance,
        seed_summary,
        seed_rankings,
        ranking_stability,
    )


def _round_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    identifier_columns = {
        "training_episode",
        "model_seed",
        "best_seed",
        "worst_seed",
        "seeds",
        "games",
        "rank_seed_count",
        "first_place_count",
        "seed_count",
        "complete_seed_count",
        "excluded_seed_count",
        "agent_count",
    }
    for column in result.select_dtypes(include="number").columns:
        if column not in identifier_columns:
            result[column] = result[column].round(3)
    return result


def _display_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = _round_numeric_columns(dataframe)
    if "agent_name" in result.columns:
        result["agent_name"] = result["agent_name"].map(display_agent_name)
    return result


def _overview_markdown(overview: dict[str, object]) -> str:
    rows = [{"field": key, "value": value} for key, value in overview.items()]
    return pd.DataFrame(rows).to_markdown(index=False)


def render_seed_stability_markdown(
    report: SeedStabilityReport,
    seed_performance: pd.DataFrame,
    seed_summary: pd.DataFrame,
    seed_rankings: pd.DataFrame,
    ranking_stability: pd.DataFrame,
) -> str:
    findings = "\n".join(
        f"{index + 1}. {finding}" for index, finding in enumerate(report.main_findings)
    )

    return "\n".join(
        [
            "# Seed stability report",
            "",
            (
                "This report analyses performance variation and agent-ranking "
                "agreement across independent model seeds. Every row comes "
                "from the final model of its training run."
            ),
            "",
            (
                "Kendall's W ranges from 0 (no ranking agreement) to 1 "
                "(identical rankings). Incomplete seed rankings are reported "
                "but excluded from W. First-place and winner-consistency "
                "rates also use the 0-to-1 scale."
            ),
            "",
            "## Overview",
            "",
            _overview_markdown(report.overview),
            "",
            "## Main findings",
            "",
            findings,
            "",
            "## Seed stability summary",
            "",
            _display_table(seed_summary).to_markdown(index=False),
            "",
            "## Ranking stability",
            "",
            _display_table(ranking_stability).to_markdown(index=False),
            "",
            "## Per-seed rankings",
            "",
            _display_table(seed_rankings).to_markdown(index=False),
            "",
            "## Per-seed performance",
            "",
            _display_table(seed_performance).to_markdown(index=False),
            "",
        ]
    )


def write_seed_stability_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    config: SeedStabilityConfig | None = None,
    report_format: str = "all",
    export_latex: bool = True,
) -> list[Path]:
    if report_format not in {"markdown", "json", "both", "all"}:
        raise ValueError(
            "Unsupported report_format. Expected one of: markdown, json, both, all."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        report,
        seed_performance,
        seed_summary,
        seed_rankings,
        ranking_stability,
    ) = build_seed_stability_report(input_path, config=config)

    created_paths: list[Path] = []
    if report_format in {"markdown", "both", "all"}:
        markdown_path = output_dir / "seed_stability.md"
        write_text(
            markdown_path,
            render_seed_stability_markdown(
                report,
                seed_performance,
                seed_summary,
                seed_rankings,
                ranking_stability,
            ),
        )
        created_paths.append(markdown_path)

    if report_format in {"json", "both", "all"}:
        json_path = output_dir / "seed_stability.json"
        write_text(
            json_path,
            json.dumps(report.to_dict(), indent=2, allow_nan=False),
        )
        created_paths.append(json_path)

    csv_exports = [
        ("seed_performance.csv", seed_performance),
        ("seed_stability_summary.csv", seed_summary),
        ("seed_rankings.csv", seed_rankings),
        ("ranking_stability.csv", ranking_stability),
    ]
    for filename, dataframe in csv_exports:
        created_paths.append(write_dataframe_csv(dataframe, output_dir / filename))

    if export_latex:
        latex_exports = [
            ("seed_stability_summary.tex", seed_summary),
            ("seed_rankings.tex", seed_rankings),
            ("ranking_stability.tex", ranking_stability),
        ]
        for filename, dataframe in latex_exports:
            created_paths.append(
                write_dataframe_latex(dataframe, output_dir / filename)
            )

    return created_paths

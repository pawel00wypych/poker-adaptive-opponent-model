from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.checkpoint_report import display_agent_name
from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ADAPTIVE_AGENT_TO_ORACLE_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.experiment_summary import (
    load_experiment_summary_data,
    write_dataframe_csv,
    write_dataframe_latex,
)
from src.evaluation.html_utils import write_text
from src.evaluation.plot_utils import ensure_output_dir, save_current_figure

ALGORITHM_MONTE_CARLO = "Monte Carlo"
ALGORITHM_Q_LEARNING = "Q-learning"
ALGORITHM_SARSA = "SARSA"
ALGORITHM_DOUBLE_Q_LEARNING = "Double Q-learning"

ADAPTIVE_AGENT_TO_ALGORITHM = {
    ADAPTIVE_MC_AGENT: ALGORITHM_MONTE_CARLO,
    ADAPTIVE_Q_LEARNING_AGENT: ALGORITHM_Q_LEARNING,
    ADAPTIVE_SARSA_AGENT: ALGORITHM_SARSA,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT: ALGORITHM_DOUBLE_Q_LEARNING,
}

ORACLE_AGENT_TO_ALGORITHM = {
    oracle_agent: ADAPTIVE_AGENT_TO_ALGORITHM[adaptive_agent]
    for adaptive_agent, oracle_agent in ADAPTIVE_AGENT_TO_ORACLE_AGENT.items()
}

ALGORITHM_ORDER = {
    ALGORITHM_MONTE_CARLO: 0,
    ALGORITHM_Q_LEARNING: 1,
    ALGORITHM_SARSA: 2,
    ALGORITHM_DOUBLE_Q_LEARNING: 3,
}

GROUP_COLUMNS = [
    "training_run",
    "opponent_name",
    "checkpoint_episode",
]

ALGORITHM_METRIC_COLUMNS = [
    "training_run",
    "checkpoint_episode",
    "opponent_name",
    "rank",
    "algorithm",
    "agent_name",
    "mean_profit_bb",
    "bb_per_100",
    "win_rate",
    "bust_rate",
    "mean_profit_bb_std_across_seeds",
    "delta_vs_monte_carlo",
    "delta_vs_rule_based",
    "delta_vs_oracle",
]

GLOBAL_RANKING_COLUMNS = [
    "global_rank",
    "algorithm",
    "avg_rank",
    "avg_mean_profit_bb",
    "avg_bb_per_100",
    "avg_win_rate",
    "avg_bust_rate",
    "avg_std_across_seeds_bb",
    "positive_matchup_count",
    "best_matchup_count",
    "evaluated_matchup_count",
]

DELTA_COLUMNS = [
    "training_run",
    "checkpoint_episode",
    "opponent_name",
    "algorithm",
    "mean_profit_bb",
    "delta_vs_monte_carlo",
    "delta_vs_rule_based",
    "delta_vs_oracle",
]

MEAN_PROFIT_BY_OPPONENT_CHART = "algorithm_mean_profit_by_opponent.png"
SEED_STABILITY_BY_OPPONENT_CHART = "algorithm_seed_stability_by_opponent.png"
GLOBAL_MEAN_PROFIT_CHART = "algorithm_global_mean_profit.png"


@dataclass(frozen=True)
class AlgorithmComparisonConfig:
    max_std_across_seeds_bb: float = 5.0
    max_label_length: int = 28


@dataclass(frozen=True)
class AlgorithmComparisonReport:
    input_path: str
    config: AlgorithmComparisonConfig
    overview: dict[str, object]
    main_findings: list[str]
    global_ranking: list[dict[str, object]]
    algorithm_by_opponent: list[dict[str, object]]
    deltas: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "config": asdict(self.config),
            "overview": self.overview,
            "main_findings": self.main_findings,
            "global_ranking": self.global_ranking,
            "algorithm_by_opponent": self.algorithm_by_opponent,
            "deltas": self.deltas,
        }


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include="number").columns:
        if column in {
            "checkpoint_episode",
            "rank",
            "global_rank",
            "positive_matchup_count",
            "best_matchup_count",
            "evaluated_matchup_count",
        }:
            continue
        result[column] = result[column].round(3)
    return result


def _display_table(df: pd.DataFrame) -> pd.DataFrame:
    result = _round_numeric_columns(df)
    if "agent_name" in result.columns:
        result["agent_name"] = result["agent_name"].map(display_agent_name)
    return result


def _format_signed(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.3f}"


def _latest_checkpoint_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "checkpoint_episode" not in rows.columns:
        return rows.copy()

    latest = rows.groupby(["training_run", "opponent_name"])[
        "checkpoint_episode"
    ].transform("max")
    return rows[rows["checkpoint_episode"] == latest].copy()


def build_algorithm_rows(aggregated: pd.DataFrame) -> pd.DataFrame:
    if aggregated.empty:
        result = aggregated.copy()
        result["algorithm"] = pd.Series(dtype="object")
        return result

    result = aggregated[
        aggregated["agent_name"].isin(ADAPTIVE_AGENT_TO_ALGORITHM)
    ].copy()
    result["algorithm"] = result["agent_name"].map(ADAPTIVE_AGENT_TO_ALGORITHM)
    result["algorithm_order"] = result["algorithm"].map(ALGORITHM_ORDER)
    return result.reset_index(drop=True)


def add_algorithm_ranking(algorithm_rows: pd.DataFrame) -> pd.DataFrame:
    if algorithm_rows.empty:
        result = algorithm_rows.copy()
        result["rank"] = pd.Series(dtype="int64")
        return result

    sort_columns = [
        "training_run",
        "opponent_name",
        "checkpoint_episode",
        "mean_profit_bb",
        "win_rate",
        "bust_rate",
        "mean_profit_bb_std_across_seeds",
        "algorithm_order",
    ]
    result = algorithm_rows.sort_values(
        sort_columns,
        ascending=[True, True, True, False, False, True, True, True],
    ).reset_index(drop=True)
    result["rank"] = result.groupby(GROUP_COLUMNS).cumcount() + 1

    front_columns = GROUP_COLUMNS + ["rank", "algorithm", "agent_name"]
    remaining_columns = [
        column for column in result.columns
        if column not in front_columns
    ]
    return result[front_columns + remaining_columns]


def _merge_baseline_delta(
    rows: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    column_name: str,
    baseline_column_name: str,
) -> pd.DataFrame:
    result = rows.copy()
    baseline_values = baseline[GROUP_COLUMNS + ["mean_profit_bb"]].rename(
        columns={"mean_profit_bb": baseline_column_name}
    )
    result = result.merge(baseline_values, on=GROUP_COLUMNS, how="left")
    result[column_name] = result["mean_profit_bb"] - result[baseline_column_name]
    return result.drop(columns=[baseline_column_name])


def add_algorithm_deltas(
    ranking: pd.DataFrame,
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    result = ranking.copy()

    monte_carlo = result[result["algorithm"] == ALGORITHM_MONTE_CARLO]
    result = _merge_baseline_delta(
        result,
        monte_carlo,
        column_name="delta_vs_monte_carlo",
        baseline_column_name="monte_carlo_mean_profit_bb",
    )

    rule_based = aggregated[aggregated["agent_name"] == RULE_BASED_AGENT]
    result = _merge_baseline_delta(
        result,
        rule_based,
        column_name="delta_vs_rule_based",
        baseline_column_name="rule_based_mean_profit_bb",
    )

    oracle = aggregated[
        aggregated["agent_name"].isin(ORACLE_AGENT_TO_ALGORITHM)
    ].copy()
    if oracle.empty:
        result["delta_vs_oracle"] = pd.NA
        return result

    oracle["algorithm"] = oracle["agent_name"].map(ORACLE_AGENT_TO_ALGORITHM)
    oracle_values = oracle[
        GROUP_COLUMNS + ["algorithm", "mean_profit_bb"]
    ].rename(columns={"mean_profit_bb": "oracle_mean_profit_bb"})

    result = result.merge(
        oracle_values,
        on=GROUP_COLUMNS + ["algorithm"],
        how="left",
    )
    result["delta_vs_oracle"] = (
        result["mean_profit_bb"] - result["oracle_mean_profit_bb"]
    )
    return result.drop(columns=["oracle_mean_profit_bb"])


def build_global_algorithm_ranking(algorithm_by_opponent: pd.DataFrame) -> pd.DataFrame:
    if algorithm_by_opponent.empty:
        return pd.DataFrame(columns=GLOBAL_RANKING_COLUMNS)

    grouped = (
        algorithm_by_opponent.groupby("algorithm")
        .agg(
            avg_rank=("rank", "mean"),
            avg_mean_profit_bb=("mean_profit_bb", "mean"),
            avg_bb_per_100=("bb_per_100", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_bust_rate=("bust_rate", "mean"),
            avg_std_across_seeds_bb=(
                "mean_profit_bb_std_across_seeds",
                "mean",
            ),
            positive_matchup_count=(
                "mean_profit_bb",
                lambda values: int((values >= 0.0).sum()),
            ),
            best_matchup_count=("rank", lambda values: int((values == 1).sum())),
            evaluated_matchup_count=("mean_profit_bb", "count"),
        )
        .reset_index()
    )
    grouped["algorithm_order"] = grouped["algorithm"].map(ALGORITHM_ORDER)
    grouped = grouped.sort_values(
        [
            "avg_mean_profit_bb",
            "positive_matchup_count",
            "best_matchup_count",
            "avg_rank",
            "avg_std_across_seeds_bb",
            "algorithm_order",
        ],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    grouped["global_rank"] = grouped.index + 1

    return grouped[_existing_columns(grouped, GLOBAL_RANKING_COLUMNS)]


def build_algorithm_overview(
    metrics: pd.DataFrame,
    algorithm_by_opponent: pd.DataFrame,
) -> dict[str, object]:
    if metrics.empty:
        return {
            "training_runs": [],
            "checkpoints": [],
            "seeds": [],
            "opponents": [],
            "algorithms": [],
            "source_raw_games": 0,
            "algorithm_summary_rows": 0,
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
        "algorithms": sorted(
            algorithm_by_opponent["algorithm"].dropna().unique(),
            key=lambda value: ALGORITHM_ORDER.get(str(value), 999),
        )
        if not algorithm_by_opponent.empty
        else [],
        "source_raw_games": int(metrics["games"].sum())
        if "games" in metrics.columns
        else 0,
        "algorithm_summary_rows": int(len(algorithm_by_opponent)),
    }


def generate_algorithm_findings(
    global_ranking: pd.DataFrame,
    algorithm_by_opponent: pd.DataFrame,
    config: AlgorithmComparisonConfig,
) -> list[str]:
    findings: list[str] = []

    if not global_ranking.empty:
        best = global_ranking.iloc[0]
        findings.append(
            "Best adaptive RL algorithm by average mean profit is "
            f"{best['algorithm']} with "
            f"{best['avg_mean_profit_bb']:.3f} BB/game."
        )

    if not algorithm_by_opponent.empty:
        positive = (
            algorithm_by_opponent.groupby("algorithm")["mean_profit_bb"]
            .apply(lambda values: int((values >= 0.0).sum()))
            .sort_values(ascending=False)
        )
        evaluated = algorithm_by_opponent.groupby("algorithm")[
            "mean_profit_bb"
        ].count()
        if not positive.empty:
            leader = positive.index[0]
            findings.append(
                f"{leader} has non-negative mean profit in "
                f"{int(positive.iloc[0])}/{int(evaluated.loc[leader])} "
                "evaluated matchups."
            )

    if "delta_vs_monte_carlo" in algorithm_by_opponent.columns:
        non_mc = algorithm_by_opponent[
            algorithm_by_opponent["algorithm"] != ALGORITHM_MONTE_CARLO
        ].dropna(subset=["delta_vs_monte_carlo"])
        if not non_mc.empty:
            avg_deltas = (
                non_mc.groupby("algorithm")["delta_vs_monte_carlo"]
                .mean()
                .sort_values(ascending=False)
            )
            best_algorithm = str(avg_deltas.index[0])
            findings.append(
                f"Largest average improvement over Monte Carlo is achieved by "
                f"{best_algorithm} "
                f"({_format_signed(float(avg_deltas.iloc[0]))} BB/game)."
            )

    unstable = algorithm_by_opponent[
        algorithm_by_opponent["mean_profit_bb_std_across_seeds"]
        > config.max_std_across_seeds_bb
    ]
    if not unstable.empty:
        examples = unstable.sort_values(
            "mean_profit_bb_std_across_seeds",
            ascending=False,
        ).head(3)
        labels = [
            f"{row.algorithm} vs {row.opponent_name} "
            f"({row.mean_profit_bb_std_across_seeds:.3f})"
            for row in examples.itertuples(index=False)
        ]
        findings.append(
            f"High seed variance detected in {len(unstable)} algorithm rows; "
            "largest examples: " + "; ".join(labels) + "."
        )

    if not algorithm_by_opponent.empty:
        best_by_opponent = algorithm_by_opponent[
            algorithm_by_opponent["rank"] == 1
        ]
        if not best_by_opponent.empty:
            winners = [
                f"{row.opponent_name}: {row.algorithm}"
                for row in best_by_opponent.sort_values(
                    ["opponent_name", "checkpoint_episode"]
                ).itertuples(index=False)
            ]
            findings.append(
                "Best algorithm by opponent: " + "; ".join(winners) + "."
            )

    return findings


def build_algorithm_comparison(
    input_path: str | Path,
    config: AlgorithmComparisonConfig | None = None,
) -> tuple[
    AlgorithmComparisonReport,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = config or AlgorithmComparisonConfig()
    metrics, aggregated = load_experiment_summary_data(input_path)

    algorithm_rows = build_algorithm_rows(aggregated)
    algorithm_ranking = add_algorithm_ranking(algorithm_rows)
    algorithm_by_opponent = add_algorithm_deltas(algorithm_ranking, aggregated)
    global_ranking = build_global_algorithm_ranking(algorithm_by_opponent)
    deltas = algorithm_by_opponent[_existing_columns(algorithm_by_opponent, DELTA_COLUMNS)]

    overview = build_algorithm_overview(metrics, algorithm_by_opponent)
    main_findings = generate_algorithm_findings(
        global_ranking,
        algorithm_by_opponent,
        config,
    )

    report = AlgorithmComparisonReport(
        input_path=str(input_path),
        config=config,
        overview=overview,
        main_findings=main_findings,
        global_ranking=global_ranking.to_dict(orient="records"),
        algorithm_by_opponent=algorithm_by_opponent.to_dict(orient="records"),
        deltas=deltas.to_dict(orient="records"),
    )

    return report, global_ranking, algorithm_by_opponent, deltas


def _overview_markdown(overview: dict[str, object]) -> str:
    rows = [{"field": key, "value": value} for key, value in overview.items()]
    return pd.DataFrame(rows).to_markdown(index=False)


def _chart_markdown(chart_paths: list[Path]) -> str:
    if not chart_paths:
        return "No charts generated."

    return "\n\n".join(
        f"![{path.stem}](charts/{path.name})"
        for path in chart_paths
    )


def render_algorithm_comparison_markdown(
    report: AlgorithmComparisonReport,
    global_ranking: pd.DataFrame,
    algorithm_by_opponent: pd.DataFrame,
    deltas: pd.DataFrame,
    chart_paths: list[Path] | None = None,
) -> str:
    findings = "\n".join(
        f"{index + 1}. {finding}"
        for index, finding in enumerate(report.main_findings)
    )

    global_table = global_ranking[
        _existing_columns(global_ranking, GLOBAL_RANKING_COLUMNS)
    ]
    opponent_table = algorithm_by_opponent[
        _existing_columns(algorithm_by_opponent, ALGORITHM_METRIC_COLUMNS)
    ]
    delta_table = deltas[_existing_columns(deltas, DELTA_COLUMNS)]

    return "\n".join(
        [
            "# RL algorithm comparison",
            "",
            "This report compares adaptive tabular reinforcement-learning "
            "algorithms evaluated in the same poker environment.",
            "",
            "Compared algorithms:",
            "",
            "- Monte Carlo",
            "- Q-learning",
            "- SARSA",
            "- Double Q-learning",
            "",
            "## Overview",
            "",
            _overview_markdown(report.overview),
            "",
            "## Main findings",
            "",
            findings,
            "",
            "## Charts",
            "",
            _chart_markdown(chart_paths or []),
            "",
            "## Global algorithm ranking",
            "",
            _display_table(global_table).to_markdown(index=False),
            "",
            "## Algorithm ranking by opponent and checkpoint",
            "",
            _display_table(opponent_table).to_markdown(index=False),
            "",
            "## Delta vs baselines",
            "",
            _display_table(delta_table).to_markdown(index=False),
            "",
        ]
    )


def _truncate_label(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "…"


def _sort_algorithm_labels(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["algorithm_order"] = result["algorithm"].map(ALGORITHM_ORDER)
    return result.sort_values(
        ["opponent_name", "algorithm_order"]
    ).reset_index(drop=True)


def plot_algorithm_mean_profit_by_opponent(
    algorithm_by_opponent: pd.DataFrame,
    output_path: str | Path,
    config: AlgorithmComparisonConfig | None = None,
) -> Path | None:
    config = config or AlgorithmComparisonConfig()
    data = _latest_checkpoint_rows(algorithm_by_opponent)

    if data.empty or "mean_profit_bb" not in data.columns:
        return None

    data = _sort_algorithm_labels(data)
    data["plot_label"] = data.apply(
        lambda row: _truncate_label(
            f"{row['opponent_name']}\n{row['algorithm']}",
            config.max_label_length,
        ),
        axis=1,
    )

    fig_width = max(10.0, min(24.0, 0.65 * len(data)))
    plt.figure(figsize=(fig_width, 6.0))
    x_positions = list(range(len(data)))
    plt.bar(x_positions, data["mean_profit_bb"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, data["plot_label"], rotation=45, ha="right")
    plt.ylabel("Mean profit per game [BB]")
    plt.title("Adaptive RL algorithm mean profit by opponent")
    plt.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    save_current_figure(output_path)
    return output_path


def plot_algorithm_seed_stability_by_opponent(
    algorithm_by_opponent: pd.DataFrame,
    output_path: str | Path,
    config: AlgorithmComparisonConfig | None = None,
) -> Path | None:
    config = config or AlgorithmComparisonConfig()
    data = _latest_checkpoint_rows(algorithm_by_opponent)

    if data.empty or "mean_profit_bb_std_across_seeds" not in data.columns:
        return None

    data = _sort_algorithm_labels(data)
    data["plot_label"] = data.apply(
        lambda row: _truncate_label(
            f"{row['opponent_name']}\n{row['algorithm']}",
            config.max_label_length,
        ),
        axis=1,
    )

    fig_width = max(10.0, min(24.0, 0.65 * len(data)))
    plt.figure(figsize=(fig_width, 6.0))
    x_positions = list(range(len(data)))
    plt.bar(x_positions, data["mean_profit_bb_std_across_seeds"])
    plt.axhline(
        config.max_std_across_seeds_bb,
        linestyle="--",
        linewidth=1,
        label=(
            "Warning threshold "
            f"({config.max_std_across_seeds_bb:.1f} BB/game)"
        ),
    )
    plt.xticks(x_positions, data["plot_label"], rotation=45, ha="right")
    plt.ylabel("Std across seeds [BB/game]")
    plt.title("Adaptive RL algorithm seed stability by opponent")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    save_current_figure(output_path)
    return output_path


def plot_algorithm_global_mean_profit(
    global_ranking: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    if global_ranking.empty or "avg_mean_profit_bb" not in global_ranking.columns:
        return None

    data = global_ranking.sort_values("global_rank").reset_index(drop=True)
    plt.figure(figsize=(9.0, 5.0))
    x_positions = list(range(len(data)))
    plt.bar(x_positions, data["avg_mean_profit_bb"])
    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, data["algorithm"], rotation=30, ha="right")
    plt.ylabel("Average mean profit per game [BB]")
    plt.title("Global adaptive RL algorithm ranking")
    plt.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    save_current_figure(output_path)
    return output_path


def create_algorithm_comparison_charts(
    global_ranking: pd.DataFrame,
    algorithm_by_opponent: pd.DataFrame,
    output_dir: str | Path,
    config: AlgorithmComparisonConfig | None = None,
) -> list[Path]:
    config = config or AlgorithmComparisonConfig()
    output_dir = ensure_output_dir(output_dir)

    chart_specs = [
        (
            MEAN_PROFIT_BY_OPPONENT_CHART,
            lambda path: plot_algorithm_mean_profit_by_opponent(
                algorithm_by_opponent,
                path,
                config,
            ),
        ),
        (
            SEED_STABILITY_BY_OPPONENT_CHART,
            lambda path: plot_algorithm_seed_stability_by_opponent(
                algorithm_by_opponent,
                path,
                config,
            ),
        ),
        (
            GLOBAL_MEAN_PROFIT_CHART,
            lambda path: plot_algorithm_global_mean_profit(
                global_ranking,
                path,
            ),
        ),
    ]

    created_paths: list[Path] = []
    for filename, plot_function in chart_specs:
        path = plot_function(output_dir / filename)
        if path is not None:
            created_paths.append(path)

    return created_paths


def write_algorithm_comparison_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    config: AlgorithmComparisonConfig | None = None,
    report_format: str = "all",
    export_latex: bool = True,
    include_charts: bool = True,
) -> list[Path]:
    if report_format not in {"markdown", "json", "both", "all"}:
        raise ValueError(
            "Unsupported report_format. Expected one of: "
            "markdown, json, both, all."
        )

    config = config or AlgorithmComparisonConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, global_ranking, algorithm_by_opponent, deltas = (
        build_algorithm_comparison(input_path, config=config)
    )

    created_paths: list[Path] = []
    chart_paths: list[Path] = []

    if include_charts:
        chart_paths = create_algorithm_comparison_charts(
            global_ranking=global_ranking,
            algorithm_by_opponent=algorithm_by_opponent,
            output_dir=output_dir / "charts",
            config=config,
        )
        created_paths.extend(chart_paths)

    if report_format in {"markdown", "both", "all"}:
        markdown_path = output_dir / "algorithm_comparison.md"
        write_text(
            markdown_path,
            render_algorithm_comparison_markdown(
                report=report,
                global_ranking=global_ranking,
                algorithm_by_opponent=algorithm_by_opponent,
                deltas=deltas,
                chart_paths=chart_paths,
            ),
        )
        created_paths.append(markdown_path)

    if report_format in {"json", "both", "all"}:
        json_path = output_dir / "algorithm_comparison.json"
        write_text(json_path, json.dumps(report.to_dict(), indent=2))
        created_paths.append(json_path)

    csv_exports = [
        ("algorithm_global_ranking.csv", global_ranking),
        ("algorithm_by_opponent.csv", algorithm_by_opponent),
        ("algorithm_deltas.csv", deltas),
    ]
    for filename, df in csv_exports:
        created_paths.append(write_dataframe_csv(df, output_dir / filename))

    if export_latex:
        latex_exports = [
            ("algorithm_global_ranking.tex", global_ranking),
            ("algorithm_by_opponent.tex", algorithm_by_opponent),
            ("algorithm_deltas.tex", deltas),
        ]
        for filename, df in latex_exports:
            created_paths.append(write_dataframe_latex(df, output_dir / filename))

    return created_paths

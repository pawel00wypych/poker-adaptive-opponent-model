from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.algorithm_metadata import ADAPTIVE_AGENTS
from src.evaluation.constants import ORACLE_AGENTS
from src.evaluation.metrics.learning_curve_metrics import (
    calculate_learning_curve_metrics,
)
from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_MARGIN_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_MAX_COLUMN,
    SEED_MIN_COLUMN,
    SEED_SPREAD_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
    add_seed_level_statistical_summary,
)
from src.evaluation.reporting.html_utils import definition_list, html_page, write_text
from src.evaluation.reporting.plot_utils import ensure_output_dir, save_current_figure
from src.evaluation.reporting.report_descriptions import (
    AGENT_LABELS,
    METRIC_DESCRIPTIONS,
    REPORT_INTRODUCTION,
)

SUMMARY_COLUMNS = [
    "agent_name",
    "opponent_name",
    "checkpoint_episode",
    "model_seed",
    "mean_profit_bb",
    "bb_per_100",
    "win_rate",
    "bust_rate",
    "global_classifier_accuracy",
    "global_classifier_coverage",
]

AGGREGATION_COLUMNS = [
    "training_run",
    "agent_name",
    "opponent_name",
    "checkpoint_episode",
]

PLOT_METRICS = [
    ("mean_profit_bb", "Mean profit per game [BB]"),
    ("bb_per_100", "BB per 100 hands"),
    ("win_rate", "Win rate [%]"),
    ("bust_rate", "Bust rate [%]"),
]

CLASSIFIER_PLOT_METRICS = [
    ("global_classifier_accuracy", "Global classifier accuracy [%]"),
    ("global_classifier_coverage", "Global classifier coverage [%]"),
]


def display_agent_name(agent_name: str) -> str:
    return AGENT_LABELS.get(agent_name, agent_name)


def load_learning_curve_report_data(
    input_path: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> pd.DataFrame:
    df = calculate_learning_curve_metrics(str(input_path))

    if opponent is not None:
        df = df[df["opponent_name"] == opponent]

    if agent is not None:
        df = df[df["agent_name"] == agent]

    return df.sort_values(
        [
            "opponent_name",
            "agent_name",
            "checkpoint_episode",
            "model_seed",
        ]
    ).reset_index(drop=True)


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    aggregated = (
        df.groupby(AGGREGATION_COLUMNS)
        .agg(
            seeds=("model_seed", "nunique"),
            games=("games", "sum"),
            mean_profit_bb=("mean_profit_bb", "mean"),
            mean_profit_bb_std_across_seeds=("mean_profit_bb", "std"),
            mean_profit_bb_min_across_seeds=("mean_profit_bb", "min"),
            mean_profit_bb_max_across_seeds=("mean_profit_bb", "max"),
            bb_per_100=("bb_per_100", "mean"),
            win_rate=("win_rate", "mean"),
            bust_rate=("bust_rate", "mean"),
            global_classifier_accuracy=("global_classifier_accuracy", "mean"),
            global_classifier_coverage=("global_classifier_coverage", "mean"),
            mean_policy_switches=("mean_policy_switches", "mean"),
        )
        .reset_index()
    )

    aggregated = add_seed_level_statistical_summary(aggregated)

    return aggregated.sort_values(
        [
            "opponent_name",
            "agent_name",
            "checkpoint_episode",
        ]
    ).reset_index(drop=True)


def best_rows_by_agent(aggregated: pd.DataFrame) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated.copy()

    indexes = aggregated.groupby(
        [
            "agent_name",
            "opponent_name",
        ]
    )["mean_profit_bb"].idxmax()

    return aggregated.loc[indexes].sort_values(
        [
            "opponent_name",
            "mean_profit_bb",
        ],
        ascending=[True, False],
    )


def format_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is not None:
        existing_columns = [column for column in columns if column in df.columns]
        df = df[existing_columns].copy()
    else:
        df = df.copy()

    if "agent_name" in df.columns:
        df["agent_name"] = df["agent_name"].map(display_agent_name)

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if column in {"checkpoint_episode", "model_seed", "games", "seeds"}:
            continue
        df[column] = df[column].round(3)

    return df


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    return format_table(df).to_html(index=False, escape=True)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    return format_table(df).to_markdown(index=False)


def plot_metric_by_checkpoint(
    aggregated: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(9, 5))

    for agent_name, group in aggregated.groupby("agent_name"):
        group = group.sort_values("checkpoint_episode")
        plt.plot(
            group["checkpoint_episode"],
            group[metric],
            marker="o",
            label=display_agent_name(agent_name),
        )

    plt.xlabel("Checkpoint episode")
    plt.ylabel(ylabel)
    plt.title(ylabel + " by checkpoint")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_current_figure(output_path)


def create_learning_curve_plots(
    aggregated: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    plots: list[Path] = []

    if aggregated.empty:
        return plots

    for metric, ylabel in PLOT_METRICS:
        if metric not in aggregated.columns:
            continue
        path = output_dir / f"checkpoint_{metric}.png"
        plot_metric_by_checkpoint(aggregated, metric, ylabel, path)
        plots.append(path)

    classifier_df = aggregated[
        aggregated["agent_name"].isin([*ADAPTIVE_AGENTS, *ORACLE_AGENTS])
    ]

    for metric, ylabel in CLASSIFIER_PLOT_METRICS:
        if metric not in classifier_df.columns or classifier_df.empty:
            continue
        path = output_dir / f"checkpoint_{metric}.png"
        plot_metric_by_checkpoint(classifier_df, metric, ylabel, path)
        plots.append(path)

    return plots


def metric_glossary_html() -> str:
    relevant_metrics = [
        "mean_profit_bb",
        "bb_per_100",
        "win_rate",
        "bust_rate",
        "standard_error",
        "ci_95_lower",
        "ci_95_upper",
        "mean_profit_bb_std_across_seeds",
        SEED_STANDARD_ERROR_COLUMN,
        SEED_CI_LOWER_COLUMN,
        SEED_CI_UPPER_COLUMN,
        SEED_CI_MARGIN_COLUMN,
        SEED_MIN_COLUMN,
        SEED_MAX_COLUMN,
        SEED_SPREAD_COLUMN,
        "global_classifier_accuracy",
        "global_classifier_coverage",
        "mean_policy_switches",
    ]
    return definition_list(
        (metric, METRIC_DESCRIPTIONS[metric]) for metric in relevant_metrics
    )


def metric_glossary_markdown() -> str:
    relevant_metrics = [
        "mean_profit_bb",
        "bb_per_100",
        "win_rate",
        "bust_rate",
        "standard_error",
        "ci_95_lower",
        "ci_95_upper",
        "mean_profit_bb_std_across_seeds",
        SEED_STANDARD_ERROR_COLUMN,
        SEED_CI_LOWER_COLUMN,
        SEED_CI_UPPER_COLUMN,
        SEED_CI_MARGIN_COLUMN,
        SEED_MIN_COLUMN,
        SEED_MAX_COLUMN,
        SEED_SPREAD_COLUMN,
        "global_classifier_accuracy",
        "global_classifier_coverage",
        "mean_policy_switches",
    ]
    return "\n".join(
        f"- **{metric}**: {METRIC_DESCRIPTIONS[metric]}" for metric in relevant_metrics
    )


def write_learning_curve_html_report(
    input_path: str | Path,
    output_dir: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    plots_dir = ensure_output_dir(output_dir / "plots")

    metrics_df = load_learning_curve_report_data(input_path, opponent, agent)
    aggregated = aggregate_across_seeds(metrics_df)
    best = best_rows_by_agent(aggregated)
    plots = create_learning_curve_plots(aggregated, plots_dir)

    plot_html = "\n".join(
        f'<div class="plot"><img src="plots/{plot.name}" alt="{plot.stem}"></div>'
        for plot in plots
    )

    body = f"""
<h1>Learning-curve analysis report</h1>
<p class="note">{REPORT_INTRODUCTION}</p>
<p>This diagnostic evaluates intermediate checkpoints to show learning
progress. Its rows must not be merged into final-model benchmark reports.</p>
<h2>Filters</h2>
<ul>
  <li><strong>Input file:</strong> <code>{Path(input_path)}</code></li>
  <li><strong>Opponent:</strong> {opponent or "all"}</li>
  <li><strong>Agent:</strong> {agent or "all"}</li>
</ul>
<h2>Metric glossary</h2>
{metric_glossary_html()}
<h2>Best checkpoint per agent and opponent</h2>
{dataframe_to_html_table(best)}
<h2>Aggregated results across seeds</h2>
{dataframe_to_html_table(aggregated)}
<h2>Plots</h2>
<div class="grid">
{plot_html}
</div>
<h2>Per-seed results</h2>
{dataframe_to_html_table(metrics_df[SUMMARY_COLUMNS])}
"""

    output_path = output_dir / "learning_curve_report.html"
    write_text(output_path, html_page("Learning-curve analysis report", body))
    return output_path


def write_learning_curve_markdown_report(
    input_path: str | Path,
    output_dir: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    plots_dir = ensure_output_dir(output_dir / "plots")

    metrics_df = load_learning_curve_report_data(input_path, opponent, agent)
    aggregated = aggregate_across_seeds(metrics_df)
    best = best_rows_by_agent(aggregated)
    plots = create_learning_curve_plots(aggregated, plots_dir)

    plot_markdown = "\n".join(f"![{plot.stem}](plots/{plot.name})" for plot in plots)

    text = f"""# Learning-curve analysis report

{REPORT_INTRODUCTION}

This diagnostic evaluates intermediate checkpoints to show learning progress.
Its rows must not be merged into final-model benchmark reports.

## Filters

- **Input file:** `{Path(input_path)}`
- **Opponent:** {opponent or "all"}
- **Agent:** {agent or "all"}

## Metric glossary

{metric_glossary_markdown()}

## Best checkpoint per agent and opponent

{dataframe_to_markdown_table(best)}

## Aggregated results across seeds

{dataframe_to_markdown_table(aggregated)}

## Plots

{plot_markdown}

## Per-seed results

{dataframe_to_markdown_table(metrics_df[SUMMARY_COLUMNS])}
"""

    output_path = output_dir / "learning_curve_report.md"
    write_text(output_path, text)
    return output_path

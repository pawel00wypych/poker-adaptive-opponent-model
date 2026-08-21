from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics.evaluation_metrics import (
    calculate_final_model_metrics,
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
from src.evaluation.reporting.plot_utils import ensure_output_dir
from src.evaluation.reporting.report_descriptions import (
    AGENT_LABELS,
    METRIC_DESCRIPTIONS,
    REPORT_INTRODUCTION,
)

SUMMARY_COLUMNS = [
    "agent_name",
    "opponent_name",
    "training_episode",
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
    "training_episode",
]


def display_agent_name(agent_name: str) -> str:
    return AGENT_LABELS.get(agent_name, agent_name)


def load_training_opponent_report_data(
    input_path: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> pd.DataFrame:
    df = calculate_final_model_metrics(str(input_path))
    if opponent is not None:
        df = df[df["opponent_name"] == opponent]
    if agent is not None:
        df = df[df["agent_name"] == agent]
    return df.sort_values(
        ["opponent_name", "agent_name", "training_episode", "model_seed"]
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
        ["opponent_name", "agent_name", "training_episode"]
    ).reset_index(drop=True)


def format_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    existing_columns = (
        [column for column in columns if column in df.columns]
        if columns is not None
        else list(df.columns)
    )
    result = df[existing_columns].copy()
    if "agent_name" in result.columns:
        result["agent_name"] = result["agent_name"].map(display_agent_name)
    for column in result.select_dtypes(include="number").columns:
        if column in {"training_episode", "model_seed", "games", "seeds"}:
            continue
        result[column] = result[column].round(3)
    return result


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    return format_table(df).to_html(index=False, escape=True)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    return format_table(df).to_markdown(index=False)


def _relevant_metrics() -> list[str]:
    return [
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


def metric_glossary_html() -> str:
    return definition_list(
        (metric, METRIC_DESCRIPTIONS[metric]) for metric in _relevant_metrics()
    )


def metric_glossary_markdown() -> str:
    return "\n".join(
        f"- **{metric}**: {METRIC_DESCRIPTIONS[metric]}"
        for metric in _relevant_metrics()
    )


def write_training_opponent_html_report(
    input_path: str | Path,
    output_dir: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    metrics = load_training_opponent_report_data(input_path, opponent, agent)
    aggregated = aggregate_across_seeds(metrics)
    body = f"""
<h1>Training-opponent evaluation report</h1>
<p class="note">{REPORT_INTRODUCTION}</p>
<p>Only final trained models are included. <code>training_episode</code>
records the completed training budget; checkpoints belong to learning-curve
analysis.</p>
<h2>Filters</h2>
<ul>
  <li><strong>Input file:</strong> <code>{Path(input_path)}</code></li>
  <li><strong>Opponent:</strong> {opponent or "all"}</li>
  <li><strong>Agent:</strong> {agent or "all"}</li>
</ul>
<h2>Metric glossary</h2>
{metric_glossary_html()}
<h2>Aggregated final-model results across seeds</h2>
{dataframe_to_html_table(aggregated)}
<h2>Per-seed final-model results</h2>
{dataframe_to_html_table(metrics[SUMMARY_COLUMNS])}
"""
    output_path = output_dir / "training_opponent_report.html"
    write_text(output_path, html_page("Training-opponent evaluation report", body))
    return output_path


def write_training_opponent_markdown_report(
    input_path: str | Path,
    output_dir: str | Path,
    opponent: str | None = None,
    agent: str | None = None,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    metrics = load_training_opponent_report_data(input_path, opponent, agent)
    aggregated = aggregate_across_seeds(metrics)
    text = f"""# Training-opponent evaluation report

{REPORT_INTRODUCTION}

Only final trained models are included. `training_episode` records the
completed training budget; checkpoints belong to learning-curve analysis.

## Filters

- **Input file:** `{Path(input_path)}`
- **Opponent:** {opponent or "all"}
- **Agent:** {agent or "all"}

## Metric glossary

{metric_glossary_markdown()}

## Aggregated final-model results across seeds

{dataframe_to_markdown_table(aggregated)}

## Per-seed final-model results

{dataframe_to_markdown_table(metrics[SUMMARY_COLUMNS])}
"""
    output_path = output_dir / "training_opponent_report.md"
    write_text(output_path, text)
    return output_path

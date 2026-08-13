from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.reporting.html_utils import definition_list, html_page, write_text
from src.evaluation.reporting.plot_utils import ensure_output_dir, save_current_figure
from src.evaluation.reporting.report_descriptions import (
    ACTION_LABELS,
    METRIC_DESCRIPTIONS,
    REPORT_INTRODUCTION,
    STATE_FIELD_DESCRIPTIONS,
)


def load_q_table_comparison(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summaries_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for summary in report.get("summaries", []):
        row = {
            "name": summary["name"],
            "states": summary["states"],
            "fully_zero_states": summary["fully_zero_states"],
            "fully_zero_rate": summary["fully_zero_rate"],
            "tied_best_states": summary["tied_best_states"],
            "tied_best_rate": summary["tied_best_rate"],
        }

        for action, count in summary.get("best_action_counts", {}).items():
            row[f"best_{action}_count"] = count

        for action, rate in summary.get("best_action_rates", {}).items():
            row[f"best_{action}_rate"] = rate

        for action_stats in summary.get("action_stats", []):
            action = action_stats["action"]
            row[f"q_{action}_mean"] = action_stats["mean_q"]
            row[f"q_{action}_median"] = action_stats["median_q"]
            row[f"q_{action}_std"] = action_stats["std_q"]
            row[f"q_{action}_min"] = action_stats["min_q"]
            row[f"q_{action}_max"] = action_stats["max_q"]
            row[f"q_{action}_zero_rate"] = action_stats["zero_rate"]

        rows.append(row)

    return pd.DataFrame(rows)


def comparisons_to_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for comparison in report.get("comparisons", []):
        row = {
            "left_name": comparison["left_name"],
            "right_name": comparison["right_name"],
            "left_states": comparison["left_states"],
            "right_states": comparison["right_states"],
            "common_states": comparison["common_states"],
            "best_action_agreement_rate": comparison[
                "best_action_agreement_rate"
            ],
            "mean_max_abs_q_delta": comparison["mean_max_abs_q_delta"],
        }

        for action, value in comparison.get(
            "mean_abs_q_delta_by_action", {}
        ).items():
            row[f"mean_abs_q_delta_{action}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if column.endswith("states") or column.endswith("count"):
            continue
        df[column] = df[column].round(3)
    return df


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    return format_table(df).to_html(index=False, escape=True)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    return format_table(df).to_markdown(index=False)


def q_metric_glossary_html() -> str:
    metric_names = [
        "states",
        "fully_zero_states",
        "tied_best_states",
        "best_action_agreement_rate",
        "mean_max_abs_q_delta",
    ]
    return definition_list(
        (metric, METRIC_DESCRIPTIONS[metric])
        for metric in metric_names
    )


def q_metric_glossary_markdown() -> str:
    metric_names = [
        "states",
        "fully_zero_states",
        "tied_best_states",
        "best_action_agreement_rate",
        "mean_max_abs_q_delta",
    ]
    return "\n".join(
        f"- **{metric}**: {METRIC_DESCRIPTIONS[metric]}"
        for metric in metric_names
    )


def state_glossary_html() -> str:
    return definition_list(STATE_FIELD_DESCRIPTIONS.items())


def state_glossary_markdown() -> str:
    return "\n".join(
        f"- **{field}**: {description}"
        for field, description in STATE_FIELD_DESCRIPTIONS.items()
    )


def plot_best_action_distribution(
    summaries: pd.DataFrame,
    output_path: str | Path,
) -> None:
    columns = ["best_fold_rate", "best_call_rate", "best_raise_rate"]
    available_columns = [column for column in columns if column in summaries.columns]

    if summaries.empty or not available_columns:
        return

    plot_df = summaries.set_index("name")[available_columns]
    plot_df.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.xlabel("Model")
    plt.ylabel("Best action share [%]")
    plt.title("Best-action distribution in Q-tables")
    plt.legend([ACTION_LABELS.get(column.split("_")[1], column) for column in available_columns])
    save_current_figure(output_path)


def plot_state_counts(
    summaries: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if summaries.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(summaries["name"], summaries["states"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Unique states")
    plt.title("Q-table state coverage")
    save_current_figure(output_path)


def plot_mean_q_by_action(
    summaries: pd.DataFrame,
    output_path: str | Path,
) -> None:
    columns = ["q_fold_mean", "q_call_mean", "q_raise_mean"]
    available_columns = [column for column in columns if column in summaries.columns]

    if summaries.empty or not available_columns:
        return

    plot_df = summaries.set_index("name")[available_columns]
    plot_df.plot(kind="bar", figsize=(10, 5))
    plt.xlabel("Model")
    plt.ylabel("Mean Q-value")
    plt.title("Mean Q-value by action")
    plt.legend([ACTION_LABELS.get(column.split("_")[1], column) for column in available_columns])
    save_current_figure(output_path)


def plot_agreement_matrix(
    comparisons: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if comparisons.empty:
        return

    labels = sorted(
        set(comparisons["left_name"]).union(set(comparisons["right_name"]))
    )
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for label in labels:
        matrix.loc[label, label] = 100.0

    for _, row in comparisons.iterrows():
        left = row["left_name"]
        right = row["right_name"]
        value = row["best_action_agreement_rate"]
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value

    plt.figure(figsize=(8, 6))
    image = plt.imshow(matrix.fillna(0.0), aspect="auto", vmin=0, vmax=100)
    plt.colorbar(image, label="Agreement [%]")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Best-action agreement matrix")
    save_current_figure(output_path)


def create_q_table_plots(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    plots = [
        output_dir / "q_table_best_action_distribution.png",
        output_dir / "q_table_state_counts.png",
        output_dir / "q_table_mean_q_by_action.png",
        output_dir / "q_table_agreement_matrix.png",
    ]

    plot_best_action_distribution(summaries, plots[0])
    plot_state_counts(summaries, plots[1])
    plot_mean_q_by_action(summaries, plots[2])
    plot_agreement_matrix(comparisons, plots[3])

    return [plot for plot in plots if plot.exists()]


def extract_largest_disagreement_rows(
    report: dict[str, Any],
    limit: int,
) -> pd.DataFrame:
    rows = []

    for pair_name, disagreements in report.get("largest_disagreements", {}).items():
        for item in disagreements[:limit]:
            rows.append(
                {
                    "comparison": pair_name,
                    "left_best_action": item.get("left_best_action"),
                    "right_best_action": item.get("right_best_action"),
                    "max_abs_q_delta": item.get("max_abs_q_delta"),
                    "state_description": item.get("state_description"),
                }
            )

    return pd.DataFrame(rows)


def write_q_table_html_report(
    input_path: str | Path,
    output_dir: str | Path,
    disagreement_limit: int = 5,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    plots_dir = ensure_output_dir(output_dir / "plots")

    report = load_q_table_comparison(input_path)
    summaries = summaries_to_dataframe(report)
    comparisons = comparisons_to_dataframe(report)
    disagreements = extract_largest_disagreement_rows(report, disagreement_limit)
    plots = create_q_table_plots(summaries, comparisons, plots_dir)

    plot_html = "\n".join(
        f'<div class="plot"><img src="plots/{plot.name}" alt="{plot.stem}"></div>'
        for plot in plots
    )

    body = f"""
<h1>Q-table comparison report</h1>
<p class="note">{REPORT_INTRODUCTION}</p>
<h2>Input</h2>
<ul>
  <li><strong>Comparison file:</strong> <code>{Path(input_path)}</code></li>
</ul>
<h2>Metric glossary</h2>
{q_metric_glossary_html()}
<h2>State encoding glossary</h2>
{state_glossary_html()}
<h2>Q-table summaries</h2>
{dataframe_to_html_table(summaries)}
<h2>Pairwise policy comparisons</h2>
{dataframe_to_html_table(comparisons)}
<h2>Largest Q-value disagreements</h2>
{dataframe_to_html_table(disagreements)}
<h2>Plots</h2>
<div class="grid">
{plot_html}
</div>
"""

    output_path = output_dir / "q_table_report.html"
    write_text(output_path, html_page("Q-table comparison report", body))
    return output_path


def write_q_table_markdown_report(
    input_path: str | Path,
    output_dir: str | Path,
    disagreement_limit: int = 5,
) -> Path:
    output_dir = ensure_output_dir(output_dir)
    plots_dir = ensure_output_dir(output_dir / "plots")

    report = load_q_table_comparison(input_path)
    summaries = summaries_to_dataframe(report)
    comparisons = comparisons_to_dataframe(report)
    disagreements = extract_largest_disagreement_rows(report, disagreement_limit)
    plots = create_q_table_plots(summaries, comparisons, plots_dir)

    plot_markdown = "\n".join(
        f"![{plot.stem}](plots/{plot.name})"
        for plot in plots
    )

    text = f"""# Q-table comparison report

{REPORT_INTRODUCTION}

## Input

- **Comparison file:** `{Path(input_path)}`

## Metric glossary

{q_metric_glossary_markdown()}

## State encoding glossary

{state_glossary_markdown()}

## Q-table summaries

{dataframe_to_markdown_table(summaries)}

## Pairwise policy comparisons

{dataframe_to_markdown_table(comparisons)}

## Largest Q-value disagreements

{dataframe_to_markdown_table(disagreements)}

## Plots

{plot_markdown}
"""

    output_path = output_dir / "q_table_report.md"
    write_text(output_path, text)
    return output_path

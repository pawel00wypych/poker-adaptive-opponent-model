from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics.seed_statistics import (
    SEED_CI_LOWER_COLUMN,
    SEED_CI_MARGIN_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_STANDARD_ERROR_COLUMN,
    add_seed_level_statistical_summary,
)
from src.evaluation.reporting.checkpoint_report import display_agent_name
from src.evaluation.reporting.plot_utils import ensure_output_dir, save_current_figure

MEAN_CI_CHART_FILENAME = "mean_profit_ci_by_opponent.png"
SEED_STABILITY_CHART_FILENAME = "seed_stability_by_opponent.png"

DEFAULT_STD_WARNING_THRESHOLD_BB = 5.0


@dataclass(frozen=True)
class ExperimentChartConfig:
    ci_multiplier: float | None = None
    max_std_across_seeds_bb: float = DEFAULT_STD_WARNING_THRESHOLD_BB
    max_label_length: int = 34


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _latest_checkpoint_rows(summary_table: pd.DataFrame) -> pd.DataFrame:
    if summary_table.empty or "checkpoint_episode" not in summary_table.columns:
        return summary_table.copy()

    latest = summary_table.groupby(["training_run", "opponent_name"])[
        "checkpoint_episode"
    ].transform("max")
    return summary_table[summary_table["checkpoint_episode"] == latest].copy()


def _add_display_labels(df: pd.DataFrame, max_label_length: int) -> pd.DataFrame:
    result = df.copy()
    result["agent_label"] = result["agent_name"].map(display_agent_name)
    result["opponent_label"] = result["opponent_name"].astype(str)
    result["plot_label"] = (
        result["opponent_label"] + "\n" + result["agent_label"]
    )

    if max_label_length > 0:
        result["plot_label"] = result["plot_label"].map(
            lambda value: _truncate_label(str(value), max_label_length)
        )

    return result


def _truncate_label(value: str, max_length: int) -> str:
    lines = []
    for line in value.split("\n"):
        if len(line) <= max_length:
            lines.append(line)
        else:
            lines.append(line[: max_length - 1] + "…")
    return "\n".join(lines)


def _sort_for_plot(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    sort_columns = _existing_columns(
        df,
        [
            "training_run",
            "checkpoint_episode",
            "opponent_name",
            value_column,
            "agent_name",
        ],
    )
    ascending = []
    for column in sort_columns:
        ascending.append(column != value_column)

    return df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def add_cross_seed_confidence_interval(
    summary_table: pd.DataFrame,
    config: ExperimentChartConfig | None = None,
) -> pd.DataFrame:
    """Ensure cross-seed confidence columns are available for charting.

    The default path delegates to the canonical Student-t seed summary. The
    optional multiplier is retained as a backward-compatible chart override.
    """

    config = config or ExperimentChartConfig()
    result = add_seed_level_statistical_summary(summary_table)

    if config.ci_multiplier is not None and not result.empty:
        margin = config.ci_multiplier * result[SEED_STANDARD_ERROR_COLUMN]
        result[SEED_CI_MARGIN_COLUMN] = margin
        result[SEED_CI_LOWER_COLUMN] = result["mean_profit_bb"] - margin
        result[SEED_CI_UPPER_COLUMN] = result["mean_profit_bb"] + margin

    # Preserve the original chart-only column names for callers that already
    # consume this helper directly.
    result["mean_profit_bb_ci_95_lower"] = result[SEED_CI_LOWER_COLUMN]
    result["mean_profit_bb_ci_95_upper"] = result[SEED_CI_UPPER_COLUMN]
    result["mean_profit_bb_ci_95_error"] = result[SEED_CI_MARGIN_COLUMN]

    return result


def plot_mean_profit_confidence_interval(
    summary_table: pd.DataFrame,
    output_path: str | Path,
    config: ExperimentChartConfig | None = None,
) -> Path | None:
    config = config or ExperimentChartConfig()
    data = _latest_checkpoint_rows(summary_table)

    if data.empty or "mean_profit_bb" not in data.columns:
        return None

    data = add_cross_seed_confidence_interval(data, config)
    data = _add_display_labels(data, config.max_label_length)
    data = _sort_for_plot(data, "mean_profit_bb")

    fig_width = max(10.0, min(24.0, 0.65 * len(data)))
    plt.figure(figsize=(fig_width, 6.0))

    x_positions = list(range(len(data)))
    plt.bar(x_positions, data["mean_profit_bb"])
    plt.errorbar(
        x_positions,
        data["mean_profit_bb"],
        yerr=data[SEED_CI_MARGIN_COLUMN],
        fmt="none",
        capsize=4,
        linewidth=1,
    )
    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, data["plot_label"], rotation=45, ha="right")
    plt.ylabel("Mean profit per game [BB]")
    plt.title("Mean profit with 95% Student-t CI across seeds")
    plt.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    save_current_figure(output_path)
    return output_path


def plot_seed_stability(
    summary_table: pd.DataFrame,
    output_path: str | Path,
    config: ExperimentChartConfig | None = None,
) -> Path | None:
    config = config or ExperimentChartConfig()
    data = _latest_checkpoint_rows(summary_table)

    if data.empty or "mean_profit_bb_std_across_seeds" not in data.columns:
        return None

    data = _add_display_labels(data, config.max_label_length)
    data = _sort_for_plot(data, "mean_profit_bb_std_across_seeds")

    fig_width = max(10.0, min(24.0, 0.65 * len(data)))
    plt.figure(figsize=(fig_width, 6.0))

    x_positions = list(range(len(data)))
    plt.bar(x_positions, data["mean_profit_bb_std_across_seeds"])
    plt.axhline(
        config.max_std_across_seeds_bb,
        linestyle="--",
        linewidth=1,
        label=f"Warning threshold ({config.max_std_across_seeds_bb:.1f} BB/game)",
    )
    plt.xticks(x_positions, data["plot_label"], rotation=45, ha="right")
    plt.ylabel("Std across seeds [BB/game]")
    plt.title("Seed stability by opponent and agent")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    save_current_figure(output_path)
    return output_path


def create_experiment_summary_charts(
    summary_table: pd.DataFrame,
    output_dir: str | Path,
    config: ExperimentChartConfig | None = None,
) -> list[Path]:
    config = config or ExperimentChartConfig()
    output_dir = ensure_output_dir(output_dir)

    chart_specs = [
        (
            MEAN_CI_CHART_FILENAME,
            plot_mean_profit_confidence_interval,
        ),
        (
            SEED_STABILITY_CHART_FILENAME,
            plot_seed_stability,
        ),
    ]

    created_paths: list[Path] = []
    for filename, plot_function in chart_specs:
        path = plot_function(summary_table, output_dir / filename, config)
        if path is not None:
            created_paths.append(path)

    return created_paths

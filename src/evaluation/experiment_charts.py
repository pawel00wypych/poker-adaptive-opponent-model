from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.checkpoint_report import display_agent_name
from src.evaluation.plot_utils import ensure_output_dir, save_current_figure

MEAN_CI_CHART_FILENAME = "mean_profit_ci_by_opponent.png"
SEED_STABILITY_CHART_FILENAME = "seed_stability_by_opponent.png"

DEFAULT_CI_MULTIPLIER = 1.96
DEFAULT_STD_WARNING_THRESHOLD_BB = 5.0


@dataclass(frozen=True)
class ExperimentChartConfig:
    ci_multiplier: float = DEFAULT_CI_MULTIPLIER
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
    """Add approximate cross-seed CI columns for mean_profit_bb.

    The checkpoint metrics already contain per-seed confidence intervals across
    games. For summary charts we need a single interval around the aggregated
    mean across training seeds, so this function uses std_across_seeds / sqrt(n).
    """

    config = config or ExperimentChartConfig()
    result = summary_table.copy()

    if result.empty:
        for column in [
            "mean_profit_bb_standard_error_across_seeds",
            "mean_profit_bb_ci_95_lower",
            "mean_profit_bb_ci_95_upper",
            "mean_profit_bb_ci_95_error",
        ]:
            result[column] = pd.Series(dtype="float64")
        return result

    seeds = result.get("seeds", 1).fillna(1).replace(0, 1).astype(float)
    std = result.get("mean_profit_bb_std_across_seeds", 0.0).fillna(0.0)
    mean = result.get("mean_profit_bb", 0.0).fillna(0.0)

    standard_error = std / seeds.pow(0.5)
    error = config.ci_multiplier * standard_error

    result["mean_profit_bb_standard_error_across_seeds"] = standard_error
    result["mean_profit_bb_ci_95_lower"] = mean - error
    result["mean_profit_bb_ci_95_upper"] = mean + error
    result["mean_profit_bb_ci_95_error"] = error

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
        yerr=data["mean_profit_bb_ci_95_error"],
        fmt="none",
        capsize=4,
        linewidth=1,
    )
    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, data["plot_label"], rotation=45, ha="right")
    plt.ylabel("Mean profit per game [BB]")
    plt.title("Mean profit with approximate 95% CI across seeds")
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

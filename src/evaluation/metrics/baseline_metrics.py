from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.metrics.checkpoint_metrics import (
    calculate_grouped_evaluation_metrics,
)

BASELINE_REPLICATE_GROUP_COLUMNS = [
    "evaluation_replicate_id",
    "experiment_name",
    "agent_name",
    "opponent_name",
]
BASELINE_AGENT_SET = frozenset(
    (ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT, RULE_BASED_AGENT)
)

BASELINE_REPLICATE_STANDARD_ERROR_COLUMN = (
    "mean_profit_bb_standard_error_across_evaluation_replicates"
)
BASELINE_REPLICATE_CI_LOWER_COLUMN = (
    "mean_profit_bb_ci_95_lower_across_evaluation_replicates"
)
BASELINE_REPLICATE_CI_UPPER_COLUMN = (
    "mean_profit_bb_ci_95_upper_across_evaluation_replicates"
)
BASELINE_REPLICATE_CI_MARGIN_COLUMN = (
    "mean_profit_bb_ci_95_margin_across_evaluation_replicates"
)
BASELINE_REPLICATE_STD_COLUMN = (
    "mean_profit_bb_std_across_evaluation_replicates"
)
BASELINE_REPLICATE_MIN_COLUMN = (
    "mean_profit_bb_min_across_evaluation_replicates"
)
BASELINE_REPLICATE_MAX_COLUMN = (
    "mean_profit_bb_max_across_evaluation_replicates"
)
BASELINE_REPLICATE_SPREAD_COLUMN = (
    "mean_profit_bb_evaluation_replicate_spread"
)


def _validated_replicate_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = (
        numeric.isna()
        | numeric.lt(0)
        | numeric.mod(1).ne(0)
    )
    if invalid.any():
        invalid_values = values[invalid].astype(str).unique().tolist()
        raise ValueError(
            "evaluation_replicate_id must contain non-negative integers; "
            f"invalid values: {invalid_values}."
        )

    return numeric.astype("int64")


def calculate_baseline_replicate_metrics(
    results_csv_path: str | Path,
) -> pd.DataFrame:
    """Calculate one row per baseline matchup and evaluation replicate."""

    df = pd.read_csv(results_csv_path)
    if "evaluation_replicate_id" not in df.columns:
        raise ValueError(
            "Baseline sanity evaluation requires the "
            "evaluation_replicate_id column. Re-run the baseline-only "
            "evaluation instead of using model_seed as a substitute."
        )

    baseline_rows = df[
        df["evaluation_replicate_id"].notna()
        & df["agent_name"].isin(BASELINE_AGENT_SET)
        & df["opponent_name"].isin(BASELINE_AGENT_SET)
    ].copy()
    if baseline_rows.empty:
        raise ValueError(
            "No baseline rows with evaluation_replicate_id were found."
        )

    baseline_rows["evaluation_replicate_id"] = _validated_replicate_ids(
        baseline_rows["evaluation_replicate_id"]
    )
    populated_model_columns = [
        column
        for column in ("model_seed", "checkpoint_episode")
        if column in baseline_rows.columns and baseline_rows[column].notna().any()
    ]
    if populated_model_columns:
        raise ValueError(
            "Baseline rows must not contain trained-model metadata: "
            f"{populated_model_columns}. Use evaluation_replicate_id only."
        )

    return calculate_grouped_evaluation_metrics(
        baseline_rows,
        BASELINE_REPLICATE_GROUP_COLUMNS,
    )


def aggregate_across_evaluation_replicates(
    replicate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate equally weighted simulation replicates, never model seeds."""

    if replicate_metrics.empty:
        return replicate_metrics.copy()

    aggregated = (
        replicate_metrics.groupby(
            ["experiment_name", "agent_name", "opponent_name"],
            dropna=False,
        )
        .agg(
            evaluation_replicates=("evaluation_replicate_id", "nunique"),
            games=("games", "sum"),
            total_hands=("total_hands", "sum"),
            mean_profit_bb=("mean_profit_bb", "mean"),
            **{
                BASELINE_REPLICATE_STD_COLUMN: ("mean_profit_bb", "std"),
                BASELINE_REPLICATE_MIN_COLUMN: ("mean_profit_bb", "min"),
                BASELINE_REPLICATE_MAX_COLUMN: ("mean_profit_bb", "max"),
            },
            bb_per_100=("bb_per_100", "mean"),
            win_rate=("win_rate", "mean"),
            bust_rate=("bust_rate", "mean"),
            ended_by_bust_rate=("ended_by_bust_rate", "mean"),
            ended_by_round_limit_rate=(
                "ended_by_round_limit_rate",
                "mean",
            ),
            global_classifier_accuracy=(
                "global_classifier_accuracy",
                "mean",
            ),
            global_classifier_coverage=(
                "global_classifier_coverage",
                "mean",
            ),
            mean_policy_switches=("mean_policy_switches", "mean"),
        )
        .reset_index()
    )

    counts = pd.to_numeric(
        aggregated["evaluation_replicates"],
        errors="coerce",
    )
    deviations = pd.to_numeric(
        aggregated[BASELINE_REPLICATE_STD_COLUMN],
        errors="coerce",
    )
    means = pd.to_numeric(aggregated["mean_profit_bb"], errors="coerce")
    valid = counts.ge(2) & deviations.notna() & means.notna()

    standard_errors = pd.Series(
        np.nan,
        index=aggregated.index,
        dtype="float64",
    )
    standard_errors.loc[valid] = (
        deviations.loc[valid] / np.sqrt(counts.loc[valid])
    )
    margins = pd.Series(np.nan, index=aggregated.index, dtype="float64")
    margins.loc[valid] = student_t.ppf(
        0.975,
        counts.loc[valid] - 1,
    ) * standard_errors.loc[valid]

    aggregated[BASELINE_REPLICATE_STANDARD_ERROR_COLUMN] = standard_errors
    aggregated[BASELINE_REPLICATE_CI_MARGIN_COLUMN] = margins
    aggregated[BASELINE_REPLICATE_CI_LOWER_COLUMN] = means - margins
    aggregated[BASELINE_REPLICATE_CI_UPPER_COLUMN] = means + margins
    aggregated[BASELINE_REPLICATE_SPREAD_COLUMN] = (
        aggregated[BASELINE_REPLICATE_MAX_COLUMN]
        - aggregated[BASELINE_REPLICATE_MIN_COLUMN]
    )
    aggregated["mean_hands_played"] = (
        aggregated["total_hands"] / aggregated["games"]
    )

    return aggregated.sort_values(
        ["opponent_name", "agent_name"]
    ).reset_index(drop=True)

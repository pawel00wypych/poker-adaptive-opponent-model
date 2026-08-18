from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SEED_CONFIDENCE_LEVEL = 0.95

SEED_STANDARD_ERROR_COLUMN = (
    "mean_profit_bb_standard_error_across_seeds"
)
SEED_CI_LOWER_COLUMN = "mean_profit_bb_ci_95_lower_across_seeds"
SEED_CI_UPPER_COLUMN = "mean_profit_bb_ci_95_upper_across_seeds"
SEED_CI_MARGIN_COLUMN = "mean_profit_bb_ci_95_margin_across_seeds"
SEED_MIN_COLUMN = "mean_profit_bb_min_across_seeds"
SEED_MAX_COLUMN = "mean_profit_bb_max_across_seeds"
SEED_SPREAD_COLUMN = "mean_profit_bb_seed_spread"

SEED_STATISTIC_COLUMNS = (
    SEED_STANDARD_ERROR_COLUMN,
    SEED_CI_LOWER_COLUMN,
    SEED_CI_UPPER_COLUMN,
    SEED_CI_MARGIN_COLUMN,
    SEED_MIN_COLUMN,
    SEED_MAX_COLUMN,
    SEED_SPREAD_COLUMN,
)


def add_seed_level_statistical_summary(
    aggregated: pd.DataFrame,
) -> pd.DataFrame:
    """Add a 95% Student-t summary across independent training seeds.

    The input contains one aggregated row per agent, opponent, and checkpoint.
    Its mean and standard deviation must be calculated from equally weighted
    per-seed means. A confidence interval is undefined for fewer than two
    seeds, so the corresponding fields remain missing instead of reporting a
    misleading zero-width interval.
    """

    result = aggregated.copy()
    if result.empty:
        for column in SEED_STATISTIC_COLUMNS:
            if column not in result.columns:
                result[column] = pd.Series(dtype="float64")
        return result

    required_columns = {
        "seeds",
        "mean_profit_bb",
        "mean_profit_bb_std_across_seeds",
    }
    missing_columns = sorted(required_columns.difference(result.columns))
    if missing_columns:
        raise ValueError(
            "Cannot calculate seed-level statistics without columns: "
            f"{missing_columns}."
        )

    seed_counts = pd.to_numeric(result["seeds"], errors="coerce")
    means = pd.to_numeric(result["mean_profit_bb"], errors="coerce")
    standard_deviations = pd.to_numeric(
        result["mean_profit_bb_std_across_seeds"],
        errors="coerce",
    )
    valid = (
        seed_counts.ge(2)
        & means.notna()
        & standard_deviations.notna()
    )

    standard_errors = pd.Series(np.nan, index=result.index, dtype="float64")
    standard_errors.loc[valid] = (
        standard_deviations.loc[valid]
        / np.sqrt(seed_counts.loc[valid])
    )

    margins = pd.Series(np.nan, index=result.index, dtype="float64")
    degrees_of_freedom = seed_counts.loc[valid] - 1
    critical_values = student_t.ppf(
        (1.0 + SEED_CONFIDENCE_LEVEL) / 2.0,
        degrees_of_freedom,
    )
    margins.loc[valid] = critical_values * standard_errors.loc[valid]

    result[SEED_STANDARD_ERROR_COLUMN] = standard_errors
    result[SEED_CI_MARGIN_COLUMN] = margins
    result[SEED_CI_LOWER_COLUMN] = means - margins
    result[SEED_CI_UPPER_COLUMN] = means + margins

    if {SEED_MIN_COLUMN, SEED_MAX_COLUMN}.issubset(result.columns):
        result[SEED_SPREAD_COLUMN] = (
            result[SEED_MAX_COLUMN] - result[SEED_MIN_COLUMN]
        )
    else:
        result[SEED_MIN_COLUMN] = np.nan
        result[SEED_MAX_COLUMN] = np.nan
        result[SEED_SPREAD_COLUMN] = np.nan

    return result

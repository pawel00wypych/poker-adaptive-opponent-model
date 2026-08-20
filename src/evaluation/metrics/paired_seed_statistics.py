from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.evaluation.metrics.seed_statistics import SEED_CONFIDENCE_LEVEL


class PairedSeedStatisticsError(ValueError):
    """Raised when seed-level rows cannot form an unambiguous pairing."""


@dataclass(frozen=True)
class PairedSeedStatistics:
    """Statistical summary of ``left - right`` values paired by model seed."""

    left_agent_name: str
    right_agent_name: str
    opponent_name: str
    checkpoint_episode: int | None
    left_seed_count: int
    right_seed_count: int
    common_seeds: tuple[object, ...]
    left_only_seeds: tuple[object, ...]
    right_only_seeds: tuple[object, ...]
    deltas_by_seed: dict[object, float]
    mean_delta: float | None
    standard_deviation: float | None
    standard_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float = SEED_CONFIDENCE_LEVEL

    @property
    def common_seed_count(self) -> int:
        return len(self.common_seeds)

    def to_details(self) -> dict[str, object]:
        return {
            "comparison": (
                f"{self.left_agent_name} - {self.right_agent_name}"
            ),
            "left_agent_name": self.left_agent_name,
            "right_agent_name": self.right_agent_name,
            "opponent_name": self.opponent_name,
            "checkpoint_episode": self.checkpoint_episode,
            "left_seed_count": self.left_seed_count,
            "right_seed_count": self.right_seed_count,
            "common_seed_count": self.common_seed_count,
            "common_seeds": list(self.common_seeds),
            "left_only_seeds": list(self.left_only_seeds),
            "right_only_seeds": list(self.right_only_seeds),
            "deltas_by_seed": {
                str(seed): delta
                for seed, delta in self.deltas_by_seed.items()
            },
            "mean_delta": self.mean_delta,
            "standard_deviation": self.standard_deviation,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


def _as_python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sorted_seed_values(values: set[object]) -> tuple[object, ...]:
    return tuple(
        sorted(
            (_as_python_scalar(value) for value in values),
            key=lambda value: (type(value).__name__, str(value)),
        )
    )


def _agent_seed_values(
    seed_rows: pd.DataFrame,
    *,
    agent_name: str,
    opponent_name: str,
    checkpoint_episode: int | None,
) -> pd.Series:
    matching = seed_rows[
        (seed_rows["agent_name"] == agent_name)
        & (seed_rows["opponent_name"] == opponent_name)
    ]
    if checkpoint_episode is not None:
        matching = matching[
            matching["checkpoint_episode"] == checkpoint_episode
        ]

    duplicated = matching.loc[
        matching["model_seed"].duplicated(keep=False),
        "model_seed",
    ]
    if not duplicated.empty:
        duplicate_seeds = _sorted_seed_values(set(duplicated.tolist()))
        raise PairedSeedStatisticsError(
            "Expected one seed-level row per agent, opponent, and checkpoint; "
            f"found duplicate model_seed values for {agent_name} vs "
            f"{opponent_name}: {list(duplicate_seeds)}."
        )

    values = pd.to_numeric(matching["mean_profit_bb"], errors="coerce")
    if values.isna().any():
        invalid_seeds = _sorted_seed_values(
            set(matching.loc[values.isna(), "model_seed"].tolist())
        )
        raise PairedSeedStatisticsError(
            "mean_profit_bb must be numeric for paired seed statistics; "
            f"invalid model_seed values for {agent_name} vs "
            f"{opponent_name}: {list(invalid_seeds)}."
        )

    return pd.Series(
        values.to_numpy(dtype="float64"),
        index=matching["model_seed"].map(_as_python_scalar),
        dtype="float64",
    )


def calculate_paired_seed_statistics(
    seed_rows: pd.DataFrame,
    *,
    left_agent_name: str,
    right_agent_name: str,
    opponent_name: str,
    checkpoint_episode: int | None = None,
) -> PairedSeedStatistics:
    """Calculate a Student-t CI for per-seed ``left - right`` deltas."""

    required_columns = {
        "agent_name",
        "opponent_name",
        "checkpoint_episode",
        "model_seed",
        "mean_profit_bb",
    }
    missing_columns = sorted(required_columns.difference(seed_rows.columns))
    if missing_columns:
        raise PairedSeedStatisticsError(
            "Cannot calculate paired seed statistics without columns: "
            f"{missing_columns}."
        )

    left_values = _agent_seed_values(
        seed_rows,
        agent_name=left_agent_name,
        opponent_name=opponent_name,
        checkpoint_episode=checkpoint_episode,
    )
    right_values = _agent_seed_values(
        seed_rows,
        agent_name=right_agent_name,
        opponent_name=opponent_name,
        checkpoint_episode=checkpoint_episode,
    )

    left_seeds = set(left_values.index.tolist())
    right_seeds = set(right_values.index.tolist())
    common_seeds = _sorted_seed_values(left_seeds & right_seeds)
    left_only_seeds = _sorted_seed_values(left_seeds - right_seeds)
    right_only_seeds = _sorted_seed_values(right_seeds - left_seeds)
    deltas_by_seed = {
        seed: float(left_values.loc[seed] - right_values.loc[seed])
        for seed in common_seeds
    }

    delta_values = np.asarray(list(deltas_by_seed.values()), dtype="float64")
    mean_delta = (
        float(np.mean(delta_values))
        if delta_values.size
        else None
    )
    standard_deviation: float | None = None
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None

    if delta_values.size >= 2:
        standard_deviation = float(np.std(delta_values, ddof=1))
        standard_error = float(
            standard_deviation / np.sqrt(delta_values.size)
        )
        critical_value = float(
            student_t.ppf(
                (1.0 + SEED_CONFIDENCE_LEVEL) / 2.0,
                delta_values.size - 1,
            )
        )
        margin = critical_value * standard_error
        ci_lower = float(mean_delta - margin)
        ci_upper = float(mean_delta + margin)

    return PairedSeedStatistics(
        left_agent_name=left_agent_name,
        right_agent_name=right_agent_name,
        opponent_name=opponent_name,
        checkpoint_episode=checkpoint_episode,
        left_seed_count=len(left_seeds),
        right_seed_count=len(right_seeds),
        common_seeds=common_seeds,
        left_only_seeds=left_only_seeds,
        right_only_seeds=right_only_seeds,
        deltas_by_seed=deltas_by_seed,
        mean_delta=mean_delta,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

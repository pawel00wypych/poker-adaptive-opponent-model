"""Learning-rate schedule shared by every tabular algorithm.

Monte Carlo used to be the only agent with a configurable learning rate while
the temporal-difference agents ran on a fixed alpha. Comparing them then mixed
an algorithm difference with a schedule difference. Keeping one implementation
means the experiment protocol can fix the same schedule for all four.
"""

import math

from src.training.constants import (
    ALPHA_MODE_CONSTANT,
    ALPHA_MODE_SQRT_VISIT,
    ALPHA_MODE_VISIT_COUNT,
    SUPPORTED_ALPHA_MODES,
)


def validate_alpha_mode(alpha_mode: str) -> str:
    if alpha_mode not in SUPPORTED_ALPHA_MODES:
        raise ValueError(
            f"Unsupported alpha_mode: {alpha_mode}. "
            f"Supported modes: {list(SUPPORTED_ALPHA_MODES)}"
        )

    return alpha_mode


def resolve_learning_rate(
    *,
    alpha: float,
    alpha_mode: str,
    visits: int,
) -> float:
    """Return the step size for one update.

    ``visits`` is the number of times the state-action pair has been updated,
    counted after the current visit, so the first update of a pair uses a step
    size of 1.0 under the visit-count schedule.
    """
    if alpha_mode == ALPHA_MODE_CONSTANT:
        return alpha

    if visits <= 0:
        raise ValueError(
            "visit count must be positive before calculating a "
            "visit-based learning rate"
        )

    if alpha_mode == ALPHA_MODE_VISIT_COUNT:
        return 1.0 / visits

    if alpha_mode == ALPHA_MODE_SQRT_VISIT:
        return 1.0 / math.sqrt(visits)

    raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")

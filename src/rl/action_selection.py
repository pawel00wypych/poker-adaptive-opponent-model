import random
from collections.abc import Sequence

import numpy as np

from src.poker.action_mapper import ActionMapper
from src.rl.types import ActionId, ValidActions


def get_legal_action_ids(
    valid_actions: ValidActions,
) -> list[ActionId]:
    """Return legal fixed RL action IDs for PyPokerEngine actions."""
    return ActionMapper.get_legal_action_ids(valid_actions)


def select_best_legal_action(
    q_values: Sequence[float] | np.ndarray,
    legal_action_ids: Sequence[ActionId],
) -> ActionId:
    """
    Select one legal action with the highest Q-value.

    Ties are intentionally broken randomly to preserve the previous Monte
    Carlo behavior.
    """
    if not legal_action_ids:
        raise ValueError(
            "legal_action_ids must not be empty"
        )

    legal_q_values = {
        action_id: q_values[action_id]
        for action_id in legal_action_ids
    }

    max_q_value = max(
        legal_q_values.values()
    )

    best_actions = [
        action_id
        for action_id, value in legal_q_values.items()
        if value == max_q_value
    ]

    return random.choice(best_actions)


def select_epsilon_greedy_action(
    q_values: Sequence[float] | np.ndarray,
    valid_actions: ValidActions,
    epsilon: float,
    training: bool = True,
) -> ActionId:
    """
    Select a legal action using epsilon-greedy exploration.

    Exploration happens only in training mode. In evaluation mode the best
    legal action is always selected.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError(
            "epsilon must be in range [0, 1]"
        )

    legal_action_ids = get_legal_action_ids(valid_actions)

    if training and random.random() < epsilon:
        return random.choice(legal_action_ids)

    return select_best_legal_action(
        q_values=q_values,
        legal_action_ids=legal_action_ids,
    )

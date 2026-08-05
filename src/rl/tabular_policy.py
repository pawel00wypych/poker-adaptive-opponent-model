from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from src.poker.action_mapper import ActionMapper
from src.rl.types import ActionId, State


class TabularPolicy:
    """
    Stores tabular state-action values and visit counts.

    The class intentionally contains no algorithm-specific update rule.
    Monte Carlo, Q-learning, SARSA, and Double Q-learning can share the
    same Q-table representation while applying different learning rules.
    """

    def __init__(
        self,
        num_actions: int = ActionMapper.NUM_ACTIONS,
    ):
        if num_actions <= 0:
            raise ValueError(
                "num_actions must be greater than zero"
            )

        self.num_actions = num_actions
        self.q_table = defaultdict(
            lambda: np.zeros(
                self.num_actions,
                dtype=float,
            )
        )
        self.visit_counts = defaultdict(
            lambda: [0] * self.num_actions
        )

    def ensure_state_exists(
        self,
        state: State,
    ) -> None:
        _ = self.q_table[state]
        _ = self.visit_counts[state]

    def get_q_values(
        self,
        state: State,
    ) -> np.ndarray:
        self.ensure_state_exists(state)
        return self.q_table[state]

    def get_q_value(
        self,
        state: State,
        action_id: ActionId,
    ) -> float:
        self.ensure_state_exists(state)
        return float(self.q_table[state][action_id])

    def set_q_value(
        self,
        state: State,
        action_id: ActionId,
        value: float,
    ) -> None:
        self.ensure_state_exists(state)
        self.q_table[state][action_id] = float(value)

    def increment_visit_count(
        self,
        state: State,
        action_id: ActionId,
    ) -> int:
        self.ensure_state_exists(state)
        self.visit_counts[state][action_id] += 1
        return self.visit_counts[state][action_id]

    def get_visit_count(
        self,
        state: State,
        action_id: ActionId,
    ) -> int:
        self.ensure_state_exists(state)
        return int(self.visit_counts[state][action_id])

    @staticmethod
    def to_plain_q_table(
        q_table: Mapping,
    ) -> dict[tuple, list[float]]:
        return {
            tuple(state): [
                float(value)
                for value in values
            ]
            for state, values in q_table.items()
        }

    @staticmethod
    def to_plain_visit_counts(
        visit_counts: Mapping,
    ) -> dict[tuple, list[int]]:
        return {
            tuple(state): [
                int(value)
                for value in values
            ]
            for state, values in visit_counts.items()
        }

    def load_plain_q_table(
        self,
        q_table: Mapping,
    ) -> None:
        self.q_table.update(
            {
                tuple(state): np.array(
                    values,
                    dtype=float,
                )
                for state, values in q_table.items()
            }
        )

    def load_plain_visit_counts(
        self,
        visit_counts: Mapping,
    ) -> None:
        self.visit_counts.update(
            {
                tuple(state): [
                    int(value)
                    for value in values
                ]
                for state, values in visit_counts.items()
            }
        )

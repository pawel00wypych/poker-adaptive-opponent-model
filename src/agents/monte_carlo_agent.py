import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.poker.action_mapper import ActionMapper


class MonteCarloAgent:
    """
    Tabular first-visit Monte Carlo control agent.

    The agent stores state-action pairs visited during one poker hand.
    After the hand ends, the terminal hand reward is propagated to every
    first-visited state-action pair.

    Epsilon is controlled externally by the training script. It is not
    automatically decayed after each poker hand.
    """

    SUPPORTED_ALPHA_MODES = {
        "constant",
        "visit_count",
        "sqrt_visit",
    }

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
        alpha_mode: str = "constant",
    ):
        if not 0 < alpha <= 1:
            raise ValueError(
                "alpha must be in range (0, 1]"
            )

        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be in range [0, 1]"
            )

        if not 0 <= epsilon_min <= 1:
            raise ValueError(
                "epsilon_min must be in range [0, 1]"
            )

        if alpha_mode not in self.SUPPORTED_ALPHA_MODES:
            raise ValueError(
                f"Unsupported alpha_mode: {alpha_mode}"
            )

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha_mode = alpha_mode

        self.q_table = defaultdict(
            lambda: np.zeros(
                ActionMapper.NUM_ACTIONS,
                dtype=float,
            )
        )

        self.visit_counts = defaultdict(
            lambda: [0] * ActionMapper.NUM_ACTIONS
        )

        self.episode: list[tuple[tuple, int]] = []
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False
        self.episode.clear()

    def set_epsilon(
        self,
        epsilon: float,
    ) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be in range [0, 1]"
            )

        self.epsilon = max(
            self.epsilon_min,
            epsilon,
        )

    def act(
        self,
        state: tuple,
        valid_actions: list[dict],
    ) -> int:
        self._ensure_state_exists(state)

        legal_action_ids = (
            ActionMapper.get_legal_action_ids(
                valid_actions
            )
        )

        if (
            self.training
            and random.random() < self.epsilon
        ):
            return random.choice(
                legal_action_ids
            )

        q_values = self.q_table[state]

        legal_q_values = {
            action_id: q_values[action_id]
            for action_id in legal_action_ids
        }

        max_q_value = max(
            legal_q_values.values()
        )

        best_actions = [
            action_id
            for action_id, value
            in legal_q_values.items()
            if value == max_q_value
        ]

        return random.choice(
            best_actions
        )

    def remember(
        self,
        state: tuple,
        action_id: int,
    ) -> None:
        if self.training:
            self.episode.append(
                (state, action_id)
            )

    def learn_from_episode(
        self,
        reward: float,
    ) -> None:
        """
        Update all first-visited state-action pairs from one poker hand.

        Epsilon is intentionally not changed here because this method is
        called once per poker hand, not once per training game.
        """
        if not self.training:
            self.episode.clear()
            return

        visited: set[tuple[tuple, int]] = set()

        for state, action_id in self.episode:
            state_action = (
                state,
                action_id,
            )

            if state_action in visited:
                continue

            visited.add(state_action)

            self._ensure_state_exists(
                state
            )

            self.visit_counts[state][action_id] += 1

            learning_rate = self._learning_rate(
                state,
                action_id,
            )

            old_value = (
                self.q_table[state][action_id]
            )

            self.q_table[state][action_id] = (
                old_value
                + learning_rate
                * (reward - old_value)
            )

        self.episode.clear()

    def _ensure_state_exists(
        self,
        state: tuple,
    ) -> None:
        _ = self.q_table[state]
        _ = self.visit_counts[state]

    def _learning_rate(
        self,
        state: tuple,
        action_id: int,
    ) -> float:
        if self.alpha_mode == "constant":
            return self.alpha

        visits = self.visit_counts[state][action_id]

        if visits <= 0:
            raise ValueError(
                "visit count must be positive before "
                "calculating visit-count alpha"
            )

        if self.alpha_mode == "visit_count":
            return 1.0 / visits

        if self.alpha_mode == "sqrt_visit":
            return 1.0 / math.sqrt(visits)

        raise ValueError(
            f"Unsupported alpha_mode: {self.alpha_mode}"
        )

    @staticmethod
    def _to_plain_q_table(
        q_table: dict,
    ) -> dict[tuple, list[float]]:
        return {
            tuple(state): [
                float(value)
                for value in values
            ]
            for state, values in q_table.items()
        }

    @staticmethod
    def _to_plain_visit_counts(
        visit_counts: dict,
    ) -> dict[tuple, list[int]]:
        return {
            tuple(state): [
                int(value)
                for value in values
            ]
            for state, values in visit_counts.items()
        }

    def save(
        self,
        path: str,
        metadata: dict | None = None,
    ) -> None:
        model_path = Path(path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "algorithm": (
                "first_visit_monte_carlo_control"
            ),
            "q_table": self._to_plain_q_table(
                self.q_table
            ),
            "visit_counts": self._to_plain_visit_counts(
                self.visit_counts
            ),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "epsilon_min": self.epsilon_min,
            "alpha_mode": self.alpha_mode,
            "metadata": metadata or {},
        }

        with model_path.open(
            "wb"
        ) as file:
            pickle.dump(
                payload,
                file,
            )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "MonteCarloAgent":
        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                "Monte Carlo model does not exist: "
                f"{model_path}"
            )

        with model_path.open(
            "rb"
        ) as file:
            payload = pickle.load(
                file
            )

        if not isinstance(payload, dict):
            raise TypeError(
                "Unsupported Monte Carlo model payload: "
                f"{type(payload)}"
            )

        agent = cls(
            alpha=payload.get("alpha", 0.1),
            epsilon=payload.get("epsilon", 0.0),
            epsilon_min=payload.get(
                "epsilon_min",
                0.05,
            ),
            alpha_mode=payload.get(
                "alpha_mode",
                "constant",
            ),
        )

        q_table = payload.get(
            "q_table",
            {},
        )

        agent.q_table.update(
            {
                tuple(state): np.array(
                    values,
                    dtype=float,
                )
                for state, values in q_table.items()
            }
        )

        visit_counts = payload.get(
            "visit_counts",
            {},
        )

        agent.visit_counts.update(
            {
                tuple(state): [
                    int(value)
                    for value in values
                ]
                for state, values in visit_counts.items()
            }
        )

        agent.eval()

        return agent

    @staticmethod
    def load_metadata(
        path: str,
    ) -> dict:
        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                "Monte Carlo model does not exist: "
                f"{model_path}"
            )

        with model_path.open(
            "rb"
        ) as file:
            payload = pickle.load(
                file
            )

        if not isinstance(payload, dict):
            return {}

        return payload.get(
            "metadata",
            {},
        )

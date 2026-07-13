import pickle
import random
from pathlib import Path

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

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
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

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        self.q_table: dict[
            tuple,
            np.ndarray,
        ] = {}

        self.episode: list[
            tuple[tuple, int]
        ] = []

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

        visited: set[
            tuple[tuple, int]
        ] = set()

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

            old_value = (
                self.q_table[state][action_id]
            )

            self.q_table[state][action_id] = (
                old_value
                + self.alpha
                * (reward - old_value)
            )

        self.episode.clear()

    def _ensure_state_exists(
        self,
        state: tuple,
    ) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(
                ActionMapper.NUM_ACTIONS,
                dtype=float,
            )

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
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "epsilon_min": self.epsilon_min,
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

        agent = cls(
            alpha=payload["alpha"],
            epsilon=payload["epsilon"],
            epsilon_min=(
                payload["epsilon_min"]
            ),
        )

        agent.q_table = payload["q_table"]

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

        return payload.get(
            "metadata",
            {},
        )
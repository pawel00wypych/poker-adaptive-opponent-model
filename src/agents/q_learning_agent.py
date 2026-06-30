import pickle
import random
from pathlib import Path

import numpy as np

from src.poker.action_mapper import ActionMapper


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[tuple, np.ndarray] = {}
        self.episode: list[tuple[tuple, int]] = []

        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def act(self, state: tuple, valid_actions: list[dict]) -> int:
        self._ensure_state_exists(state)

        legal_action_ids = ActionMapper.get_legal_action_ids(valid_actions)

        if self.training and random.random() < self.epsilon:
            return random.choice(legal_action_ids)

        q_values = self.q_table[state]

        legal_q_values = {
            action_id: q_values[action_id]
            for action_id in legal_action_ids
        }

        return max(legal_q_values, key=legal_q_values.get)

    def remember(self, state: tuple, action_id: int) -> None:
        if self.training:
            self.episode.append((state, action_id))

    def learn_from_episode(self, reward: float) -> None:
        if not self.training:
            self.episode = []
            return

        for state, action_id in self.episode:
            self._ensure_state_exists(state)

            old_value = self.q_table[state][action_id]
            new_value = old_value + self.alpha * (reward - old_value)

            self.q_table[state][action_id] = new_value

        self.episode = []
        self._decay_epsilon()

    def _ensure_state_exists(self, state: tuple) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(ActionMapper.NUM_ACTIONS)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

        with open(path, "wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as file:
            payload = pickle.load(file)

        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )

        agent.q_table = payload["q_table"]

        return agent
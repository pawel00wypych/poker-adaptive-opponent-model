import math

from src.rl.action_selection import select_epsilon_greedy_action
from src.rl.constants import (
    ALGORITHM_KEY,
    ALGORITHM_MONTE_CARLO,
    METADATA_KEY,
    Q_TABLE_KEY,
    VISIT_COUNTS_KEY,
)
from src.rl.model_io import (
    load_model_metadata,
    load_model_payload,
    save_model_payload,
)
from src.rl.tabular_policy import TabularPolicy
from src.training.constants import (
    ALPHA_MODE_CONSTANT,
    ALPHA_MODE_SQRT_VISIT,
    ALPHA_MODE_VISIT_COUNT,
    SUPPORTED_ALPHA_MODES,
)


class MonteCarloAgent:
    """
    Tabular first-visit Monte Carlo control agent.

    The agent stores state-action pairs visited during one poker hand.
    After the hand ends, the terminal hand reward is propagated to every
    first-visited state-action pair.

    Epsilon is controlled externally by the training script. It is not
    automatically decayed after each poker hand.
    """

    SUPPORTED_ALPHA_MODES = set(SUPPORTED_ALPHA_MODES)
    ALGORITHM_ID = ALGORITHM_MONTE_CARLO

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
        alpha_mode: str = ALPHA_MODE_CONSTANT,
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

        self.policy = TabularPolicy()
        self.q_table = self.policy.q_table
        self.visit_counts = self.policy.visit_counts

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
        q_values = self.policy.get_q_values(state)

        return select_epsilon_greedy_action(
            q_values=q_values,
            valid_actions=valid_actions,
            epsilon=self.epsilon,
            training=self.training,
        )

    def remember(
        self,
        state: tuple,
        action_id: int,
        valid_actions: list[dict] | None = None,
    ) -> None:
        # valid_actions is accepted for API compatibility with TD agents.
        # Monte Carlo does not need next-state legal actions for updates.
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

            self.policy.increment_visit_count(
                state,
                action_id,
            )

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
        self.policy.ensure_state_exists(state)

    def _learning_rate(
        self,
        state: tuple,
        action_id: int,
    ) -> float:
        if self.alpha_mode == ALPHA_MODE_CONSTANT:
            return self.alpha

        visits = self.visit_counts[state][action_id]

        if visits <= 0:
            raise ValueError(
                "visit count must be positive before "
                "calculating visit-count alpha"
            )

        if self.alpha_mode == ALPHA_MODE_VISIT_COUNT:
            return 1.0 / visits

        if self.alpha_mode == ALPHA_MODE_SQRT_VISIT:
            return 1.0 / math.sqrt(visits)

        raise ValueError(
            f"Unsupported alpha_mode: {self.alpha_mode}"
        )

    def save(
        self,
        path: str,
        metadata: dict | None = None,
    ) -> None:
        payload = {
            ALGORITHM_KEY: self.ALGORITHM_ID,
            Q_TABLE_KEY: TabularPolicy.to_plain_q_table(
                self.q_table
            ),
            VISIT_COUNTS_KEY: TabularPolicy.to_plain_visit_counts(
                self.visit_counts
            ),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "epsilon_min": self.epsilon_min,
            "alpha_mode": self.alpha_mode,
            METADATA_KEY: metadata or {},
        }

        save_model_payload(
            path=path,
            payload=payload,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "MonteCarloAgent":
        payload = load_model_payload(
            path=path,
            model_name="Monte Carlo",
            expected_algorithm=cls.ALGORITHM_ID,
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
                ALPHA_MODE_CONSTANT,
            ),
        )

        q_table = payload.get(
            Q_TABLE_KEY,
            {},
        )
        agent.policy.load_plain_q_table(q_table)

        visit_counts = payload.get(
            VISIT_COUNTS_KEY,
            {},
        )
        agent.policy.load_plain_visit_counts(visit_counts)

        agent.eval()

        return agent

    @staticmethod
    def load_metadata(
        path: str,
    ) -> dict:
        return load_model_metadata(
            path=path,
            model_name="Monte Carlo",
        )

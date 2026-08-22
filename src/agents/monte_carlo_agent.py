
from src.rl.action_selection import select_epsilon_greedy_action
from src.rl.constants import (
    ALGORITHM_KEY,
    ALGORITHM_MONTE_CARLO,
    METADATA_KEY,
    Q_TABLE_KEY,
    VISIT_COUNTS_KEY,
)
from src.rl.decision_diagnostics import DecisionDiagnostics
from src.rl.learning_rate import resolve_learning_rate, validate_alpha_mode
from src.rl.model_io import (
    load_model_metadata,
    load_model_payload,
    save_model_payload,
)
from src.rl.tabular_policy import TabularPolicy
from src.training.constants import (
    ALPHA_MODE_CONSTANT,
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
        gamma: float = 1.0,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
        alpha_mode: str = ALPHA_MODE_CONSTANT,
    ):
        if not 0 < alpha <= 1:
            raise ValueError(
                "alpha must be in range (0, 1]"
            )

        if not 0 <= gamma <= 1:
            raise ValueError(
                "gamma must be in range [0, 1]"
            )

        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be in range [0, 1]"
            )

        if not 0 <= epsilon_min <= 1:
            raise ValueError(
                "epsilon_min must be in range [0, 1]"
            )

        validate_alpha_mode(alpha_mode)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha_mode = alpha_mode

        self.policy = TabularPolicy()
        self.q_table = self.policy.q_table
        self.visit_counts = self.policy.visit_counts

        self.episode: list[tuple[tuple, int]] = []
        self.training = True
        self.diagnostics = DecisionDiagnostics()

    def reset_decision_diagnostics(self) -> None:
        self.diagnostics.reset()

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
        visit_counts = self.policy.peek_visit_counts(state)
        q_values = self.policy.get_q_values(state)

        action_id = select_epsilon_greedy_action(
            q_values=q_values,
            valid_actions=valid_actions,
            epsilon=self.epsilon,
            training=self.training,
        )

        self.diagnostics.record(
            visit_counts=visit_counts,
            action_id=action_id,
        )

        return action_id

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
        """Update all first-visited state-action pairs from one poker hand.

        The environment pays out only at the end of a hand, so the return of a
        visited pair is the terminal reward discounted by how many decisions
        still followed it. With the default gamma of 1.0 every first visit
        receives the full terminal reward.

        Epsilon is intentionally not changed here because this method is
        called once per poker hand, not once per training game.
        """
        if not self.training:
            self.episode.clear()
            return

        visited: set[tuple[tuple, int]] = set()
        final_index = len(self.episode) - 1

        for index, (state, action_id) in enumerate(self.episode):
            state_action = (
                state,
                action_id,
            )

            if state_action in visited:
                continue

            visited.add(state_action)

            visits = self.policy.increment_visit_count(
                state,
                action_id,
            )

            learning_rate = resolve_learning_rate(
                alpha=self.alpha,
                alpha_mode=self.alpha_mode,
                visits=visits,
            )

            discounted_return = (
                self.gamma ** (final_index - index)
            ) * reward

            old_value = (
                self.q_table[state][action_id]
            )

            self.q_table[state][action_id] = (
                old_value
                + learning_rate
                * (discounted_return - old_value)
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
        return resolve_learning_rate(
            alpha=self.alpha,
            alpha_mode=self.alpha_mode,
            visits=self.visit_counts[state][action_id],
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
            "gamma": self.gamma,
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
            gamma=payload.get("gamma", 1.0),
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

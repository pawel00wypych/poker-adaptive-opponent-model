from src.rl.action_selection import (
    get_legal_action_ids,
    select_epsilon_greedy_action,
)
from src.rl.constants import (
    ALGORITHM_KEY,
    ALGORITHM_SARSA,
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
from src.rl.types import ActionId, State, ValidActions
from src.training.constants import ALPHA_MODE_CONSTANT


class SarsaAgent:
    """
    Tabular SARSA agent.

    SARSA is an on-policy TD control method. The agent keeps the same player
    interface as MonteCarloAgent and QLearningAgent: it stores the hand
    trajectory and applies TD backups when the terminal hand reward is known.

    Intermediate transitions receive reward 0.0. The final transition receives
    the terminal reward in big blinds. Non-terminal SARSA targets use the
    actual next action from the stored trajectory instead of max_a Q(s', a).
    """

    ALGORITHM_ID = ALGORITHM_SARSA

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
        self.alpha_mode = alpha_mode
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        self.policy = TabularPolicy()
        self.q_table = self.policy.q_table
        self.visit_counts = self.policy.visit_counts

        self.episode: list[tuple[State, ActionId, tuple[ActionId, ...]]] = []
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
        state: State,
        valid_actions: ValidActions,
    ) -> ActionId:
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
        state: State,
        action_id: ActionId,
        valid_actions: ValidActions | None = None,
    ) -> None:
        if not self.training:
            return

        legal_action_ids = (
            tuple(get_legal_action_ids(valid_actions))
            if valid_actions is not None
            else tuple(range(self.policy.num_actions))
        )

        self.episode.append(
            (
                state,
                action_id,
                legal_action_ids,
            )
        )

    def learn_from_transition(
        self,
        *,
        state: State,
        action_id: ActionId,
        reward: float,
        next_state: State | None = None,
        next_action_id: ActionId | None = None,
        done: bool = False,
    ) -> None:
        if not self.training:
            return

        visits = self.policy.increment_visit_count(
            state,
            action_id,
        )

        old_value = self.policy.get_q_value(
            state,
            action_id,
        )

        target = self._target(
            reward=reward,
            next_state=next_state,
            next_action_id=next_action_id,
            done=done,
        )

        learning_rate = resolve_learning_rate(
            alpha=self.alpha,
            alpha_mode=self.alpha_mode,
            visits=visits,
        )

        updated_value = old_value + learning_rate * (
            target - old_value
        )

        self.policy.set_q_value(
            state,
            action_id,
            updated_value,
        )

    def learn_from_episode(
        self,
        reward: float,
    ) -> None:
        """
        Apply SARSA backups over the remembered hand trajectory.

        Since the environment reward is only available after the hand ends,
        non-terminal transitions receive 0.0 reward and the final transition
        receives the terminal hand reward.
        """
        if not self.training:
            self.episode.clear()
            return

        # Replay backwards so the terminal reward reaches every state
        # visited in the hand. Iterating forwards moved it a single
        # step per hand, which starves long trajectories.
        for index in reversed(range(len(self.episode))):
            state, action_id, _legal_action_ids = self.episode[index]
            is_terminal_transition = index == len(self.episode) - 1

            next_state = None
            next_action_id = None

            if not is_terminal_transition:
                (
                    next_state,
                    next_action_id,
                    _next_legal_action_ids,
                ) = self.episode[index + 1]

            transition_reward = (
                reward
                if is_terminal_transition
                else 0.0
            )

            self.learn_from_transition(
                state=state,
                action_id=action_id,
                reward=transition_reward,
                next_state=next_state,
                next_action_id=next_action_id,
                done=is_terminal_transition,
            )

        self.episode.clear()

    def _target(
        self,
        *,
        reward: float,
        next_state: State | None,
        next_action_id: ActionId | None,
        done: bool,
    ) -> float:
        if done or next_state is None or next_action_id is None:
            return float(reward)

        next_value = self.policy.get_q_value(
            next_state,
            next_action_id,
        )

        return float(
            reward
            + self.gamma * next_value
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
    ) -> "SarsaAgent":
        payload = load_model_payload(
            path=path,
            model_name="SARSA",
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
            model_name="SARSA",
        )

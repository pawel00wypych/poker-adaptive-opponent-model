import random

import numpy as np

from src.rl.action_selection import (
    get_legal_action_ids,
    select_best_legal_action,
    select_epsilon_greedy_action,
)
from src.rl.constants import (
    ALGORITHM_DOUBLE_Q_LEARNING,
    ALGORITHM_KEY,
    METADATA_KEY,
    Q1_TABLE_KEY,
    Q1_VISIT_COUNTS_KEY,
    Q2_TABLE_KEY,
    Q2_VISIT_COUNTS_KEY,
    Q_TABLE_KEY,
    VISIT_COUNTS_KEY,
)
from src.rl.model_io import (
    load_model_metadata,
    load_model_payload,
    save_model_payload,
)
from src.rl.tabular_policy import TabularPolicy
from src.rl.types import ActionId, State, ValidActions

UPDATE_Q1 = "q1"
UPDATE_Q2 = "q2"
DOUBLE_Q_LEARNING_ALGORITHM_KEY = ALGORITHM_DOUBLE_Q_LEARNING


class DoubleQLearningAgent:
    """
    Tabular Double Q-learning agent.

    The agent keeps two independent Q-tables. On every update it randomly
    selects one table to update. The updated table selects the greedy next
    action, while the other table evaluates that action. This reduces the
    overestimation bias of standard Q-learning while preserving the same player
    interface used by MonteCarloAgent, QLearningAgent, and SarsaAgent.

    Because the poker environment exposes the final reward at the end of a
    hand, the remembered hand trajectory is replayed with reward 0.0 for
    non-terminal transitions and the terminal reward in big blinds for the last
    transition.
    """

    ALGORITHM_ID = ALGORITHM_DOUBLE_Q_LEARNING

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.5,
        epsilon_min: float = 0.05,
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

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        self.q1_policy = TabularPolicy()
        self.q2_policy = TabularPolicy()

        # Compatibility view used by existing players/trainers/tests. It stores
        # the average of Q1 and Q2, while action selection uses Q1 + Q2.
        self.policy = TabularPolicy()
        self.q_table = self.policy.q_table
        self.visit_counts = self.policy.visit_counts

        self.q1_table = self.q1_policy.q_table
        self.q2_table = self.q2_policy.q_table
        self.q1_visit_counts = self.q1_policy.visit_counts
        self.q2_visit_counts = self.q2_policy.visit_counts

        self.episode: list[tuple[State, ActionId, tuple[ActionId, ...]]] = []
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
        state: State,
        valid_actions: ValidActions,
    ) -> ActionId:
        q_values = self.get_combined_q_values(state)
        self._refresh_combined_state(state)

        return select_epsilon_greedy_action(
            q_values=q_values,
            valid_actions=valid_actions,
            epsilon=self.epsilon,
            training=self.training,
        )

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

    def get_combined_q_values(
        self,
        state: State,
    ) -> np.ndarray:
        self.q1_policy.ensure_state_exists(state)
        self.q2_policy.ensure_state_exists(state)
        return self.q1_table[state] + self.q2_table[state]

    def learn_from_transition(
        self,
        *,
        state: State,
        action_id: ActionId,
        reward: float,
        next_state: State | None = None,
        next_legal_action_ids: tuple[ActionId, ...] | None = None,
        done: bool = False,
        update_table: str | None = None,
    ) -> str | None:
        if not self.training:
            return None

        selected_update_table = update_table or self._select_update_table()

        if selected_update_table == UPDATE_Q1:
            self._update_selected_table(
                update_policy=self.q1_policy,
                evaluation_policy=self.q2_policy,
                state=state,
                action_id=action_id,
                reward=reward,
                next_state=next_state,
                next_legal_action_ids=next_legal_action_ids,
                done=done,
            )
        elif selected_update_table == UPDATE_Q2:
            self._update_selected_table(
                update_policy=self.q2_policy,
                evaluation_policy=self.q1_policy,
                state=state,
                action_id=action_id,
                reward=reward,
                next_state=next_state,
                next_legal_action_ids=next_legal_action_ids,
                done=done,
            )
        else:
            raise ValueError(
                "update_table must be one of: q1, q2"
            )

        self._refresh_combined_state(state)
        if next_state is not None:
            self._refresh_combined_state(next_state)

        return selected_update_table

    def learn_from_episode(
        self,
        reward: float,
    ) -> None:
        if not self.training:
            self.episode.clear()
            return

        for index, (
            state,
            action_id,
            _legal_action_ids,
        ) in enumerate(self.episode):
            is_terminal_transition = index == len(self.episode) - 1

            next_state = None
            next_legal_action_ids = None

            if not is_terminal_transition:
                (
                    next_state,
                    _next_action_id,
                    next_legal_action_ids,
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
                next_legal_action_ids=next_legal_action_ids,
                done=is_terminal_transition,
            )

        self.episode.clear()

    def _select_update_table(self) -> str:
        return UPDATE_Q1 if random.random() < 0.5 else UPDATE_Q2

    def _update_selected_table(
        self,
        *,
        update_policy: TabularPolicy,
        evaluation_policy: TabularPolicy,
        state: State,
        action_id: ActionId,
        reward: float,
        next_state: State | None,
        next_legal_action_ids: tuple[ActionId, ...] | None,
        done: bool,
    ) -> None:
        update_policy.increment_visit_count(
            state,
            action_id,
        )

        old_value = update_policy.get_q_value(
            state,
            action_id,
        )

        target = self._target(
            reward=reward,
            next_state=next_state,
            next_legal_action_ids=next_legal_action_ids,
            done=done,
            update_policy=update_policy,
            evaluation_policy=evaluation_policy,
        )

        updated_value = old_value + self.alpha * (
            target - old_value
        )

        update_policy.set_q_value(
            state,
            action_id,
            updated_value,
        )

    def _target(
        self,
        *,
        reward: float,
        next_state: State | None,
        next_legal_action_ids: tuple[ActionId, ...] | None,
        done: bool,
        update_policy: TabularPolicy,
        evaluation_policy: TabularPolicy,
    ) -> float:
        if done or next_state is None:
            return float(reward)

        update_q_values = update_policy.get_q_values(next_state)

        legal_action_ids = (
            next_legal_action_ids
            if next_legal_action_ids
            else tuple(range(update_policy.num_actions))
        )

        best_next_action = select_best_legal_action(
            q_values=update_q_values,
            legal_action_ids=legal_action_ids,
        )

        evaluated_next_value = evaluation_policy.get_q_value(
            next_state,
            best_next_action,
        )

        return float(
            reward
            + self.gamma * evaluated_next_value
        )

    def _refresh_combined_state(
        self,
        state: State,
    ) -> None:
        self.q1_policy.ensure_state_exists(state)
        self.q2_policy.ensure_state_exists(state)
        self.policy.ensure_state_exists(state)

        self.q_table[state] = (
            self.q1_table[state]
            + self.q2_table[state]
        ) / 2.0

        self.visit_counts[state] = [
            int(q1_count) + int(q2_count)
            for q1_count, q2_count in zip(
                self.q1_visit_counts[state],
                self.q2_visit_counts[state],
            )
        ]

    def _refresh_all_combined_states(self) -> None:
        states = set(self.q1_table.keys()) | set(self.q2_table.keys())
        for state in states:
            self._refresh_combined_state(state)

    def save(
        self,
        path: str,
        metadata: dict | None = None,
    ) -> None:
        self._refresh_all_combined_states()

        payload = {
            ALGORITHM_KEY: self.ALGORITHM_ID,
            Q1_TABLE_KEY: TabularPolicy.to_plain_q_table(
                self.q1_table
            ),
            Q2_TABLE_KEY: TabularPolicy.to_plain_q_table(
                self.q2_table
            ),
            Q1_VISIT_COUNTS_KEY: TabularPolicy.to_plain_visit_counts(
                self.q1_visit_counts
            ),
            Q2_VISIT_COUNTS_KEY: TabularPolicy.to_plain_visit_counts(
                self.q2_visit_counts
            ),
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
    ) -> "DoubleQLearningAgent":
        payload = load_model_payload(
            path=path,
            model_name="Double Q-learning",
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
        )

        agent.q1_policy.load_plain_q_table(
            payload.get(Q1_TABLE_KEY, {})
        )
        agent.q2_policy.load_plain_q_table(
            payload.get(Q2_TABLE_KEY, {})
        )
        agent.q1_policy.load_plain_visit_counts(
            payload.get(Q1_VISIT_COUNTS_KEY, {})
        )
        agent.q2_policy.load_plain_visit_counts(
            payload.get(Q2_VISIT_COUNTS_KEY, {})
        )
        agent._refresh_all_combined_states()
        agent.eval()

        return agent

    @staticmethod
    def load_metadata(
        path: str,
    ) -> dict:
        return load_model_metadata(
            path=path,
            model_name="Double Q-learning",
        )

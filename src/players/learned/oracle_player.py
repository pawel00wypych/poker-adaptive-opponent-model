from collections import Counter

from src.features.state_encoder import StateEncoder
from src.players.base.player_template import PlayerTemplate
from src.poker.action_mapper import ActionMapper
from src.poker.betting import to_decision_actions
from src.poker.constants import (
    POLICY_TYPES,
    TRAINING_OPPONENT_TYPES,
)
from src.poker.round_state_utils import (
    get_player_stack,
    is_small_blind,
    get_round_count,
)
from src.players.constants import PLAYER_NAME_ORACLE_MC


class OraclePlayer(PlayerTemplate):
    """
    Adaptive upper-bound baseline.

    The player knows the opponent type from the first decision and always
    uses the corresponding specialist policy.

    It does not use the classifier.
    """

    REQUIRED_AGENTS = set(POLICY_TYPES)

    SUPPORTED_ORACLE_TYPES = set(TRAINING_OPPONENT_TYPES)

    def __init__(
        self,
        agents: dict,
        oracle_opponent_type: str,
        player_name: str = PLAYER_NAME_ORACLE_MC,
        verbose: bool = False,
        log_interval: int = 1,
    ):
        super().__init__(player_name=player_name)

        missing_agents = self.REQUIRED_AGENTS - set(agents.keys())

        if missing_agents:
            raise ValueError(
                "Missing oracle agents: "
                f"{sorted(missing_agents)}"
            )

        if oracle_opponent_type not in self.SUPPORTED_ORACLE_TYPES:
            raise ValueError(
                "Unsupported oracle opponent type: "
                f"{oracle_opponent_type}"
            )

        if log_interval <= 0:
            raise ValueError("log_interval must be greater than zero")

        self.agents = agents
        self.oracle_opponent_type = oracle_opponent_type
        self.active_policy_type = oracle_opponent_type

        self.verbose = verbose
        self.log_interval = log_interval

        self.policy_usage_counts = Counter()

    @property
    def final_predicted_type(self) -> str:
        return self.oracle_opponent_type

    def declare_action(
        self,
        valid_actions,
        hole_card,
        round_state,
    ):
        my_stack = get_player_stack(
            round_state,
            self.uuid,
        )

        if self.initial_stack is None:
            self.initial_stack = my_stack

        self.policy_usage_counts[
            self.active_policy_type
        ] += 1

        decision_actions = to_decision_actions(
            valid_actions,
            round_state,
            self.uuid,
        )

        active_agent = self.agents[
            self.active_policy_type
        ]

        state = StateEncoder.encode(
            player_stack=my_stack,
            valid_actions=decision_actions,
            round_state=round_state,
            hole_cards=hole_card,
            is_small_blind=is_small_blind(round_state, self.uuid),
        )

        action_id = active_agent.act(
            state,
            decision_actions,
        )

        action, amount = ActionMapper.to_engine_action(
            action_id,
            valid_actions,
        )

        if active_agent.training:
            active_agent.remember(
                state,
                action_id,
                valid_actions=decision_actions,
            )

        if self.verbose:
            print(
                "[OracleDecision] "
                f"round={get_round_count(round_state)}, "
                f"oracle_type={self.oracle_opponent_type}, "
                f"active_policy={self.active_policy_type}, "
                f"state={state}, "
                f"action={action}, "
                f"amount={amount}"
            )

        return action, amount

    def receive_game_start_message(
        self,
        game_info,
    ):
        super().receive_game_start_message(game_info)

        for agent in self.agents.values():
            agent.reset_decision_diagnostics()

        self.policy_usage_counts.clear()
        self.active_policy_type = self.oracle_opponent_type

    def receive_game_update_message(
        self,
        action,
        round_state,
    ):
        pass

    def receive_round_result_message(
        self,
        winners,
        hand_info,
        round_state,
    ):
        final_stack = self.get_my_stack_from_round_state(round_state)
        reward_bb = self.calculate_reward_bb(final_stack)

        active_agent = self.agents[
            self.active_policy_type
        ]

        if active_agent.training:
            active_agent.learn_from_episode(reward_bb)

        self.total_reward_bb += reward_bb
        self.update_round_tracking_after_result(final_stack)

        if (
            self.verbose
            and self.hands_played % self.log_interval == 0
        ):
            print(
                "[OraclePlayer] "
                f"round={get_round_count(round_state)}, "
                f"stack={final_stack}, "
                f"reward_bb={reward_bb:.2f}, "
                f"total_reward_bb={self.total_reward_bb:.2f}, "
                f"oracle_type={self.oracle_opponent_type}, "
                f"active_policy={self.active_policy_type}"
            )

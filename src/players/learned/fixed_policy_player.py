from src.features.state_encoder import StateEncoder
from src.players.base.player_template import PlayerTemplate
from src.poker.action_mapper import ActionMapper
from src.poker.betting import to_decision_actions
from src.poker.constants import POLICY_TYPES
from src.poker.round_state_utils import (
    get_player_stack,
    get_round_count,
)
from src.players.constants import PLAYER_NAME_FIXED_POLICY


class FixedPolicyPlayer(PlayerTemplate):
    """
    Evaluation-only player that always uses one selected policy.

    It is used for cross-policy evaluation, for example:
        unknown policy vs calling opponent
        tight policy vs calling opponent
        aggressive policy vs calling opponent
        calling policy vs calling opponent
    """

    SUPPORTED_POLICY_TYPES = set(POLICY_TYPES)

    def __init__(
        self,
        agent,
        policy_type: str,
        player_name: str = PLAYER_NAME_FIXED_POLICY,
        verbose: bool = False,
        log_interval: int = 1,
    ):
        super().__init__(player_name=player_name)

        if policy_type not in self.SUPPORTED_POLICY_TYPES:
            raise ValueError(f"Unsupported policy type: {policy_type}")

        if log_interval <= 0:
            raise ValueError("log_interval must be greater than zero")

        self.agent = agent
        self.policy_type = policy_type
        self.verbose = verbose
        self.log_interval = log_interval

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

        decision_actions = to_decision_actions(
            valid_actions,
            round_state,
            self.uuid,
        )

        state = StateEncoder.encode(
            player_stack=my_stack,
            valid_actions=decision_actions,
            round_state=round_state,
            hole_cards=hole_card,
        )

        action_id = self.agent.act(
            state,
            decision_actions,
        )

        action, amount = ActionMapper.to_engine_action(
            action_id,
            valid_actions,
        )

        if self.agent.training:
            self.agent.remember(
                state,
                action_id,
                valid_actions=decision_actions,
            )

        return action, amount

    def receive_game_start_message(
        self,
        game_info,
    ):
        super().receive_game_start_message(game_info)
        self.agent.reset_decision_diagnostics()

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

        if self.agent.training:
            self.agent.learn_from_episode(reward_bb)

        self.total_reward_bb += reward_bb
        self.update_round_tracking_after_result(final_stack)

        if (
            self.verbose
            and self.hands_played % self.log_interval == 0
        ):
            print(
                "[FixedPolicyPlayer] "
                f"policy={self.policy_type}, "
                f"round={get_round_count(round_state)}, "
                f"stack={final_stack}, "
                f"reward_bb={reward_bb:.2f}, "
                f"total_reward_bb={self.total_reward_bb:.2f}"
            )

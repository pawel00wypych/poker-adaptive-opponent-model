from src.features.state_encoder import StateEncoder
from src.players.player_template import PlayerTemplate
from src.poker.action_mapper import ActionMapper
from src.poker.constants import POLICY_TYPES
from src.poker.round_state_utils import (
    get_player_stack,
    get_round_count,
)
from src.players.constants import PLAYER_NAME_FIXED_POLICY
from src.config import GameConfig


class FixedPolicyPlayer(PlayerTemplate):
    """
    Evaluation-only player that always uses one selected policy.

    It is used for cross-policy evaluation, for example:
        unknown policy vs calling opponent
        fish policy vs calling opponent
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
        super().__init__(
            player_name=player_name
        )

        if policy_type not in self.SUPPORTED_POLICY_TYPES:
            raise ValueError(
                f"Unsupported policy type: {policy_type}"
            )

        if log_interval <= 0:
            raise ValueError(
                "log_interval must be greater than zero"
            )

        self.agent = agent
        self.policy_type = policy_type
        self.verbose = verbose
        self.log_interval = log_interval

        self.initial_stack: int | None = None
        self.previous_stack: int | None = None
        self.hands_played = 0
        self.total_reward_bb = 0.0

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

        if self.previous_stack is None:
            self.previous_stack = my_stack

        state = StateEncoder.encode(
            player_stack=my_stack,
            valid_actions=valid_actions,
            round_state=round_state,
            hole_cards=hole_card,
            opponent_type=self.policy_type,
        )

        action_id = self.agent.act(
            state,
            valid_actions,
        )

        action, amount = ActionMapper.to_engine_action(
            action_id,
            valid_actions,
        )

        if self.agent.training:
            self.agent.remember(
                state,
                action_id,
            )

        return action, amount

    def receive_game_start_message(
        self,
        game_info,
    ):
        self.initial_stack = None
        self.previous_stack = None
        self.hands_played = 0
        self.total_reward_bb = 0.0

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
        my_stack = get_player_stack(
            round_state,
            self.uuid,
        )

        if self.initial_stack is None:
            self.initial_stack = my_stack

        if self.previous_stack is None:
            self.previous_stack = my_stack

        reward = my_stack - self.previous_stack
        reward_bb = reward / (GameConfig.small_blind_amount * 2)

        self.agent.learn_from_episode(
            reward_bb
        )

        self.total_reward_bb += reward_bb
        self.previous_stack = my_stack
        self.hands_played += 1

        if (
            self.verbose
            and self.hands_played % self.log_interval == 0
        ):
            print(
                "[FixedPolicyPlayer] "
                f"policy={self.policy_type}, "
                f"round={get_round_count(round_state)}, "
                f"stack={my_stack}, "
                f"reward_bb={reward_bb:.2f}, "
                f"total_reward_bb={self.total_reward_bb:.2f}"
            )

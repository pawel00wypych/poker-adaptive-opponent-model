from collections import Counter

from src.config import GameConfig
from src.features.state_encoder import StateEncoder
from src.players.player_template import PlayerTemplate
from src.poker.action_mapper import ActionMapper
from src.poker.constants import (
    POLICY_TYPES,
    TRAINING_OPPONENT_TYPES,
)
from src.poker.round_state_utils import (
    get_player_stack,
    get_round_count,
)
from src.players.constants import PLAYER_NAME_ORACLE_ADAPTIVE


class OracleAdaptivePlayer(PlayerTemplate):
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
        player_name: str = PLAYER_NAME_ORACLE_ADAPTIVE,
        verbose: bool = False,
        log_interval: int = 1,
    ):
        super().__init__(
            player_name=player_name
        )

        missing_agents = (
            self.REQUIRED_AGENTS
            - set(agents.keys())
        )

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
            raise ValueError(
                "log_interval must be greater than zero"
            )

        self.agents = agents
        self.oracle_opponent_type = oracle_opponent_type
        self.active_policy_type = oracle_opponent_type

        self.verbose = verbose
        self.log_interval = log_interval

        self.initial_stack: int | None = None
        self.previous_stack: int | None = None
        self.hands_played = 0
        self.total_reward_bb = 0.0

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

        if self.previous_stack is None:
            self.previous_stack = my_stack

        self.policy_usage_counts[
            self.active_policy_type
        ] += 1

        active_agent = self.agents[
            self.active_policy_type
        ]

        state = StateEncoder.encode(
            player_stack=my_stack,
            valid_actions=valid_actions,
            round_state=round_state,
            hole_cards=hole_card,
            opponent_type=self.active_policy_type,
        )

        action_id = active_agent.act(
            state,
            valid_actions,
        )

        action, amount = ActionMapper.to_engine_action(
            action_id,
            valid_actions,
        )

        if active_agent.training:
            active_agent.remember(
                state,
                action_id,
            )

        if self.verbose:
            print(
                "[OracleAdaptiveDecision] "
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
        self.initial_stack = None
        self.previous_stack = None
        self.hands_played = 0
        self.total_reward_bb = 0.0
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

        active_agent = self.agents[
            self.active_policy_type
        ]

        active_agent.learn_from_episode(
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
                "[OracleAdaptivePlayer] "
                f"round={get_round_count(round_state)}, "
                f"stack={my_stack}, "
                f"reward_bb={reward_bb:.2f}, "
                f"total_reward_bb={self.total_reward_bb:.2f}, "
                f"oracle_type={self.oracle_opponent_type}, "
                f"active_policy={self.active_policy_type}"
            )

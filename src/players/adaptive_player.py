from collections import Counter

from src.config import GameConfig
from src.players.player_template import PlayerTemplate
from src.features.opponent_stats import OpponentStats
from src.features.state_encoder import StateEncoder
from src.opponent_model.rule_based_classifier import (
    RuleBasedOpponentClassifier,
)
from src.poker.action_mapper import ActionMapper
from src.poker.constants import (
    OPPONENT_TYPE_UNKNOWN,
    POLICY_TYPES,
)
from src.poker.round_state_utils import (
    get_player_stack,
    get_round_count,
)
from src.players.constants import PLAYER_NAME_ADAPTIVE_MC


class AdaptivePlayer(PlayerTemplate):
    """
    Adaptive player selecting a specialist policy based on the
    classified opponent type.

    Before classification, the general policy identified as unknown
    is used.
    """

    REQUIRED_AGENTS = set(POLICY_TYPES)

    def __init__(
        self,
        agents: dict,
        player_name: str = PLAYER_NAME_ADAPTIVE_MC,
        expected_opponent_type: str | None = None,
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
                "Missing adaptive agents: "
                f"{sorted(missing_agents)}"
            )

        if log_interval <= 0:
            raise ValueError(
                "log_interval must be greater than zero"
            )

        self.agents = agents
        self.expected_opponent_type = (
            expected_opponent_type
        )

        self.verbose = verbose
        self.log_interval = log_interval

        self.classifier = (
            RuleBasedOpponentClassifier(
                min_actions=5,
            )
        )

        self.opponent_stats = OpponentStats()

        self.initial_stack: int | None = None
        self.hand_start_stack: int | None = None

        self.hands_played = 0
        self.total_reward_bb = 0.0

        self.current_opponent_type = OPPONENT_TYPE_UNKNOWN
        self.active_policy_type = OPPONENT_TYPE_UNKNOWN

        self.classification_counts = Counter()
        self.policy_usage_counts = Counter()

        self.classified_decisions = 0
        self.correct_classifications = 0
        self.incorrect_classifications = 0
        self.unknown_classifications = 0

        self.policy_switches = 0

        self.first_classification_hand: int | None = None
        self.first_correct_classification_hand: int | None = None
        self.first_classification_action_count: int | None = None
        self.first_correct_classification_action_count: int | None = None

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

        if self.hand_start_stack is None:
            self.hand_start_stack = my_stack

        predicted_type = self.classifier.classify(
            self.opponent_stats
        )

        self.current_opponent_type = predicted_type

        self._record_classification(
            predicted_type
        )

        new_policy_type = self._select_policy_type(
            predicted_type
        )

        if new_policy_type != self.active_policy_type:
            self.policy_switches += 1
            self.active_policy_type = new_policy_type

            if self.verbose:
                print(
                    "[AdaptivePolicySwitch] "
                    f"round="
                    f"{get_round_count(round_state)}, "
                    f"predicted_type="
                    f"{predicted_type}, "
                    f"active_policy="
                    f"{self.active_policy_type}, "
                    f"expected_type="
                    f"{self.expected_opponent_type}"
                )

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
                "[AdaptiveDecision] "
                f"round={get_round_count(round_state)}, "
                f"predicted_type={predicted_type}, "
                f"active_policy="
                f"{self.active_policy_type}, "
                f"expected_type="
                f"{self.expected_opponent_type}, "
                f"state={state}, "
                f"action={action}, "
                f"amount={amount}"
            )

        return action, amount

    def _select_policy_type(
        self,
        predicted_type: str,
    ) -> str:
        if predicted_type in self.agents:
            return predicted_type

        return OPPONENT_TYPE_UNKNOWN

    def _record_classification(
        self,
        predicted_type: str,
    ) -> None:
        self.classification_counts[
            predicted_type
        ] += 1

        if predicted_type == OPPONENT_TYPE_UNKNOWN:
            self.unknown_classifications += 1
            return

        self.classified_decisions += 1

        if self.first_classification_hand is None:
            self.first_classification_hand = (
                self.hands_played + 1
            )

        if self.first_classification_action_count is None:
            self.first_classification_action_count = (
                self.opponent_stats.total_actions
            )

        if self.expected_opponent_type is None:
            return

        if predicted_type == self.expected_opponent_type:
            self.correct_classifications += 1

            if (
                self.first_correct_classification_hand
                is None
            ):
                self.first_correct_classification_hand = (
                    self.hands_played + 1
                )

            if (
                self.first_correct_classification_action_count
                is None
            ):
                self.first_correct_classification_action_count = (
                    self.opponent_stats.total_actions
                )
        else:
            self.incorrect_classifications += 1

    @property
    def classifier_accuracy(self) -> float:
        evaluated = (
            self.correct_classifications
            + self.incorrect_classifications
        )

        if evaluated == 0:
            return 0.0

        return (
            self.correct_classifications
            / evaluated
        )

    @property
    def classifier_coverage(self) -> float:
        total = (
            self.classified_decisions
            + self.unknown_classifications
        )

        if total == 0:
            return 0.0

        return (
            self.classified_decisions
            / total
        )

    @property
    def final_predicted_type(self) -> str:
        return self.current_opponent_type

    def receive_game_start_message(
        self,
        game_info,
    ):
        self.initial_stack = None
        self.hand_start_stack = None
        self.hands_played = 0
        self.total_reward_bb = 0.0

        self.current_opponent_type = OPPONENT_TYPE_UNKNOWN
        self.active_policy_type = OPPONENT_TYPE_UNKNOWN

        self.opponent_stats = OpponentStats()

        self.classification_counts.clear()
        self.policy_usage_counts.clear()

        self.classified_decisions = 0
        self.correct_classifications = 0
        self.incorrect_classifications = 0
        self.unknown_classifications = 0

        self.policy_switches = 0

        self.first_classification_hand = None
        self.first_correct_classification_hand = None

    def receive_game_update_message(
        self,
        action,
        round_state,
    ):
        action_player_uuid = action.get(
            "player_uuid"
        )

        if action_player_uuid != self.uuid:
            self.opponent_stats.update_action(
                action.get("action")
            )

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

        if self.hand_start_stack is None:
            self.hand_start_stack = my_stack

        reward = my_stack - self.hand_start_stack
        reward_bb = reward / (GameConfig.small_blind_amount * 2)

        for agent in self.agents.values():
            if agent.training:
                agent.learn_from_episode(
                    reward_bb
                )

        self.total_reward_bb += reward_bb
        self.hand_start_stack = my_stack
        self.hands_played += 1

        self.opponent_stats.finish_hand()

        if (
            self.verbose
            and self.hands_played % self.log_interval == 0
        ):
            print(
                "[AdaptivePlayer] "
                f"round={get_round_count(round_state)}, "
                f"stack={my_stack}, "
                f"reward_bb={reward_bb:.2f}, "
                f"total_reward_bb="
                f"{self.total_reward_bb:.2f}, "
                f"predicted_type="
                f"{self.current_opponent_type}, "
                f"active_policy="
                f"{self.active_policy_type}, "
                f"expected_type="
                f"{self.expected_opponent_type}, "
                f"accuracy="
                f"{self.classifier_accuracy:.3f}, "
                f"coverage="
                f"{self.classifier_coverage:.3f}, "
                f"switches={self.policy_switches}, "
                f"classifications="
                f"{dict(self.classification_counts)}, "
                f"policy_usage="
                f"{dict(self.policy_usage_counts)}, "
                f"stats={self.opponent_stats.as_dict()}"
            )

from src.agents.player_template import PlayerTemplate
from src.features.opponent_stats import OpponentStats
from src.features.state_encoder import StateEncoder
from src.opponent_model.rule_based_classifier import RuleBasedOpponentClassifier
from src.poker.action_mapper import ActionMapper
from src.poker.round_state_utils import get_player_stack, get_round_count


class AdaptivePlayer(PlayerTemplate):
    def __init__(self, agent, player_name: str = "adaptive_player"):
        super().__init__(player_name=player_name)

        self.agent = agent

        self.opponent_stats = OpponentStats()
        self.classifier = RuleBasedOpponentClassifier(min_actions=5)

        self.initial_stack: int | None = None
        self.previous_stack: int | None = None

        self.hands_played: int = 0
        self.total_reward_bb: float = 0.0
        self.current_opponent_type: str = "unknown"

        self.last_state: tuple | None = None
        self.last_action_id: int | None = None

    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack = get_player_stack(round_state, self.uuid)

        if self.initial_stack is None:
            self.initial_stack = my_stack

        if self.previous_stack is None:
            self.previous_stack = my_stack

        self.current_opponent_type = self.classifier.classify(self.opponent_stats)

        state = StateEncoder.encode(
            player_stack=my_stack,
            valid_actions=valid_actions,
            round_state=round_state,
            hole_cards=hole_card,
            opponent_type=self.current_opponent_type,
        )

        action_id = self.agent.act(state, valid_actions)
        action, amount = ActionMapper.to_engine_action(action_id, valid_actions)

        self.agent.remember(state, action_id)

        self.last_state = state
        self.last_action_id = action_id

        return action, amount

    def receive_game_start_message(self, game_info):
        self.initial_stack = None
        self.previous_stack = None
        self.hands_played = 0
        self.total_reward_bb = 0.0
        self.current_opponent_type = "unknown"
        self.opponent_stats = OpponentStats()

    def receive_game_update_message(self, action, round_state):
        action_player_uuid = action.get("player_uuid")

        if action_player_uuid != self.uuid:
            self.opponent_stats.update_action(action.get("action"))

    def receive_round_result_message(self, winners, hand_info, round_state):
        my_stack = get_player_stack(round_state, self.uuid)

        if self.initial_stack is None:
            self.initial_stack = my_stack

        if self.previous_stack is None:
            self.previous_stack = my_stack

        reward = my_stack - self.previous_stack
        reward_bb = reward / 10

        self.agent.learn_from_episode(reward_bb)

        self.total_reward_bb += reward_bb
        self.previous_stack = my_stack
        self.hands_played += 1

        self.opponent_stats.finish_hand()

        print(
            "[AdaptivePlayer] "
            f"round={get_round_count(round_state)}, "
            f"stack={my_stack}, "
            f"reward_bb={reward_bb:.2f}, "
            f"total_reward_bb={self.total_reward_bb:.2f}, "
            f"opponent_type={self.current_opponent_type}, "
            f"stats={self.opponent_stats.as_dict()}"
        )
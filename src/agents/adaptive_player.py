from src.agents.player_template import PlayerTemplate
from src.features.opponent_stats import OpponentStats
from src.features.state_encoder import StateEncoder
from src.opponent_model.rule_based_classifier import RuleBasedOpponentClassifier
from src.poker.action_mapper import ActionMapper


class AdaptivePlayer(PlayerTemplate):
    def __init__(self, agent, player_name: str = "adaptive_player"):
        super().__init__(player_name=player_name)
        self.agent = agent

        self.opponent_stats = OpponentStats()
        self.classifier = RuleBasedOpponentClassifier(min_actions=20)

        self.last_stack = None
        self.current_opponent_type = "unknown"

    def declare_action(self, valid_actions, hole_card, round_state):
        self.current_opponent_type = self.classifier.classify(self.opponent_stats)

        state = StateEncoder.encode(
            player_stack=self.stack,
            valid_actions=valid_actions,
            round_state=round_state,
            opponent_type=self.current_opponent_type,
        )

        action_id = self.agent.act(state, valid_actions)
        action, amount = ActionMapper.to_engine_action(action_id, valid_actions)

        self.agent.remember(state, action_id)

        return action, amount

    def receive_game_start_message(self, game_info):
        self.initial_stack = self.stack
        self.last_stack = self.stack

    def receive_game_update_message(self, action, round_state):
        """
        action example usually contains:
        {
            'player_uuid': ...,
            'action': 'call',
            'amount': ...
        }

        We only update opponent statistics when the action was made by another player.
        """
        action_player_uuid = action.get("player_uuid")

        if action_player_uuid != self.uuid:
            self.opponent_stats.update_action(action.get("action"))

    def receive_round_result_message(self, winners, hand_info, round_state):
        final_stack = self.stack

        if self.last_stack is None:
            self.last_stack = self.initial_stack

        reward = final_stack - self.last_stack

        # Normalize roughly to small blind units.
        # If small blind = 5, then 2 * small blind = big blind = 10.
        reward_bb = reward / 10

        self.agent.learn_from_episode(reward_bb)

        self.opponent_stats.finish_hand()
        self.last_stack = final_stack

        print(
            f"[AdaptivePlayer] reward_bb={reward_bb:.2f}, "
            f"opponent_type={self.current_opponent_type}, "
            f"stats={self.opponent_stats.as_dict()}"
        )

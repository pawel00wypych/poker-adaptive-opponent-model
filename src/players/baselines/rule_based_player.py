from src.players.base.player_template import PlayerTemplate
from src.poker.round_state_utils import get_player_stack


class RuleBasedPlayer(PlayerTemplate):
    """
    Simple rule-based baseline.

    Strategy:
    - check/call if free,
    - fold expensive calls,
    - sometimes min-raise,
    - otherwise call cheap actions.

    This player does not use cards, opponent modelling or learning.
    """

    def __init__(self, player_name: str = "rule_based"):
        super().__init__(player_name=player_name)
        self.action_counter = 0
        self.reset_tracking()

    def declare_action(self, valid_actions, hole_card, round_state):
        self.action_counter += 1

        fold_action = self._find_action(valid_actions, "fold")
        call_action = self._find_action(valid_actions, "call")
        raise_action = self._find_action(valid_actions, "raise")

        call_amount = call_action["amount"] if call_action else 0

        if call_action and call_amount == 0:
            return "call", 0

        if call_amount >= 30 and fold_action:
            return "fold", fold_action["amount"]

        if self.action_counter % 10 == 0 and raise_action is not None:
            amount = raise_action["amount"]

            if isinstance(amount, dict):
                min_raise = amount.get("min")
                max_raise = amount.get("max")

                if min_raise is not None and max_raise is not None and min_raise != -1 and max_raise != -1:
                    return "raise", min_raise

        if call_action:
            return "call", call_action["amount"]

        if fold_action:
            return "fold", fold_action["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]

    def receive_game_start_message(self, game_info):
        self.reset_tracking()

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        current_stack = get_player_stack(round_state, self.uuid)
        self.update_tracking_after_round(current_stack=current_stack, big_blind=10)

    @staticmethod
    def _find_action(valid_actions, action_name):
        return next((item for item in valid_actions if item["action"] == action_name), None)
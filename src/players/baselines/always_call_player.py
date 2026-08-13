from src.players.base.player_template import PlayerTemplate
from src.poker.round_state_utils import get_player_stack


class AlwaysCallPlayer(PlayerTemplate):
    """
    Deterministic sanity baseline that calls or checks whenever possible.

    This player is intentionally simple. It is useful as an evaluation baseline
    for detecting whether an opponent can be exploited by pure passive calling,
    similarly to how AlwaysRaisePlayer detects vulnerability to trivial
    aggression.
    """

    def __init__(self, player_name: str = "always_call"):
        super().__init__(player_name=player_name)
        self.reset_tracking()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        action_by_name = {
            action["action"]: action
            for action in valid_actions
        }

        call_action = action_by_name.get("call")
        if call_action is not None:
            return call_action["action"], call_action["amount"]

        fold_action = action_by_name.get("fold")
        if fold_action is not None:
            return fold_action["action"], fold_action["amount"]

        first_action = valid_actions[0]
        return first_action["action"], first_action["amount"]

    def receive_game_start_message(self, game_info):
        self.reset_tracking()

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        current_stack = get_player_stack(round_state, self.uuid)
        self.update_tracking_after_round(current_stack=current_stack, big_blind=10)

from src.players.base.player_template import PlayerTemplate
from src.poker.round_state_utils import get_player_stack

class CallingPlayer(PlayerTemplate):
    """
    Simple passive baseline that calls or checks whenever possible.

    This player intentionally models a clear call-heavy behavioural profile,
    not a strong poker heuristic. More selective passive play is represented by
    CallingExtremePlayer and rule-based logic is kept in RuleBasedPlayer.
    """

    def __init__(self, player_name: str = "calling_player"):
        super().__init__(player_name=player_name)
        self.reset_tracking()

    def declare_action(self, valid_actions, hole_card, round_state):
        if not valid_actions:
            return "fold", 0

        call_action = self._find_action(valid_actions, "call")
        if call_action is not None:
            return call_action["action"], call_action["amount"]

        fold_action = self._find_action(valid_actions, "fold")
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
        self.update_tracking_after_round(current_stack=current_stack)

    @staticmethod
    def _find_action(valid_actions, action_name: str):
        return next(
            (
                action
                for action in valid_actions
                if action["action"] == action_name
            ),
            None,
        )

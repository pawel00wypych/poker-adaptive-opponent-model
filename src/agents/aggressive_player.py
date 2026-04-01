import random

from src.agents.player_template import PlayerTemplate


class AggressivePlayer(PlayerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        chance_to_raise = random.random() * 100 + 1

        # helper to get the first dict matching action type
        def get_action_info(action_name):
            return next(a for a in valid_actions if a["action"] == action_name)

        if chance_to_raise > 20:
            raise_action_info = get_action_info("raise")
            action, amount = (
                raise_action_info["action"],
                raise_action_info["amount"]["min"],
            )
        elif 5 < chance_to_raise <= 20:
            call_action_info = get_action_info("call")
            action, amount = call_action_info["action"], call_action_info["amount"]
        else:
            fold_action_info = get_action_info("fold")
            action, amount = fold_action_info["action"], fold_action_info["amount"]

        return action, amount

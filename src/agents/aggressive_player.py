from src.agents.player_template import PlayerTemplate
from pypokerengine.engine.poker_constants import PokerConstants as Const
import random

class AggressivePlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        chance_to_raise = random.random() * 100 + 1
        if chance_to_raise > 20:
            raise_action_info = valid_actions[Const.Action.RAISE]
            action, amount = raise_action_info["action"], raise_action_info[
                "amount"]["min"]
        elif 5 < chance_to_raise <= 20:
            call_action_info = valid_actions[Const.Action.CALL]
            action, amount = call_action_info["action"], call_action_info[
                "amount"]
        else:
            fold_action_info = valid_actions[Const.Action.FOLD]
            action, amount = fold_action_info["action"], fold_action_info[
                "amount"]

        return action, amount
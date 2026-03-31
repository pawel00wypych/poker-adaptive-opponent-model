from src.agents.player_template import PlayerTemplate
from pypokerengine.engine.poker_constants import PokerConstants as Const
from src.cards.hand_estimator import HandStrength
import random

class FishPlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        if round_state["street"] == Const.Street.PREFLOP:
            hole_card_strength = self.hand_estimator.check_preflop_hole_card_strength(hole_card)
            print(f"{self.player_name} -> {hole_card} = {hole_card_strength}")
            if hole_card_strength.value > HandStrength.MEDIUM_CARDS:
                call_action_info = valid_actions[Const.Action.CALL]
                action, amount = call_action_info["action"], call_action_info[
                    "amount"]
            else:
                fold_action_info = valid_actions[Const.Action.FOLD]
                action, amount = fold_action_info["action"], fold_action_info[
                    "amount"]
            return action, amount
        elif round_state["street"] == Const.Street.FLOP:
            call_action_info = valid_actions[Const.Action.CALL]
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

            return action, amount
        elif round_state["street"] == Const.Street.TURN:
            call_action_info = valid_actions[Const.Action.CALL]
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

            return action, amount
        elif round_state["street"] == Const.Street.RIVER:
            call_action_info = valid_actions[Const.Action.CALL]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                    "amount"]

            return action, amount
        return None



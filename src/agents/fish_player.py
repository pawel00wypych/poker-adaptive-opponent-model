from src.agents.player_template import PlayerTemplate
from PyPokerEngine.pypokerengine.engine.poker_constants import PokerConstants as Const
from src.cards.hand_estimator import HandStrength
import random

class FishPlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        if round_state["street"] == "preflop":
            hole_card_strength = self.hand_estimator.check_preflop_hole_card_strength(hole_card)
            print(f"{self.player_name} -> {hole_card} = {hole_card_strength}")
            if hole_card_strength > HandStrength.MEDIUM_CARDS:

                call_action_info = PlayerTemplate.get_action(valid_actions,"call")
                action, amount = call_action_info["action"], call_action_info[
                    "amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions,
                                                             "fold")
                action, amount = fold_action_info["action"], fold_action_info[
                    "amount"]
        elif round_state["street"] == "flop":
            call_action_info = PlayerTemplate.get_action(valid_actions,"call")
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

        elif round_state["street"] == "turn":
            call_action_info = PlayerTemplate.get_action(valid_actions,"call")
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

        elif round_state["street"] == "river":
            call_action_info = PlayerTemplate.get_action(valid_actions,"call")
            action, amount = call_action_info["action"], call_action_info[
                    "amount"]

        else:
            call_action_info = PlayerTemplate.get_action(valid_actions, "call")
            action, amount = call_action_info["action"], call_action_info["amount"]

        return action, amount




from src.agents.player_template import PlayerTemplate
import random

class FishPlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        if round_state["street"] == 'preflop':
            hole_card_strength = self.hand_estimator.check_preflop_hole_card_strength(hole_card)
            print(f"{self.player_name} -> {hole_card} = {hole_card_strength}")
            if hole_card_strength.value > 2:
                call_action_info = valid_actions[
                    1]  # 0 - fold 1 - call 2 - raise
                action, amount = call_action_info["action"], call_action_info[
                    "amount"]
                return action, amount
            else:
                fold_action_info = valid_actions[
                    0]  # 0 - fold 1 - call 2 - raise
                action, amount = fold_action_info["action"], fold_action_info[
                    "amount"]
                return action, amount
        elif round_state["street"] == 'flop':
            call_action_info = valid_actions[1]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

            return action, amount
        elif round_state["street"] == 'turn':
            call_action_info = valid_actions[1]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                "amount"]

            return action, amount
        elif round_state["street"] == 'river':
            call_action_info = valid_actions[1]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                    "amount"]

            return action, amount



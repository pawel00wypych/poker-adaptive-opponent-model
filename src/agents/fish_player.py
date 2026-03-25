from src.agents.player_template import PlayerTemplate
import random

class FishPlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        chance_to_call = random.random() * 100 + 1
        if chance_to_call > 20:
            call_action_info = valid_actions[1]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                "amount"]
        elif 5 < chance_to_call <= 20:
            fold_action_info = valid_actions[0]  # 0 - fold 1 - call 2 - raise
            action, amount = fold_action_info["action"], fold_action_info[
                "amount"]
        else:
            raise_action_info = valid_actions[2]  # 0 - fold 1 - call 2 - raise
            action, amount = raise_action_info["action"], raise_action_info[
                "amount"]["min"]
        return action, amount   # action returned here is sent to the poker engine
from src.agents.player_template import PlayerTemplate
import random

class AggressivePlayer(PlayerTemplate):

    def __init__(self):
        super().__init__()

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        chance_to_raise = random.random() * 100 + 1
        if chance_to_raise > 20:
            raise_action_info = valid_actions[2] # 0 - fold 1 - call 2 - raise
            action, amount = raise_action_info["action"], raise_action_info[
                "amount"]["min"]
        elif 5 < chance_to_raise <= 20:
            call_action_info = valid_actions[1]  # 0 - fold 1 - call 2 - raise
            action, amount = call_action_info["action"], call_action_info[
                "amount"]
        else:
            fold_action_info = valid_actions[0]  # 0 - fold 1 - call 2 - raise
            action, amount = fold_action_info["action"], fold_action_info[
                "amount"]

        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        print(f"hole cards of player {self.name} in round {round_count} = "
              f"{hole_card}")

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


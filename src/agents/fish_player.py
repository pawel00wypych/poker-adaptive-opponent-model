from src.agents.player_template import PlayerTemplate
from src.cards.evaluator_Interface import EvaluatorInterface

class FishPlayer(PlayerTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):

        player_hand_info = EvaluatorInterface.evaluate(hole_card, [])
        print(f"{self.player_name} -> {hole_card}  score = {player_hand_info['score']}")

        if round_state["street"] == "preflop":
            if player_hand_info["score"] > 43433:
                # 43433 is arbitrary, it is equivalent to hole = ["C9", "DT"]
                call_action_info = PlayerTemplate.get_action(valid_actions,"call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions,"fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "flop":
            if player_hand_info["hand"]["strength"] != 'HIGHCARD':
                call_action_info = PlayerTemplate.get_action(valid_actions,"call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "turn":

            if player_hand_info["score"] >= 143922:
                #143922 is = TWOPAIR: 22 and 33
                call_action_info = PlayerTemplate.get_action(valid_actions, "call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "river":
            if player_hand_info["score"] >= 164482:
                # 164482 is = TWOPAIR: 22 and 88
                call_action_info = PlayerTemplate.get_action(valid_actions, "call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        else:
            call_action_info = PlayerTemplate.get_action(valid_actions, "call")
            action, amount = call_action_info["action"], call_action_info["amount"]

        return action, amount
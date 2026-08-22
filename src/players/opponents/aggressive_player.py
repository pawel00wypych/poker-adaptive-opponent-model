import random

from src.cards.evaluator_interface import EvaluatorInterface
from src.players.base.player_template import PlayerTemplate
from src.poker.betting import avoid_free_fold


class AggressivePlayer(PlayerTemplate):
    """Raise-heavy scripted training opponent.

    Hand-strength thresholds are deliberately crude; the point is a clear
    raise-first behavioural profile rather than sound poker. The only rule
    enforced on top is that it never gives up a hand for free, because folding
    a free check is strictly dominated and would hand the opponent equity for
    nothing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def declare_action(self, valid_actions, hole_card, round_state):
        action, amount = self._choose_action(
            valid_actions,
            hole_card,
            round_state,
        )

        return avoid_free_fold(
            action,
            amount,
            valid_actions,
            round_state,
            self.player_uuid,
        )

    def _choose_action(self, valid_actions, hole_card, round_state):

        player_hand_info = EvaluatorInterface.evaluate(hole_card,
                                                       round_state['community_card'])
        rand_num = random.random() * 100 + 1  # [1, 101)

        if round_state["street"] == "preflop":
            if player_hand_info["score"] > 23433 and rand_num >= 2:
                # 23433 is arbitrary
                raise_action_info = PlayerTemplate.get_action(valid_actions,
                                                             "raise")
                action, amount = (raise_action_info["action"],
                                  raise_action_info["amount"]["min"])
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "flop":
            if rand_num >= 10:
                raise_action_info = PlayerTemplate.get_action(valid_actions,
                                                             "raise")
                action, amount = (raise_action_info["action"],
                                  raise_action_info["amount"]["min"])
            elif rand_num >= 2:
                call_action_info = PlayerTemplate.get_action(valid_actions,
                                                             "call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "turn":
            if rand_num >= 10:
                raise_action_info = PlayerTemplate.get_action(valid_actions, "raise")
                action, amount = (
                    raise_action_info["action"],
                    raise_action_info["amount"]["min"],
                )
            elif rand_num >= 2:
                call_action_info = PlayerTemplate.get_action(valid_actions, "call")
                action, amount = call_action_info["action"], call_action_info["amount"]
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        elif round_state["street"] == "river":
            if player_hand_info["score"] >= 164482 and rand_num >= 2:
                # 164482 is = TWOPAIR: 22 and 88
                raise_action_info = PlayerTemplate.get_action(valid_actions,
                                                             "raise")
                action, amount = (raise_action_info["action"],
                                  raise_action_info["amount"]["min"])
            else:
                fold_action_info = PlayerTemplate.get_action(valid_actions, "fold")
                action, amount = fold_action_info["action"], fold_action_info["amount"]
        else:
            raise_action_info = PlayerTemplate.get_action(valid_actions,
                                                          "raise")
            action, amount = raise_action_info["action"], raise_action_info[
                "amount"]["min"]

        return action, amount

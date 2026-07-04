from src.agents.player_template import PlayerTemplate


class RuleBasedPlayer(PlayerTemplate):
    """
    Simple rule-based baseline.

    Strategy:
    - fold to expensive calls,
    - call/check cheap actions,
    - rarely raise with minimum raise.

    This player does not use cards, opponent modelling or learning.
    It is intentionally simple and interpretable.
    """

    def __init__(self, player_name: str = "rule_based"):
        super().__init__(player_name=player_name)
        self.action_counter = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        self.action_counter += 1

        fold_action = self._find_action(valid_actions, "fold")
        call_action = self._find_action(valid_actions, "call")
        raise_action = self._find_action(valid_actions, "raise")

        call_amount = call_action["amount"] if call_action else 0

        # If checking is free, check/call.
        if call_action and call_amount == 0:
            return "call", 0

        # Fold expensive calls.
        if call_amount >= 30 and fold_action:
            return "fold", fold_action["amount"]

        # Occasionally min-raise.
        if self.action_counter % 10 == 0 and raise_action is not None:
            amount = raise_action["amount"]

            if isinstance(amount, dict):
                min_raise = amount.get("min")
                max_raise = amount.get("max")

                if min_raise is not None and max_raise is not None and min_raise != -1 and max_raise != -1:
                    return "raise", min_raise

        # Default: call/check.
        if call_action:
            return "call", call_action["amount"]

        if fold_action:
            return "fold", fold_action["amount"]

        first = valid_actions[0]
        return first["action"], first["amount"]

    @staticmethod
    def _find_action(valid_actions, action_name):
        return next((item for item in valid_actions if item["action"] == action_name), None)
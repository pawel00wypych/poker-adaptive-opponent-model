from typing import Any


class StateEncoder:
    """
    Converts PyPokerEngine round state into a discrete state for tabular RL.

    This version intentionally does not modify PyPokerEngine and does not depend
    on custom hand evaluation.
    """

    @staticmethod
    def encode(
        player_stack: int,
        valid_actions: list[dict[str, Any]],
        round_state: dict[str, Any],
        opponent_type: str = "unknown",
    ) -> tuple:
        street = StateEncoder._street(round_state)
        pot_bucket = StateEncoder._pot_bucket(round_state)
        stack_bucket = StateEncoder._stack_bucket(player_stack)
        call_bucket = StateEncoder._call_amount_bucket(valid_actions)
        opponent_id = StateEncoder._opponent_type_id(opponent_type)

        return (
            street,
            stack_bucket,
            pot_bucket,
            call_bucket,
            opponent_id,
        )

    @staticmethod
    def _street(round_state: dict[str, Any]) -> int:
        community_cards = round_state.get("community_card", [])
        count = len(community_cards)

        if count == 0:
            return 0  # preflop
        if count == 3:
            return 1  # flop
        if count == 4:
            return 2  # turn
        if count == 5:
            return 3  # river

        return 0

    @staticmethod
    def _pot_bucket(round_state: dict[str, Any]) -> int:
        pot = round_state.get("pot", {}).get("main", {}).get("amount", 0)

        if pot <= 20:
            return 0
        if pot <= 50:
            return 1
        if pot <= 100:
            return 2
        return 3

    @staticmethod
    def _stack_bucket(stack: int) -> int:
        if stack <= 25:
            return 0
        if stack <= 75:
            return 1
        if stack <= 150:
            return 2
        return 3

    @staticmethod
    def _call_amount_bucket(valid_actions: list[dict[str, Any]]) -> int:
        call_action = next((item for item in valid_actions if item["action"] == "call"), None)

        if call_action is None:
            return 0

        amount = call_action["amount"]

        if amount == 0:
            return 0
        if amount <= 10:
            return 1
        if amount <= 30:
            return 2
        return 3

    @staticmethod
    def _opponent_type_id(opponent_type: str) -> int:
        mapping = {
            "unknown": 0,
            "fish": 1,
            "aggressive": 2,
            "tight": 3,
            "balanced": 4,
            "random": 5,
        }

        return mapping.get(opponent_type, 0)
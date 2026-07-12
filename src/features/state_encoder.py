from typing import Any

from src.features.preflop_hand_encoder import PreflopHandEncoder
from src.features.hand_strength_encoder import HandStrengthEncoder


class StateEncoder:
    @staticmethod
    def encode(
        player_stack: int,
        valid_actions: list[dict[str, Any]],
        round_state: dict[str, Any],
        hole_cards: list[str],
        opponent_type: str = "unknown",
    ) -> tuple:
        street = StateEncoder._street(round_state)
        hand_bucket = HandStrengthEncoder.encode(hole_cards, round_state.get("community_card", []))
        pot_bucket = StateEncoder._pot_bucket(round_state)
        stack_bucket = StateEncoder._stack_bucket(player_stack)
        call_bucket = StateEncoder._call_amount_bucket(valid_actions)
        opponent_id = StateEncoder._opponent_type_id(opponent_type)

        return (
            street,
            hand_bucket,
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
            return 0
        if count == 3:
            return 1
        if count == 4:
            return 2
        if count == 5:
            return 3

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
        call_action = next(
            (
                item
                for item in valid_actions
                if item["action"] == "call"
            ),
            None,
        )

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
            "calling": 6,
        }

        return mapping.get(opponent_type, 0)
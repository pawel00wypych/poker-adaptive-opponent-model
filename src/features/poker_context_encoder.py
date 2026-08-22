from typing import Any

from pypokerengine.engine.card import Card


class PokerContextEncoder:
    NO_PAIR_CONTEXT = 0
    UNDER_PAIR = 1
    BOTTOM_PAIR = 2
    MIDDLE_PAIR = 3
    TOP_PAIR = 4
    OVERPAIR = 5
    TWO_PAIR_OR_BETTER = 6

    @classmethod
    def pair_strength_bucket(
        cls,
        hole_cards: list[str],
        community_cards: list[str],
        hand_strength_bin: int,
    ) -> int:
        if not community_cards:
            return cls.NO_PAIR_CONTEXT

        if hand_strength_bin >= 2:
            return cls.TWO_PAIR_OR_BETTER

        hole = [
            Card.from_str(card)
            for card in hole_cards
        ]
        board = [
            Card.from_str(card)
            for card in community_cards
        ]

        hole_ranks = [
            card.rank
            for card in hole
        ]
        board_ranks = [
            card.rank
            for card in board
        ]

        if hand_strength_bin == 0:
            return cls.NO_PAIR_CONTEXT

        if hole_ranks[0] == hole_ranks[1]:
            pocket_rank = hole_ranks[0]

            if pocket_rank > max(board_ranks):
                return cls.OVERPAIR

            return cls.UNDER_PAIR

        paired_hole_ranks = [
            rank
            for rank in hole_ranks
            if rank in board_ranks
        ]

        if not paired_hole_ranks:
            return cls.UNDER_PAIR

        paired_rank = max(paired_hole_ranks)
        unique_board_ranks = sorted(
            set(board_ranks),
            reverse=True,
        )

        if paired_rank == unique_board_ranks[0]:
            return cls.TOP_PAIR

        if (
            len(unique_board_ranks) >= 2
            and paired_rank == unique_board_ranks[-1]
        ):
            return cls.BOTTOM_PAIR

        return cls.MIDDLE_PAIR

    @staticmethod
    def pot_odds_bucket(
        valid_actions: list[dict[str, Any]],
        round_state: dict[str, Any],
    ) -> int:
        call_amount = PokerContextEncoder._call_amount(
            valid_actions
        )

        if call_amount <= 0:
            return 0

        pot = (
            round_state
            .get("pot", {})
            .get("main", {})
            .get("amount", 0)
        )

        denominator = pot + call_amount

        if denominator <= 0:
            return 0

        pot_odds = call_amount / denominator

        if pot_odds <= 0.15:
            return 1

        if pot_odds <= 0.30:
            return 2

        if pot_odds <= 0.45:
            return 3

        return 4

    @staticmethod
    def spr_bucket(
        player_stack: int,
        round_state: dict[str, Any],
    ) -> int:
        pot = (
            round_state
            .get("pot", {})
            .get("main", {})
            .get("amount", 0)
        )

        if pot <= 0:
            return 3

        spr = player_stack / pot

        if spr <= 1:
            return 0

        if spr <= 3:
            return 1

        if spr <= 6:
            return 2

        return 3

    @staticmethod
    def _call_amount(
        valid_actions: list[dict[str, Any]],
    ) -> int:
        call_action = next(
            (
                action
                for action in valid_actions
                if action["action"] == "call"
            ),
            None,
        )

        if call_action is None:
            return 0

        amount = call_action["amount"]

        if not isinstance(amount, int):
            return 0

        return amount

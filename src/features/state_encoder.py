from typing import Any

from src.features.hand_strength_encoder import (
    HandStrengthEncoder,
)
from src.features.poker_context_encoder import (
    PokerContextEncoder,
)


class StateEncoder:
    """Discretised state used by every tabular policy.

    The opponent type is deliberately not encoded. Each policy owns a separate
    Q-table and always encoded its own type, so the field was constant within
    any table: it carried no information while multiplying the nominal state
    space by five.
    """

    @staticmethod
    def encode(
        player_stack: int,
        valid_actions: list[dict[str, Any]],
        round_state: dict[str, Any],
        hole_cards: list[str],
    ) -> tuple:
        community_cards = round_state.get(
            "community_card",
            [],
        )

        street = StateEncoder._street(
            round_state
        )

        hand_strength_bin = (
            HandStrengthEncoder.encode(
                hole_cards,
                community_cards,
            )
        )

        pair_strength_bin = (
            PokerContextEncoder
            .pair_strength_bucket(
                hole_cards=hole_cards,
                community_cards=community_cards,
                hand_strength_bin=(
                    hand_strength_bin
                ),
            )
        )

        pot_bucket = StateEncoder._pot_bucket(
            round_state
        )

        pot_odds_bin = (
            PokerContextEncoder
            .pot_odds_bucket(
                valid_actions=valid_actions,
                round_state=round_state,
            )
        )

        spr_bin = (
            PokerContextEncoder.spr_bucket(
                player_stack=player_stack,
                round_state=round_state,
            )
        )

        return (
            street,
            hand_strength_bin,
            pair_strength_bin,
            pot_bucket,
            pot_odds_bin,
            spr_bin,
        )

    @staticmethod
    def _street(
        round_state: dict[str, Any],
    ) -> int:
        community_cards = round_state.get(
            "community_card",
            [],
        )

        count = len(community_cards)

        if count == 0:
            return 0

        if count == 3:
            return 1

        if count == 4:
            return 2

        if count == 5:
            return 3

        raise ValueError(
            "Unsupported number of community cards: "
            f"{count}"
        )

    @staticmethod
    def _pot_bucket(
        round_state: dict[str, Any],
    ) -> int:
        pot = (
            round_state
            .get("pot", {})
            .get("main", {})
            .get("amount", 0)
        )

        if pot <= 20:
            return 0

        if pot <= 50:
            return 1

        if pot <= 100:
            return 2

        return 3

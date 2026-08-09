from typing import Sequence

from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator

from src.features.preflop_hand_encoder import PreflopHandEncoder


class HandStrengthEncoder:
    """
    Encodes the player's hand into one discrete strength bucket.

    Preflop:
        uses PreflopHandEncoder.

    Postflop:
        uses PyPokerEngine HandEvaluator to detect the current made hand.

    The numeric scale is shared across all streets:
        0 - weak / high card
        1 - speculative preflop / one pair
        2 - medium preflop / two pair
        3 - strong preflop / three of a kind
        4 - premium preflop / straight
        5 - flush
        6 - full house
        7 - four of a kind
        8 - straight flush
    """

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8

    POSTFLOP_STRENGTH_MAP = {
        "HIGHCARD": HIGH_CARD,
        "ONEPAIR": ONE_PAIR,
        "TWOPAIR": TWO_PAIR,
        "THREECARD": THREE_OF_A_KIND,
        "STRAIGHT": STRAIGHT,
        "FLASH": FLUSH,
        "FULLHOUSE": FULL_HOUSE,
        "FOURCARD": FOUR_OF_A_KIND,
        "STRAIGHTFLASH": STRAIGHT_FLUSH,
    }

    @classmethod
    def encode(
        cls,
        hole_cards: Sequence[str],
        community_cards: Sequence[str],
    ) -> int:
        if len(hole_cards) != 2:
            raise ValueError(
                "Exactly two hole cards are required."
            )

        if not community_cards:
            return PreflopHandEncoder.encode(hole_cards)

        if len(community_cards) not in {3, 4, 5}:
            raise ValueError(
                "Community cards must contain 0, 3, 4 or 5 cards."
            )

        hole = [Card.from_str(card) for card in hole_cards]
        community = [Card.from_str(card) for card in community_cards]

        hand_info = HandEvaluator.gen_hand_rank_info(
            hole,
            community,
        )

        strength_name = hand_info["hand"]["strength"]

        try:
            return cls.POSTFLOP_STRENGTH_MAP[strength_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported hand strength: {strength_name}"
            ) from error
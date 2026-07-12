from typing import Sequence


RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


class PreflopHandEncoder:
    PREMIUM = 4
    STRONG = 3
    MEDIUM = 2
    SPECULATIVE = 1
    WEAK = 0

    @classmethod
    def encode(cls, hole_cards: Sequence[str]) -> int:
        if len(hole_cards) != 2:
            return cls.WEAK

        rank_1, suit_1 = cls._parse_card(hole_cards[0])
        rank_2, suit_2 = cls._parse_card(hole_cards[1])

        high = max(rank_1, rank_2)
        low = min(rank_1, rank_2)

        is_pair = rank_1 == rank_2
        is_suited = suit_1 == suit_2
        gap = high - low

        if is_pair and high >= 12:
            return cls.PREMIUM

        if {high, low} == {14, 13}:
            return cls.PREMIUM

        if is_pair and high >= 9:
            return cls.STRONG

        if high == 14 and low >= 11:
            return cls.STRONG

        if high == 13 and low >= 11:
            return cls.STRONG

        if is_pair:
            return cls.MEDIUM

        if high >= 11 and low >= 10:
            return cls.MEDIUM

        if is_suited and gap <= 2 and high >= 7:
            return cls.SPECULATIVE

        if high == 14 and is_suited:
            return cls.SPECULATIVE

        return cls.WEAK

    @staticmethod
    def _parse_card(card: str) -> tuple[int, str]:
        if len(card) != 2:
            raise ValueError(f"Unsupported card format: {card}")

        # PyPokerEngine commonly uses suit first, for example HA, ST, D7.
        suit = card[0]
        rank = card[1]

        if rank not in RANK_TO_VALUE:
            raise ValueError(f"Unsupported card rank: {rank}")

        return RANK_TO_VALUE[rank], suit
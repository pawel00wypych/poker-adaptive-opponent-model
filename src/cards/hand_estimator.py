from enum import Enum

class HandStrength(Enum):
    VERY_STRONG_CARDS = 4
    STRONG_CARDS = 3
    MEDIUM_CARDS = 2
    WEAK_CARDS = 1

class HandEstimator:

    def __init__(self):
        """
        Full 169-hand preflop matrix compressed into 3 tiers

        standard notation:
        - 'AA' = pair
        - 'AKs" = suited
        - 'AKo' = offsuit
        """
        self.VERY_STRONG_PREFLOP_CARDS = [
            "AA", "KK", "QQ", "JJ",
            "AKs", "AQs", "AJs",
            "KQs"
        ]

        self.STRONG_PREFLOP_CARDS = [
            "TT", "99", "88",
            "AKo",
            "ATs", "A9s", "A8s",
            "KJs", "KTs", "QJs", "QTs",
            "JTs",
            "KQo"
        ]

        self.MEDIUM_PREFLOP_CARDS = [
            "77", "66", "55",
            "AQo", "AJo", "ATo",
            "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
            "K9s", "K8s",
            "Q9s", "Q8s",
            "J9s", "J8s",
            "T9s", "98s", "87s", "76s",
            "KJo", "QJo"
        ]

    def check_preflop_hole_card_strength(self, hole_card):
        card1, card2 = hole_card[0], hole_card[1]
        hand = self.to_notation(card1, card2)

        if hand in self.VERY_STRONG_PREFLOP_CARDS:
            return HandStrength.VERY_STRONG_CARDS
        elif hand in self.STRONG_PREFLOP_CARDS:
            return HandStrength.STRONG_CARDS
        elif hand in self.MEDIUM_PREFLOP_CARDS:
            return HandStrength.MEDIUM_CARDS
        else:
            return HandStrength.WEAK_CARDS

    """
    Convert hole cards to standard notation
    """
    def to_notation(self, card1, card2):
        ranks = "23456789TJQKA"

        r1, s1 = card1[1], card1[0]
        r2, s2 = card2[1], card2[0]

        # sort by rank strength
        if ranks.index(r1) < ranks.index(r2):
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        if r1 == r2:
            return r1 + r2  # pair

        suited = "s" if s1 == s2 else "o"
        return r1 + r2 + suited


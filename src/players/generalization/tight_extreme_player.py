import random

from src.players.opponents.tight_player import (
    MEDIUM_HAND_STRENGTH_BIN,
    PREMIUM_HAND_STRENGTH_BIN,
    SPECULATIVE_HAND_STRENGTH_BIN,
    STRONG_HAND_STRENGTH_BIN,
    WEAK_HAND_STRENGTH_BIN,
    TightPlayer,
)


class TightExtremePlayer(TightPlayer):
    """
    Held-out tight-family stress-test opponent.

    Compared with TightPlayer, this player is more selective and more
    fold-heavy: it continues only with clearly stronger hands, folds expensive
    calls more often, and raises only rarely with strong or premium hands.
    """

    def __init__(
        self,
        player_name: str = "tight_extreme",
        rng: random.Random | None = None,
    ):
        super().__init__(
            player_name=player_name,
            rng=rng,
            max_call_stack_ratio=0.12,
            strong_raise_probability=0.3,
            premium_raise_probability=0.55,
        )

    def _continue_probability(self, hand_strength: int) -> float:
        if hand_strength <= WEAK_HAND_STRENGTH_BIN:
            return 0.02
        if hand_strength <= SPECULATIVE_HAND_STRENGTH_BIN:
            return 0.08
        if hand_strength <= MEDIUM_HAND_STRENGTH_BIN:
            return 0.24
        if hand_strength <= STRONG_HAND_STRENGTH_BIN:
            return 0.58
        if hand_strength <= PREMIUM_HAND_STRENGTH_BIN:
            return 0.78
        return 0.86

    def _fold_expensive_probability(self, hand_strength: int) -> float:
        if hand_strength <= WEAK_HAND_STRENGTH_BIN:
            return 0.98
        if hand_strength <= SPECULATIVE_HAND_STRENGTH_BIN:
            return 0.92
        if hand_strength <= MEDIUM_HAND_STRENGTH_BIN:
            return 0.78
        if hand_strength <= STRONG_HAND_STRENGTH_BIN:
            return 0.35
        return 0.15

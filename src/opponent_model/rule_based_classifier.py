from src.features.opponent_stats import OpponentStats
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_BALANCED,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


class RuleBasedOpponentClassifier:
    """
    Simple opponent classifier based on observed action frequencies.

    This is intentionally interpretable and suitable as a baseline for a thesis.
    """

    def __init__(self, min_actions: int = 5):
        self.min_actions = min_actions

    def classify(self, stats: OpponentStats) -> str:
        if stats.total_actions < self.min_actions:
            return OPPONENT_TYPE_UNKNOWN

        if stats.raise_rate >= 0.35:
            return OPPONENT_TYPE_AGGRESSIVE

        if stats.call_rate >= 0.80 and stats.raise_rate <= 0.05:
            return OPPONENT_TYPE_CALLING

        if stats.call_rate >= 0.50 and stats.fold_rate >= 0.30:
            return OPPONENT_TYPE_FISH

        if (stats.fold_rate >= 0.55 and stats.raise_rate >= 0.10 and
                stats.call_rate <= 0.25):
            return OPPONENT_TYPE_TIGHT

        return OPPONENT_TYPE_BALANCED

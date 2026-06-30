from src.features.opponent_stats import OpponentStats


class RuleBasedOpponentClassifier:
    """
    Simple opponent classifier based on observed action frequencies.

    This is intentionally interpretable and suitable as a baseline for a thesis.
    """

    def __init__(self, min_actions: int = 20):
        self.min_actions = min_actions

    def classify(self, stats: OpponentStats) -> str:
        if stats.total_actions < self.min_actions:
            return "unknown"

        if stats.raise_rate >= 0.35:
            return "aggressive"

        if stats.call_rate >= 0.55:
            return "fish"

        if stats.fold_rate >= 0.55:
            return "tight"

        return "balanced"
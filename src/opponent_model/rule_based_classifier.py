from src.features.opponent_stats import OpponentStats


class RuleBasedOpponentClassifier:
    """
    Simple opponent classifier based on observed action frequencies.

    This is intentionally interpretable and suitable as a baseline for a thesis.
    """

    def __init__(self, min_actions: int = 5):
        self.min_actions = min_actions

    def classify(self, stats: OpponentStats) -> str:
        if stats.total_actions < self.min_actions:
            return "unknown"

        if stats.raise_rate >= 0.35:
            return "aggressive"

        if stats.call_rate >= 0.80 and stats.raise_rate <= 0.05:
            return "calling"

        if stats.call_rate >= 0.50 and stats.fold_rate >= 0.30:
            return "fish"

        if (stats.fold_rate >= 0.55 and stats.raise_rate >= 0.10 and
                stats.call_rate <= 0.25):
            return "tight"

        return 'balanced'
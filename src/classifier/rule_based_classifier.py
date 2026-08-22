from src.features.opponent_stats import OpponentStats
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_OTHER,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


class RuleBasedOpponentClassifier:
    """Interpretable opponent classifier based on observed action frequencies.

    Two statistics carry the decision:

    ``raise_rate``
        separates the aggressive family from everything else.

    ``fold_to_bet_rate``
        separates tight from passive. Overall fold rate cannot do this: about
        three quarters of decisions in this environment are free checks, so a
        tight opponent that is never bet into simply checks and looks like a
        calling station. Conditioning on decisions that actually cost chips
        gives a wide, stable margin - measured across passive, aggressive and
        mixed opponents, tight families sit at 0.78-0.94 and calling families
        at 0.07-0.16.

    A verdict is withheld until enough pressure has been applied, because
    fold-to-bet is undefined before the opponent has faced a bet.
    """

    def __init__(
        self,
        min_actions: int = 5,
        min_bet_decisions: int = 3,
        aggressive_raise_rate: float = 0.35,
        tight_fold_to_bet_rate: float = 0.45,
        calling_fold_to_bet_rate: float = 0.30,
        calling_max_raise_rate: float = 0.15,
    ):
        self.min_actions = min_actions
        self.min_bet_decisions = min_bet_decisions
        self.aggressive_raise_rate = aggressive_raise_rate
        self.tight_fold_to_bet_rate = tight_fold_to_bet_rate
        self.calling_fold_to_bet_rate = calling_fold_to_bet_rate
        self.calling_max_raise_rate = calling_max_raise_rate

    def classify(self, stats: OpponentStats) -> str:
        if stats.total_actions < self.min_actions:
            return OPPONENT_TYPE_UNKNOWN

        if stats.raise_rate >= self.aggressive_raise_rate:
            return OPPONENT_TYPE_AGGRESSIVE

        if stats.decisions_facing_a_bet < self.min_bet_decisions:
            # The opponent has not been put under pressure yet, so there is no
            # evidence to tell a tight player from a passive one.
            return OPPONENT_TYPE_UNKNOWN

        if stats.fold_to_bet_rate >= self.tight_fold_to_bet_rate:
            return OPPONENT_TYPE_TIGHT

        if (
            stats.fold_to_bet_rate <= self.calling_fold_to_bet_rate
            and stats.raise_rate <= self.calling_max_raise_rate
        ):
            return OPPONENT_TYPE_CALLING

        return OPPONENT_TYPE_OTHER

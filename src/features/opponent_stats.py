from dataclasses import dataclass


@dataclass
class OpponentStats:
    """Behavioural counters used to classify an opponent.

    Checks are counted separately from calls. PyPokerEngine reports both as a
    ``call`` action, and in this environment roughly three quarters of all
    decisions are free checks, so folding them together makes every opponent
    look like a calling station and hides how it responds to pressure.

    The metric that separates a tight opponent from a passive one is therefore
    ``fold_to_bet_rate``: what it does when continuing actually costs chips.
    """

    hands_observed: int = 0
    folds: int = 0
    checks: int = 0
    calls: int = 0
    raises: int = 0

    def update_action(self, action: str, paid: int | None = None) -> None:
        """Record one observed action.

        ``paid`` is the number of chips the action added. A ``call`` that added
        nothing is a check. When ``paid`` is unknown the action is recorded as
        a paid call, which keeps the counters conservative rather than
        inventing free checks.
        """
        if action == "fold":
            self.folds += 1
        elif action == "call":
            if paid is not None and paid <= 0:
                self.checks += 1
            else:
                self.calls += 1
        elif action == "raise":
            self.raises += 1

    def finish_hand(self) -> None:
        self.hands_observed += 1

    @property
    def total_actions(self) -> int:
        return self.folds + self.checks + self.calls + self.raises

    @property
    def decisions_facing_a_bet(self) -> int:
        """Decisions where staying in the hand cost chips.

        Raises are excluded because a raise may open the betting rather than
        answer it, and the two cannot be told apart from the action alone.
        """
        return self.folds + self.calls

    def _rate(self, count: int) -> float:
        return count / self.total_actions if self.total_actions > 0 else 0.0

    @property
    def fold_rate(self) -> float:
        return self._rate(self.folds)

    @property
    def check_rate(self) -> float:
        return self._rate(self.checks)

    @property
    def call_rate(self) -> float:
        """Share of paid calls, excluding free checks."""
        return self._rate(self.calls)

    @property
    def passive_rate(self) -> float:
        """Share of actions that neither folded nor raised."""
        return self._rate(self.checks + self.calls)

    @property
    def raise_rate(self) -> float:
        return self._rate(self.raises)

    @property
    def fold_to_bet_rate(self) -> float:
        """How often the opponent gives up once continuing costs chips."""
        facing = self.decisions_facing_a_bet
        return self.folds / facing if facing > 0 else 0.0

    @property
    def aggression_ratio(self) -> float:
        passive_actions = self.checks + self.calls + self.folds
        return (
            self.raises / passive_actions
            if passive_actions > 0
            else float(self.raises)
        )

    def as_dict(self) -> dict:
        return {
            "hands_observed": self.hands_observed,
            "folds": self.folds,
            "checks": self.checks,
            "calls": self.calls,
            "raises": self.raises,
            "fold_rate": self.fold_rate,
            "check_rate": self.check_rate,
            "call_rate": self.call_rate,
            "passive_rate": self.passive_rate,
            "raise_rate": self.raise_rate,
            "fold_to_bet_rate": self.fold_to_bet_rate,
            "decisions_facing_a_bet": self.decisions_facing_a_bet,
            "aggression_ratio": self.aggression_ratio,
        }

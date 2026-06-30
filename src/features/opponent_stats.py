from dataclasses import dataclass


@dataclass
class OpponentStats:
    hands_observed: int = 0
    folds: int = 0
    calls: int = 0
    raises: int = 0

    def update_action(self, action: str) -> None:
        if action == "fold":
            self.folds += 1
        elif action == "call":
            self.calls += 1
        elif action == "raise":
            self.raises += 1

    def finish_hand(self) -> None:
        self.hands_observed += 1

    @property
    def total_actions(self) -> int:
        return self.folds + self.calls + self.raises

    @property
    def fold_rate(self) -> float:
        return self.folds / self.total_actions if self.total_actions > 0 else 0.0

    @property
    def call_rate(self) -> float:
        return self.calls / self.total_actions if self.total_actions > 0 else 0.0

    @property
    def raise_rate(self) -> float:
        return self.raises / self.total_actions if self.total_actions > 0 else 0.0

    @property
    def aggression_ratio(self) -> float:
        passive_actions = self.calls + self.folds
        return self.raises / passive_actions if passive_actions > 0 else float(self.raises)

    def as_dict(self) -> dict:
        return {
            "hands_observed": self.hands_observed,
            "folds": self.folds,
            "calls": self.calls,
            "raises": self.raises,
            "fold_rate": self.fold_rate,
            "call_rate": self.call_rate,
            "raise_rate": self.raise_rate,
            "aggression_ratio": self.aggression_ratio,
        }
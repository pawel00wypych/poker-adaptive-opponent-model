"""Counters describing how much of a decision was actually learned.

Tabular Q-values start at zero and unvisited entries are indistinguishable
from entries that were visited and evaluated as worth zero. Two consequences
follow, and both are measured here rather than corrected:

1. In a state where nothing was ever learned, every legal action ties at zero
   and the greedy choice degenerates into a uniform random pick, even in
   evaluation mode with epsilon set to zero.
2. In a partially explored state, an action that was never tried still holds
   zero, which outranks any action that was tried and learned to be losing.

Neither behaviour is changed by these counters. They exist so that reported
results can state how large the effect is instead of leaving it unknown.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass
class DecisionDiagnostics:
    decisions: int = 0
    unseen_state_decisions: int = 0
    untried_action_selections: int = 0

    def reset(self) -> None:
        self.decisions = 0
        self.unseen_state_decisions = 0
        self.untried_action_selections = 0

    def record(
        self,
        *,
        visit_counts: Sequence[int],
        action_id: int,
    ) -> None:
        """Classify one decision using visit counts taken before the update.

        Visit counts are used instead of table membership because reading a
        Q-table inserts the state; a state can therefore be present while
        nothing was ever learned about it.
        """
        self.decisions += 1

        if sum(visit_counts) == 0:
            self.unseen_state_decisions += 1
            return

        if visit_counts[action_id] == 0:
            self.untried_action_selections += 1

    @property
    def unseen_state_decision_rate(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.unseen_state_decisions / self.decisions

    @property
    def untried_action_selection_rate(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.untried_action_selections / self.decisions


def merge_decision_diagnostics(
    diagnostics: Iterable[DecisionDiagnostics],
) -> DecisionDiagnostics:
    """Combine counters from every policy a player may switch between."""
    merged = DecisionDiagnostics()

    for item in diagnostics:
        merged.decisions += item.decisions
        merged.unseen_state_decisions += item.unseen_state_decisions
        merged.untried_action_selections += item.untried_action_selections

    return merged

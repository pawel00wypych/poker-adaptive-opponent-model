"""How much of the state space a trained policy actually reaches.

The nominal state space is the cross-product of every bucket the encoder can
emit. It is an upper bound only: the buckets are not independent - a preflop
state cannot hold a postflop hand rank, and pair strength is constrained by
hand strength - so the reachable space is far smaller than the product suggests.

Reporting both numbers matters. The nominal figure alone invites the conclusion
that the training budget is hopeless, while the reached figure combined with
``unseen_state_decision_rate`` shows whether that actually harms decisions.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from src.features.hand_strength_encoder import HandStrengthEncoder
from src.features.poker_context_encoder import PokerContextEncoder
from src.poker.action_mapper import ActionMapper

# Bucket counts are derived from the encoders wherever they define named
# levels, so the documented state-space size cannot drift away from the code.
STATE_FIELD_BUCKETS: dict[str, int] = {
    "street": 4,
    "hand_strength_bin": HandStrengthEncoder.STRAIGHT_FLUSH + 1,
    "pair_strength_bin": PokerContextEncoder.TWO_PAIR_OR_BETTER + 1,
    "pot_bucket": 4,
    "pot_odds_bin": 5,
    "spr_bin": 4,
    "is_small_blind": 2,
}


def nominal_state_space_size() -> int:
    size = 1
    for buckets in STATE_FIELD_BUCKETS.values():
        size *= buckets
    return size


def nominal_state_action_space_size() -> int:
    return nominal_state_space_size() * ActionMapper.NUM_ACTIONS


@dataclass(frozen=True)
class StateCoverage:
    """Coverage of one Q-table, counted from visit counts rather than keys.

    Reading a Q-table inserts the state, so the number of keys overstates what
    was learned. Only entries with a recorded visit count as reached.
    """

    table_entries: int
    reached_states: int
    reached_state_actions: int
    nominal_states: int
    nominal_state_actions: int

    @property
    def state_coverage(self) -> float:
        if self.nominal_states == 0:
            return 0.0
        return self.reached_states / self.nominal_states

    @property
    def state_action_coverage(self) -> float:
        if self.nominal_state_actions == 0:
            return 0.0
        return self.reached_state_actions / self.nominal_state_actions

    @property
    def mean_actions_tried_per_reached_state(self) -> float:
        if self.reached_states == 0:
            return 0.0
        return self.reached_state_actions / self.reached_states

    def to_dict(self) -> dict[str, float | int]:
        return {
            "table_entries": self.table_entries,
            "reached_states": self.reached_states,
            "reached_state_actions": self.reached_state_actions,
            "nominal_states": self.nominal_states,
            "nominal_state_actions": self.nominal_state_actions,
            "state_coverage": self.state_coverage,
            "state_action_coverage": self.state_action_coverage,
            "mean_actions_tried_per_reached_state": (
                self.mean_actions_tried_per_reached_state
            ),
        }


def describe_state_coverage(
    q_table: Mapping,
    visit_counts: Mapping,
) -> StateCoverage:
    reached_states = 0
    reached_state_actions = 0

    for state in q_table:
        counts = visit_counts.get(state) if hasattr(visit_counts, "get") else None

        if not counts:
            continue

        tried = sum(1 for count in counts if count > 0)

        if tried:
            reached_states += 1
            reached_state_actions += tried

    return StateCoverage(
        table_entries=len(q_table),
        reached_states=reached_states,
        reached_state_actions=reached_state_actions,
        nominal_states=nominal_state_space_size(),
        nominal_state_actions=nominal_state_action_space_size(),
    )

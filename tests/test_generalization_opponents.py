import random

import pytest

from src.players.aggressive_variant_player import AggressiveExtremePlayer
from src.players.constants import (
    GENERALIZATION_OPPONENTS,
    OPPONENT_AGGRESSIVE_EXTREME,
    OPPONENT_CALLING_EXTREME,
    OPPONENT_TIGHT_EXTREME,
)
from src.players.generalization_opponents import (
    build_generalization_opponent_player,
    get_generalization_opponent_base_type,
    was_generalization_opponent_seen_during_training,
)
from src.players.calling_extreme_player import CallingExtremePlayer
from src.players.tight_extreme_player import TightExtremePlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
)


VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 100}},
]


def round_state(
    *,
    street: str = "preflop",
    player_stack: int = 100,
    community_card: list[str] | None = None,
) -> dict:
    return {
        "street": street,
        "community_card": community_card if community_card is not None else [],
        "seats": [
            {
                "name": "variant",
                "uuid": "uuid-variant",
                "stack": player_stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": 100,
                "state": "participating",
            },
        ],
    }


def test_build_generalization_opponent_builds_calling_extreme():
    player = build_generalization_opponent_player(
        OPPONENT_CALLING_EXTREME,
        rng=random.Random(1),
    )

    assert isinstance(player, CallingExtremePlayer)
    assert player.player_name == OPPONENT_CALLING_EXTREME


def test_build_generalization_opponent_builds_aggressive_extreme():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_EXTREME,
        rng=random.Random(1),
    )

    assert isinstance(player, AggressiveExtremePlayer)
    assert player.player_name == OPPONENT_AGGRESSIVE_EXTREME


def test_build_generalization_opponent_builds_tight_extreme():
    player = build_generalization_opponent_player(
        OPPONENT_TIGHT_EXTREME,
        rng=random.Random(1),
    )

    assert isinstance(player, TightExtremePlayer)
    assert player.player_name == OPPONENT_TIGHT_EXTREME


def test_generalization_opponent_factory_supports_all_defaults():
    players = [
        build_generalization_opponent_player(
            opponent_name,
            rng=random.Random(1),
        )
        for opponent_name in GENERALIZATION_OPPONENTS
    ]

    assert len(players) == 3
    assert [player.player_name for player in players] == list(
        GENERALIZATION_OPPONENTS
    )


def test_generalization_opponent_to_base_type_mapping():
    assert (
        get_generalization_opponent_base_type(OPPONENT_CALLING_EXTREME)
        == OPPONENT_TYPE_CALLING
    )
    assert (
        get_generalization_opponent_base_type(OPPONENT_AGGRESSIVE_EXTREME)
        == OPPONENT_TYPE_AGGRESSIVE
    )
    assert (
        get_generalization_opponent_base_type(OPPONENT_TIGHT_EXTREME)
        == OPPONENT_TYPE_TIGHT
    )


def test_generalization_opponents_are_held_out_from_training():
    for opponent_name in GENERALIZATION_OPPONENTS:
        assert was_generalization_opponent_seen_during_training(opponent_name) is False


def test_calling_extreme_folds_weak_expensive_call():
    player = build_generalization_opponent_player(
        OPPONENT_CALLING_EXTREME,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["S2", "D7"],
        round_state=round_state(player_stack=100),
    )

    assert action == "fold"
    assert amount == 0


def test_calling_extreme_remains_passive_with_ordinary_call():
    player = build_generalization_opponent_player(
        OPPONENT_CALLING_EXTREME,
        rng=random.Random(5),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["S2", "D7"],
        round_state=round_state(player_stack=100),
    )

    assert action == "call"
    assert amount == 10


def test_tight_extreme_folds_weak_expensive_call():
    player = build_generalization_opponent_player(
        OPPONENT_TIGHT_EXTREME,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["S2", "D7"],
        round_state=round_state(player_stack=100),
    )

    assert action == "fold"
    assert amount == 0


def test_tight_extreme_can_continue_with_premium_hand():
    player = build_generalization_opponent_player(
        OPPONENT_TIGHT_EXTREME,
        rng=random.Random(5),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 10},
        ],
        hole_card=["SA", "DA"],
        round_state=round_state(player_stack=100),
    )

    assert action == "call"
    assert amount == 10


def test_aggressive_extreme_is_not_deterministic_always_raise():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_EXTREME,
        rng=random.Random(19),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["S2", "D7"],
        round_state=round_state(street="preflop"),
    )

    assert action == "call"
    assert amount == 10


def test_aggressive_extreme_can_raise_to_max_amount():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_EXTREME,
        rng=random.Random(28),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(street="flop"),
    )

    assert action == "raise"
    assert amount == 100


def test_generalization_factory_rejects_unknown_opponent():
    with pytest.raises(ValueError):
        build_generalization_opponent_player("unknown_variant")


def test_base_type_mapping_rejects_unknown_opponent():
    with pytest.raises(ValueError):
        get_generalization_opponent_base_type("unknown_variant")

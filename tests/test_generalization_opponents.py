import random

import pytest

from src.players.aggressive_variant_player import (
    AggressiveExtremePlayer,
    AggressiveLightPlayer,
)
from src.players.calling_player import CallingPlayer
from src.players.constants import (
    AGGRESSIVE_GENERALIZATION_OPPONENTS,
    GENERALIZATION_OPPONENTS,
    OPPONENT_AGGRESSIVE_EXTREME,
    OPPONENT_AGGRESSIVE_LIGHT,
    OPPONENT_STRONG_CALLING,
)
from src.players.generalization_opponents import (
    build_generalization_opponent_player,
    get_generalization_opponent_base_type,
    was_generalization_opponent_seen_during_training,
)
from src.players.strong_calling_player import StrongCallingPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
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
) -> dict:
    return {
        "street": street,
        "community_card": [],
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


def test_build_generalization_opponent_builds_base_calling_reference():
    player = build_generalization_opponent_player(
        OPPONENT_TYPE_CALLING,
        rng=random.Random(1),
    )

    assert isinstance(player, CallingPlayer)
    assert player.player_name == OPPONENT_TYPE_CALLING


def test_build_generalization_opponent_builds_strong_calling():
    player = build_generalization_opponent_player(
        OPPONENT_STRONG_CALLING,
        rng=random.Random(1),
    )

    assert isinstance(player, StrongCallingPlayer)
    assert player.player_name == OPPONENT_STRONG_CALLING


def test_build_generalization_opponent_builds_aggressive_light():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_LIGHT,
        rng=random.Random(1),
    )

    assert isinstance(player, AggressiveLightPlayer)
    assert player.player_name == OPPONENT_AGGRESSIVE_LIGHT


def test_build_generalization_opponent_builds_aggressive_extreme():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_EXTREME,
        rng=random.Random(1),
    )

    assert isinstance(player, AggressiveExtremePlayer)
    assert player.player_name == OPPONENT_AGGRESSIVE_EXTREME


def test_generalization_opponent_factory_supports_all_defaults():
    players = [
        build_generalization_opponent_player(
            opponent_name,
            rng=random.Random(1),
        )
        for opponent_name in GENERALIZATION_OPPONENTS
    ]

    assert len(players) == 4
    assert [player.player_name for player in players] == list(
        GENERALIZATION_OPPONENTS
    )


def test_generalization_opponent_to_base_type_mapping():
    assert get_generalization_opponent_base_type(OPPONENT_TYPE_CALLING) == OPPONENT_TYPE_CALLING
    assert get_generalization_opponent_base_type(OPPONENT_STRONG_CALLING) == OPPONENT_TYPE_CALLING

    for opponent_name in AGGRESSIVE_GENERALIZATION_OPPONENTS:
        assert get_generalization_opponent_base_type(opponent_name) == OPPONENT_TYPE_AGGRESSIVE


def test_base_calling_reference_is_marked_as_seen_during_training():
    assert was_generalization_opponent_seen_during_training(OPPONENT_TYPE_CALLING) is True
    assert was_generalization_opponent_seen_during_training(OPPONENT_STRONG_CALLING) is False


def test_strong_calling_folds_weak_expensive_call():
    player = build_generalization_opponent_player(
        OPPONENT_STRONG_CALLING,
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


def test_strong_calling_remains_passive_with_ordinary_call():
    player = build_generalization_opponent_player(
        OPPONENT_STRONG_CALLING,
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


def test_aggressive_light_raises_when_roll_is_below_threshold():
    player = build_generalization_opponent_player(
        OPPONENT_AGGRESSIVE_LIGHT,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(street="flop"),
    )

    assert action == "raise"
    assert amount == 20


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

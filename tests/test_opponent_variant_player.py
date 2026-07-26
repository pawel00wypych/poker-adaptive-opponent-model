import random

import pytest

from src.players.opponent_variant_player import (
    AGGRESSIVE_OPPONENT_VARIANTS,
    CALLING_OPPONENT_VARIANTS,
    GENERALIZATION_OPPONENT_VARIANTS,
    OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
    OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
    OPPONENT_VARIANT_CALLING_MEDIUM,
    OPPONENT_VARIANT_CALLING_STRONG,
    OPPONENT_VARIANT_CALLING_WEAK,
    AggressiveVariantPlayer,
    CallingVariantPlayer,
    build_aggressive_variant_player,
    build_calling_variant_player,
    build_opponent_variant,
    get_opponent_variant_base_type,
)
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)


VALID_ACTIONS = [
    {
        "action": "fold",
        "amount": 0,
    },
    {
        "action": "call",
        "amount": 10,
    },
    {
        "action": "raise",
        "amount": {
            "min": 20,
            "max": 100,
        },
    },
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


def test_build_calling_variant_player_uses_requested_variant_name():
    player = build_calling_variant_player(
        OPPONENT_VARIANT_CALLING_WEAK,
        rng=random.Random(1),
    )

    assert isinstance(player, CallingVariantPlayer)
    assert player.player_name == OPPONENT_VARIANT_CALLING_WEAK
    assert player.config.name == OPPONENT_VARIANT_CALLING_WEAK


def test_build_aggressive_variant_player_uses_requested_variant_name():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
        rng=random.Random(1),
    )

    assert isinstance(player, AggressiveVariantPlayer)
    assert player.player_name == OPPONENT_VARIANT_AGGRESSIVE_EXTREME
    assert player.config.name == OPPONENT_VARIANT_AGGRESSIVE_EXTREME


def test_build_opponent_variant_supports_all_generalization_variants():
    players = [
        build_opponent_variant(
            variant_name,
            rng=random.Random(1),
        )
        for variant_name in GENERALIZATION_OPPONENT_VARIANTS
    ]

    assert len(players) == 5
    assert [player.player_name for player in players] == list(
        GENERALIZATION_OPPONENT_VARIANTS
    )


def test_variant_to_base_type_mapping():
    for variant_name in CALLING_OPPONENT_VARIANTS:
        assert get_opponent_variant_base_type(variant_name) == OPPONENT_TYPE_CALLING

    for variant_name in AGGRESSIVE_OPPONENT_VARIANTS:
        assert get_opponent_variant_base_type(variant_name) == OPPONENT_TYPE_AGGRESSIVE


def test_calling_weak_calls_when_call_is_available():
    player = build_calling_variant_player(
        OPPONENT_VARIANT_CALLING_WEAK,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(),
    )

    assert action == "call"
    assert amount == 10


def test_calling_strong_can_fold_expensive_call():
    player = build_calling_variant_player(
        OPPONENT_VARIANT_CALLING_STRONG,
        rng=random.Random(3),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(player_stack=20),
    )

    assert action == "fold"
    assert amount == 0


def test_calling_medium_can_make_rare_raise():
    player = build_calling_variant_player(
        OPPONENT_VARIANT_CALLING_MEDIUM,
        rng=random.Random(31),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["HA", "HK"],
        round_state=round_state(),
    )

    assert action == "raise"
    assert amount == 20


def test_aggressive_light_raises_when_roll_is_below_street_threshold():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
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


def test_aggressive_light_calls_when_roll_is_between_raise_and_call_thresholds():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
        rng=random.Random(7),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(street="preflop"),
    )

    assert action == "call"
    assert amount == 10


def test_aggressive_extreme_is_not_deterministic_always_raise():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
        rng=random.Random(19),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state=round_state(street="preflop"),
    )

    assert action == "call"
    assert amount == 10


def test_aggressive_variant_raises_more_often_with_strong_hand():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
        rng=random.Random(5),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["HA", "HK"],
        round_state=round_state(street="preflop"),
    )

    assert action == "raise"
    assert amount == 20


def test_aggressive_variant_raises_less_often_with_weak_hand():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_LIGHT,
        rng=random.Random(5),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["S2", "D7"],
        round_state=round_state(street="preflop"),
    )

    assert action == "fold"
    assert amount == 0


def test_calling_variant_can_value_raise_strong_hand():
    player = build_calling_variant_player(
        OPPONENT_VARIANT_CALLING_STRONG,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["HA", "HK"],
        round_state=round_state(street="preflop"),
    )

    assert action == "raise"
    assert amount == 20


def test_aggressive_extreme_can_raise_to_max_amount():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
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


def test_variant_falls_back_to_call_when_raise_is_invalid():
    player = build_aggressive_variant_player(
        OPPONENT_VARIANT_AGGRESSIVE_EXTREME,
        rng=random.Random(1),
    )
    player.uuid = "uuid-variant"

    action, amount = player.declare_action(
        valid_actions=[
            {
                "action": "fold",
                "amount": 0,
            },
            {
                "action": "call",
                "amount": 10,
            },
            {
                "action": "raise",
                "amount": {
                    "min": -1,
                    "max": -1,
                },
            },
        ],
        hole_card=[],
        round_state=round_state(),
    )

    assert action == "call"
    assert amount == 10


def test_build_opponent_variant_rejects_unknown_variant():
    with pytest.raises(ValueError):
        build_opponent_variant("unknown_variant")


def test_get_opponent_variant_base_type_rejects_unknown_variant():
    with pytest.raises(ValueError):
        get_opponent_variant_base_type("unknown_variant")

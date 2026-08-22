import random

import pytest

from src.players.baselines.always_call_player import AlwaysCallPlayer
from src.players.generalization.calling_extreme_player import CallingExtremePlayer
from src.players.opponents.calling_player import CallingPlayer

VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 100}},
]


class FixedRandom:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


def test_calling_player_calls_when_call_available():
    player = CallingPlayer(rng=FixedRandom(0.50))

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 10


def test_calling_player_checks_when_call_amount_is_zero():
    rng = FixedRandom(0.00)
    player = CallingPlayer(rng=rng)

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 0},
            {"action": "raise", "amount": {"min": 10, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 0
    assert rng.calls == 0


def test_calling_player_can_make_rare_minimum_raise():
    player = CallingPlayer(rng=FixedRandom(0.01))

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state={},
    )

    assert action == "raise"
    assert amount == 20


def test_calling_player_sometimes_folds_paid_call():
    player = CallingPlayer(rng=FixedRandom(0.95))

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=[],
        round_state={},
    )

    assert action == "fold"
    assert amount == 0


def test_calling_player_is_distinct_from_always_call_baseline():
    calling_action = CallingPlayer(
        rng=FixedRandom(0.95)
    ).declare_action(VALID_ACTIONS, [], {})
    always_call_action = AlwaysCallPlayer().declare_action(
        VALID_ACTIONS,
        [],
        {},
    )

    assert calling_action == ("fold", 0)
    assert always_call_action == ("call", 10)


def test_calling_player_reassigns_invalid_raise_probability_to_call():
    player = CallingPlayer(rng=FixedRandom(0.01))

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 10},
            {"action": "raise", "amount": {"min": -1, "max": -1}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "call"
    assert amount == 10


def test_calling_player_folds_when_call_not_available():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "fold"
    assert amount == 0


def test_calling_player_uses_first_action_as_last_fallback():
    player = CallingPlayer()

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "raise", "amount": {"min": 20, "max": 100}},
        ],
        hole_card=[],
        round_state={},
    )

    assert action == "raise"
    assert amount == {"min": 20, "max": 100}


def test_calling_player_default_rng_respects_global_seed():
    random.seed(123)
    first_player = CallingPlayer()
    first_actions = [
        first_player.declare_action(VALID_ACTIONS, [], {})
        for _ in range(20)
    ]

    random.seed(123)
    second_player = CallingPlayer()
    second_actions = [
        second_player.declare_action(VALID_ACTIONS, [], {})
        for _ in range(20)
    ]

    assert first_actions == second_actions


@pytest.mark.parametrize(
    ("call_probability", "raise_probability"),
    [
        (-0.01, 0.02),
        (1.01, 0.00),
        (0.90, -0.01),
        (0.90, 0.11),
    ],
)
def test_calling_player_rejects_invalid_probabilities(
    call_probability,
    raise_probability,
):
    with pytest.raises(ValueError):
        CallingPlayer(
            call_probability=call_probability,
            raise_probability=raise_probability,
        )


def test_calling_player_tracks_round_results():
    player = CallingPlayer(player_name="calling")
    player.uuid = "uuid-calling"

    round_state_1 = {
        "seats": [
            {"name": "calling", "uuid": "uuid-calling", "stack": 100},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ]
    }

    round_state_2 = {
        "seats": [
            {"name": "calling", "uuid": "uuid-calling", "stack": 120},
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 80},
        ]
    }

    player.receive_round_result_message([], [], round_state_1)
    player.receive_round_result_message([], [], round_state_2)

    assert player.hands_played == 2
    assert player.total_reward_bb == 2.0


def _round_state_for_calling_extreme(player_stack: int = 100) -> dict:
    return {
        "community_card": [],
        "seats": [
            {
                "name": "calling_extreme",
                "uuid": "uuid-calling-extreme",
                "stack": player_stack,
            },
            {"name": "opponent", "uuid": "uuid-opponent", "stack": 100},
        ],
    }


def test_calling_extreme_folds_some_weak_expensive_calls():
    player = CallingExtremePlayer(rng=random.Random(1))
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["S2", "D7"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "fold"
    assert amount == 0


def test_calling_extreme_can_continue_with_strong_expensive_hand():
    player = CallingExtremePlayer(rng=random.Random(5))
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=[
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 80},
        ],
        hole_card=["HA", "HK"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "call"
    assert amount == 80


def test_calling_extreme_can_value_raise_strong_hand():
    player = CallingExtremePlayer(rng=random.Random(1), strong_raise_probability=1.0)
    player.uuid = "uuid-calling-extreme"

    action, amount = player.declare_action(
        valid_actions=VALID_ACTIONS,
        hole_card=["HA", "HK"],
        round_state=_round_state_for_calling_extreme(player_stack=100),
    )

    assert action == "raise"
    assert amount == 20

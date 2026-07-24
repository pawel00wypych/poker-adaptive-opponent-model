from pathlib import Path

from src.evaluation.checkpoint_evaluator import (
    ModelBundle,
    build_tested_player,
)
from src.evaluation.constants import SUPPORTED_TESTED_AGENTS
from src.players.always_raise_player import AlwaysRaisePlayer


def start_always_raise_round(
    player: AlwaysRaisePlayer,
    player_stack: int = 100,
    opponent_stack: int = 100,
    round_count: int = 1,
):
    player.receive_round_start_message(
        round_count=round_count,
        hole_card=["HA", "DA"],
        seats=[
            {
                "name": "always_raise",
                "uuid": "uuid-tested",
                "stack": player_stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": opponent_stack,
                "state": "participating",
            },
        ],
    )


def round_state(
    player_stack: int = 100,
    opponent_stack: int = 100,
) -> dict:
    return {
        "round_count": 1,
        "community_card": [],
        "seats": [
            {
                "name": "always_raise",
                "uuid": "uuid-tested",
                "stack": player_stack,
                "state": "participating",
            },
            {
                "name": "opponent",
                "uuid": "uuid-opponent",
                "stack": opponent_stack,
                "state": "participating",
            },
        ],
        "pot": {
            "main": {
                "amount": 15,
            }
        },
    }


def test_always_raise_player_raises_minimum_when_raise_is_legal():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

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
                    "min": 20,
                    "max": 200,
                },
            },
        ],
        hole_card=["HA", "DA"],
        round_state=round_state(),
    )

    assert action == "raise"
    assert amount == 20


def test_always_raise_player_supports_numeric_raise_amount():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

    action, amount = player.declare_action(
        valid_actions=[
            {
                "action": "call",
                "amount": 10,
            },
            {
                "action": "raise",
                "amount": 30,
            },
        ],
        hole_card=["HA", "DA"],
        round_state=round_state(),
    )

    assert action == "raise"
    assert amount == 30


def test_always_raise_player_calls_when_raise_is_unavailable():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

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
        ],
        hole_card=["HA", "DA"],
        round_state=round_state(),
    )

    assert action == "call"
    assert amount == 10


def test_always_raise_player_calls_when_raise_limits_are_invalid():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

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
        hole_card=["HA", "DA"],
        round_state=round_state(),
    )

    assert action == "call"
    assert amount == 10


def test_always_raise_player_folds_when_raise_and_call_are_unavailable():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

    action, amount = player.declare_action(
        valid_actions=[
            {
                "action": "fold",
                "amount": 0,
            },
        ],
        hole_card=["HA", "DA"],
        round_state=round_state(),
    )

    assert action == "fold"
    assert amount == 0


def test_always_raise_player_tracks_completed_rounds():
    player = AlwaysRaisePlayer()
    player.uuid = "uuid-tested"

    start_always_raise_round(
        player,
        player_stack=100,
        opponent_stack=100,
    )

    player.receive_round_result_message(
        winners=[],
        hand_info=[],
        round_state=round_state(
            player_stack=120,
            opponent_stack=80,
        ),
    )

    assert player.hands_played == 1
    assert player.stack == 120
    assert player.hand_start_stack is None


def test_always_raise_is_supported_checkpoint_baseline():
    assert "always_raise" in SUPPORTED_TESTED_AGENTS


def test_checkpoint_evaluator_builds_always_raise_player(
    tmp_path,
):
    bundle = ModelBundle(
        training_run_directory=tmp_path / "run",
        seed=42,
        checkpoint_episode=2000,
        unknown_model_path=Path("unknown.pkl"),
        fish_model_path=Path("fish.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )

    player = build_tested_player(
        tested_agent_name="always_raise",
        opponent_name="fish",
        bundle=bundle,
    )

    assert isinstance(
        player,
        AlwaysRaisePlayer,
    )
    assert player.player_name == "always_raise"

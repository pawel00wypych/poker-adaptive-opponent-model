import pytest

from src.players.opponents.aggressive_player import AggressivePlayer
from src.players.opponents.calling_player import CallingPlayer
from src.players.opponents.factory import (
    build_opponent,
    build_training_opponent,
)
from src.players.opponents.tight_player import TightPlayer


@pytest.mark.parametrize(
    ("opponent_type", "expected_class"),
    [
        ("tight", TightPlayer),
        ("aggressive", AggressivePlayer),
        ("calling", CallingPlayer),
    ],
)
def test_build_opponent_returns_correct_player(
    opponent_type,
    expected_class,
):
    opponent = build_opponent(
        opponent_type
    )

    assert isinstance(
        opponent,
        expected_class,
    )


def test_build_opponent_rejects_unsupported_type():
    with pytest.raises(
        ValueError,
        match="Unsupported opponent type",
    ):
        build_opponent(
            "other"
        )


@pytest.mark.parametrize(
    ("episode", "expected_type"),
    [
        (0, "tight"),
        (1, "aggressive"),
        (2, "calling"),
        (3, "tight"),
        (4, "aggressive"),
        (5, "calling"),
    ],
)
def test_build_training_opponent_cycles_types(
    episode,
    expected_type,
):
    opponent_type, opponent = (
        build_training_opponent(
            episode
        )
    )

    assert opponent_type == expected_type
    assert opponent is not None

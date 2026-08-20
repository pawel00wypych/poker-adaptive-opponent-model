import random

import pytest

from src.players.generalization.aggressive_extreme_player import (
    AggressiveExtremePlayer,
)
from src.players.generalization.calling_extreme_player import CallingExtremePlayer
from src.players.generalization.tight_extreme_player import TightExtremePlayer
from src.players.opponents.calling_player import CallingPlayer
from src.players.opponents.tight_player import TightPlayer


@pytest.mark.parametrize(
    "player_factory",
    [
        CallingPlayer,
        TightPlayer,
        CallingExtremePlayer,
        AggressiveExtremePlayer,
        TightExtremePlayer,
    ],
)
def test_stochastic_opponent_default_rng_uses_global_seeded_stream(
    player_factory,
):
    player = player_factory()

    assert player.rng is random

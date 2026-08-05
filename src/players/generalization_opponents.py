import random

from src.players.aggressive_variant_player import (
    AggressiveExtremePlayer,
    AggressiveLightPlayer,
)
from src.players.calling_player import CallingPlayer
from src.players.constants import (
    GENERALIZATION_OPPONENTS,
    OPPONENT_AGGRESSIVE_EXTREME,
    OPPONENT_AGGRESSIVE_LIGHT,
    OPPONENT_STRONG_CALLING,
)
from src.players.strong_calling_player import StrongCallingPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
)


GENERALIZATION_OPPONENT_TO_BASE_TYPE = {
    OPPONENT_TYPE_CALLING: OPPONENT_TYPE_CALLING,
    OPPONENT_STRONG_CALLING: OPPONENT_TYPE_CALLING,
    OPPONENT_AGGRESSIVE_LIGHT: OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_AGGRESSIVE_EXTREME: OPPONENT_TYPE_AGGRESSIVE,
}

GENERALIZATION_OPPONENT_SEEN_IN_TRAINING = {
    OPPONENT_TYPE_CALLING: True,
    OPPONENT_STRONG_CALLING: False,
    OPPONENT_AGGRESSIVE_LIGHT: False,
    OPPONENT_AGGRESSIVE_EXTREME: False,
}


def build_generalization_opponent_player(
    opponent_name: str,
    rng: random.Random | None = None,
):
    if opponent_name == OPPONENT_TYPE_CALLING:
        return CallingPlayer(
            player_name=OPPONENT_TYPE_CALLING,
        )

    if opponent_name == OPPONENT_STRONG_CALLING:
        return StrongCallingPlayer(
            player_name=OPPONENT_STRONG_CALLING,
            rng=rng,
        )

    if opponent_name == OPPONENT_AGGRESSIVE_LIGHT:
        return AggressiveLightPlayer(
            player_name=OPPONENT_AGGRESSIVE_LIGHT,
            rng=rng,
        )

    if opponent_name == OPPONENT_AGGRESSIVE_EXTREME:
        return AggressiveExtremePlayer(
            player_name=OPPONENT_AGGRESSIVE_EXTREME,
            rng=rng,
        )

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )


def get_generalization_opponent_base_type(opponent_name: str) -> str:
    if opponent_name not in GENERALIZATION_OPPONENT_TO_BASE_TYPE:
        raise ValueError(
            f"Unsupported generalization opponent: {opponent_name}. "
            f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
        )

    return GENERALIZATION_OPPONENT_TO_BASE_TYPE[opponent_name]


def was_generalization_opponent_seen_during_training(opponent_name: str) -> bool:
    if opponent_name not in GENERALIZATION_OPPONENT_SEEN_IN_TRAINING:
        raise ValueError(
            f"Unsupported generalization opponent: {opponent_name}. "
            f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
        )

    return GENERALIZATION_OPPONENT_SEEN_IN_TRAINING[opponent_name]

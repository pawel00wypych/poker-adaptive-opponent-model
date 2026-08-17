import random

from src.players.generalization.aggressive_extreme_player import AggressiveExtremePlayer
from src.players.constants import (
    GENERALIZATION_OPPONENTS,
    GENERALIZATION_OPPONENT_SEEN_IN_TRAINING,
    GENERALIZATION_OPPONENT_TO_BASE_TYPE,
    OPPONENT_AGGRESSIVE_EXTREME,
    OPPONENT_CALLING_EXTREME,
    OPPONENT_TIGHT_EXTREME,
)
from src.players.generalization.calling_extreme_player import CallingExtremePlayer
from src.players.generalization.tight_extreme_player import TightExtremePlayer


def build_generalization_opponent_player(
    opponent_name: str,
    rng: random.Random | None = None,
):
    if opponent_name == OPPONENT_CALLING_EXTREME:
        return CallingExtremePlayer(
            player_name=OPPONENT_CALLING_EXTREME,
            rng=rng,
        )

    if opponent_name == OPPONENT_AGGRESSIVE_EXTREME:
        return AggressiveExtremePlayer(
            player_name=OPPONENT_AGGRESSIVE_EXTREME,
            rng=rng,
        )

    if opponent_name == OPPONENT_TIGHT_EXTREME:
        return TightExtremePlayer(
            player_name=OPPONENT_TIGHT_EXTREME,
            rng=rng,
        )

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )


def get_generalization_opponent_base_type(opponent_name: str) -> str:
    if opponent_name in GENERALIZATION_OPPONENT_TO_BASE_TYPE:
        return GENERALIZATION_OPPONENT_TO_BASE_TYPE[opponent_name]

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )


def was_generalization_opponent_seen_during_training(opponent_name: str) -> bool:
    if opponent_name in GENERALIZATION_OPPONENT_SEEN_IN_TRAINING:
        return GENERALIZATION_OPPONENT_SEEN_IN_TRAINING[opponent_name]

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )

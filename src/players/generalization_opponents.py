import random

from src.players.aggressive_variant_player import (
    AggressiveExtremePlayer,
    AggressiveLightPlayer,
)
from src.players.constants import (
    GENERALIZATION_OPPONENTS,
    OPPONENT_AGGRESSIVE_EXTREME,
    OPPONENT_AGGRESSIVE_LIGHT,
    OPPONENT_CALLING_EXTREME,
    OPPONENT_TIGHT_EXTREME,
)
from src.players.calling_extreme_player import CallingExtremePlayer
from src.players.tight_extreme_player import TightExtremePlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
)


GENERALIZATION_OPPONENT_TO_BASE_TYPE = {
    OPPONENT_CALLING_EXTREME: OPPONENT_TYPE_CALLING,
    OPPONENT_AGGRESSIVE_EXTREME: OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TIGHT_EXTREME: OPPONENT_TYPE_TIGHT,
}

# Keep aggressive_light constructible for older ad-hoc experiments, but it is
# no longer part of the default/final generalization set.
LEGACY_GENERALIZATION_OPPONENT_TO_BASE_TYPE = {
    OPPONENT_AGGRESSIVE_LIGHT: OPPONENT_TYPE_AGGRESSIVE,
}

GENERALIZATION_OPPONENT_SEEN_IN_TRAINING = {
    OPPONENT_CALLING_EXTREME: False,
    OPPONENT_AGGRESSIVE_EXTREME: False,
    OPPONENT_TIGHT_EXTREME: False,
}


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

    if opponent_name == OPPONENT_AGGRESSIVE_LIGHT:
        return AggressiveLightPlayer(
            player_name=OPPONENT_AGGRESSIVE_LIGHT,
            rng=rng,
        )

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )


def get_generalization_opponent_base_type(opponent_name: str) -> str:
    if opponent_name in GENERALIZATION_OPPONENT_TO_BASE_TYPE:
        return GENERALIZATION_OPPONENT_TO_BASE_TYPE[opponent_name]

    if opponent_name in LEGACY_GENERALIZATION_OPPONENT_TO_BASE_TYPE:
        return LEGACY_GENERALIZATION_OPPONENT_TO_BASE_TYPE[opponent_name]

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )


def was_generalization_opponent_seen_during_training(opponent_name: str) -> bool:
    if opponent_name in GENERALIZATION_OPPONENT_SEEN_IN_TRAINING:
        return GENERALIZATION_OPPONENT_SEEN_IN_TRAINING[opponent_name]

    if opponent_name in LEGACY_GENERALIZATION_OPPONENT_TO_BASE_TYPE:
        return False

    raise ValueError(
        f"Unsupported generalization opponent: {opponent_name}. "
        f"Supported opponents: {sorted(GENERALIZATION_OPPONENTS)}"
    )

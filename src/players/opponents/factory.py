from src.players.opponents.aggressive_player import AggressivePlayer
from src.players.opponents.calling_player import CallingPlayer
from src.players.opponents.tight_player import TightPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    TRAINING_OPPONENT_TYPES,
)

SUPPORTED_OPPONENTS = set(TRAINING_OPPONENT_TYPES)


def build_opponent(opponent_type: str):
    if opponent_type == OPPONENT_TYPE_TIGHT:
        return TightPlayer(
            player_name=OPPONENT_TYPE_TIGHT,
        )

    if opponent_type == OPPONENT_TYPE_AGGRESSIVE:
        return AggressivePlayer(
            player_name=OPPONENT_TYPE_AGGRESSIVE,
        )

    if opponent_type == OPPONENT_TYPE_CALLING:
        return CallingPlayer(
            player_name=OPPONENT_TYPE_CALLING,
        )

    raise ValueError(
        f"Unsupported opponent type: {opponent_type}"
    )


def build_training_opponent(episode: int):
    opponent_types = TRAINING_OPPONENT_TYPES

    opponent_type = opponent_types[
        episode % len(opponent_types)
    ]

    return (
        opponent_type,
        build_opponent(opponent_type),
    )

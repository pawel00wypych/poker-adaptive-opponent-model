from src.players.aggressive_player import AggressivePlayer
from src.players.calling_player import CallingPlayer
from src.players.fish_player import FishPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
    TRAINING_OPPONENT_TYPES,
)


SUPPORTED_OPPONENTS = set(TRAINING_OPPONENT_TYPES)


def build_opponent(opponent_type: str):
    if opponent_type == OPPONENT_TYPE_FISH:
        return FishPlayer(
            player_name=OPPONENT_TYPE_FISH,
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

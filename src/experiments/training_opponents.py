from src.players.aggressive_player import AggressivePlayer
from src.players.calling_player import CallingPlayer
from src.players.fish_player import FishPlayer


SUPPORTED_OPPONENTS = {
    "fish",
    "aggressive",
    "calling",
}


def build_opponent(opponent_type: str):
    if opponent_type == "fish":
        return FishPlayer(
            player_name="fish",
        )

    if opponent_type == "aggressive":
        return AggressivePlayer(
            player_name="aggressive",
        )

    if opponent_type == "calling":
        return CallingPlayer(
            player_name="calling",
        )

    raise ValueError(
        f"Unsupported opponent type: {opponent_type}"
    )


def build_training_opponent(episode: int):
    opponent_types = [
        "fish",
        "aggressive",
        "calling",
    ]

    opponent_type = opponent_types[
        episode % len(opponent_types)
    ]

    return (
        opponent_type,
        build_opponent(opponent_type),
    )
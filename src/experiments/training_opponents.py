from src.agents.aggressive_player import AggressivePlayer
from src.agents.calling_player import CallingPlayer
from src.agents.fish_player import FishPlayer


def build_training_opponent(episode: int):
    opponent_type = episode % 3

    if opponent_type == 0:
        return "fish", FishPlayer(player_name="fish")

    if opponent_type == 1:
        return (
            "aggressive",
            AggressivePlayer(player_name="aggressive"),
        )

    return (
        "calling",
        CallingPlayer(player_name="calling"),
    )
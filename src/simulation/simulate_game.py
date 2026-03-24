from pypokerengine.api.game import setup_config, start_poker
from src.agents.fish_player import FishPlayer
from src.agents.aggressive_player import AggressivePlayer
import json
from pathlib import Path

# Create a Path object
result_path = Path("simulation_results/result.json")

config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1_fish", algorithm=FishPlayer(
    player_name = "p1_fish"))
config.register_player(name="p2_aggressive", algorithm=AggressivePlayer(
    player_name = "p2_aggressive"))
game_result = start_poker(config, verbose=1)

result_path.parent.mkdir(parents=True, exist_ok=True)
with result_path.open("w") as f:
    json.dump(game_result,
              f,
              indent=4)
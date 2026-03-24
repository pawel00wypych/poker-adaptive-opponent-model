from pypokerengine.api.game import setup_config, start_poker
from src.agents.fish_player import FishPlayer
from src.agents.aggressive_player import AggressivePlayer

config = setup_config(max_round=100, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1_fish", algorithm=FishPlayer("p1_fish"))
config.register_player(name="p2_aggressive", algorithm=AggressivePlayer(
    "p2_aggressive"))
game_result1 = start_poker(config, verbose=1)
game_result2 = start_poker(config, verbose=1)
print(f"\nGAME RESULT_1 = {game_result1}")
print(f"\nGAME RESULT_2 = {game_result2}")
from src.agents.simple_agent import SimpleAgent
from src.simulation.play_game_with_rl_player import play_game_with_rl_player

agent = SimpleAgent()

for episode in range(1000):
    # 1 hand = 1 episode
    play_game_with_rl_player(agent) # runs PyPokerEngine game

    if episode % 10 == 0:
        print(f"Episode {episode}, memory size: {len(agent.memory)}")
        print(f"memory: {agent.memory}")

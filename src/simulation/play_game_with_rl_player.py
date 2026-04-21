from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker
from src.agents.adaptive_player import AdaptivePlayer
from src.agents.aggressive_player import AggressivePlayer
from src.agents.fish_player import FishPlayer

def play_hand(env, agent):
    # it will be used later, when I wrap pypokerengine for gymnasium env
    transitions = []

    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)

        next_state, reward, done, info = env.step(action)

        transitions.append((state, action, reward, next_state, done))

        state = next_state

    return transitions

def play_game_with_rl_player(agent):
    # each hand = separate episode internally
    # max_round = number of hands in one game
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    config.register_player(
        name="rl_player",
        algorithm=AdaptivePlayer(
                        agent=agent,
                        player_name="rl_player"
        ))
    config.register_player(
        name="p1_fish",
        algorithm=FishPlayer(
            player_name="p1_fish"
        ))
    config.register_player(
        name="p2_aggressive",
        algorithm=AggressivePlayer(
            player_name="p2_aggressive"
        ))
    game_result = start_poker(config, verbose=0)

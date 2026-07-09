from collections import Counter

from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker

from src.agents.adaptive_player import AdaptivePlayer
from src.agents.aggressive_player import AggressivePlayer
from src.agents.calling_player import CallingPlayer
from src.agents.fish_player import FishPlayer
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import GameConfig, TrainingConfig


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


def run_training() -> None:
    game_config = GameConfig(
        max_round=100,
        initial_stack=100,
        small_blind_amount=5,
    )

    training_config = TrainingConfig()

    agent = MonteCarloAgent(
        alpha=training_config.alpha,
        epsilon=training_config.epsilon_start,
        epsilon_min=training_config.epsilon_min,
        epsilon_decay=training_config.epsilon_decay,
    )

    agent.train()

    opponent_counter = Counter()

    for episode in range(training_config.episodes):
        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=game_config.small_blind_amount,
        )

        adaptive_player = AdaptivePlayer(
            agent=agent,
            player_name="adaptive_mc",
        )

        config.register_player(
            name="adaptive_mc",
            algorithm=adaptive_player,
        )

        opponent_name, opponent = build_training_opponent(episode)

        opponent_counter[opponent_name] += 1

        config.register_player(
            name=opponent_name,
            algorithm=opponent,
        )

        start_poker(config, verbose=0)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{training_config.episodes}, "
                f"opponent={opponent_name}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"q_states={len(agent.q_table)}"
            )

    print(
        "Training opponents distribution: "
        f"{dict(opponent_counter)}"
    )

    agent.save(training_config.model_path)

    print(f"Saved model to {training_config.model_path}")


if __name__ == "__main__":
    run_training()
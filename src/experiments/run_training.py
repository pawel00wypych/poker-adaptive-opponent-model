from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker

from src.agents.adaptive_player import AdaptivePlayer
from src.agents.aggressive_player import AggressivePlayer
from src.agents.fish_player import FishPlayer
from src.agents.q_learning_agent import QLearningAgent
from src.config import GameConfig, TrainingConfig


def run_training() -> None:
    game_config = GameConfig(max_round=100, initial_stack=100, small_blind_amount=5)
    training_config = TrainingConfig(episodes=200)

    agent = QLearningAgent(
        alpha=training_config.alpha,
        gamma=training_config.gamma,
        epsilon=training_config.epsilon_start,
        epsilon_min=training_config.epsilon_min,
        epsilon_decay=training_config.epsilon_decay,
    )

    agent.train()

    for episode in range(training_config.episodes):
        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=game_config.small_blind_amount,
        )

        config.register_player(
            name="adaptive_rl",
            algorithm=AdaptivePlayer(
                agent=agent,
                player_name="adaptive_rl",
            ),
        )

        if episode % 2 == 0:
            config.register_player(
                name="fish",
                algorithm=FishPlayer(player_name="fish"),
            )
        else:
            config.register_player(
                name="aggressive",
                algorithm=AggressivePlayer(player_name="aggressive"),
            )

        start_poker(config, verbose=0)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{training_config.episodes}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"q_states={len(agent.q_table)}"
            )

    agent.save(training_config.model_path)
    print(f"Saved model to {training_config.model_path}")


if __name__ == "__main__":
    run_training()
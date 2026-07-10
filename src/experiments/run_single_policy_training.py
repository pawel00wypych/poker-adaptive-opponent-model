from collections import Counter

from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.single_policy_player import SinglePolicyPlayer
from src.config import GameConfig, TrainingConfig
from src.experiments.training_opponents import (
    build_training_opponent,
)


def run_single_policy_training() -> None:
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

        config.register_player(
            name="single_policy_mc",
            algorithm=SinglePolicyPlayer(
                agent=agent,
                player_name="single_policy_mc",
            ),
        )

        opponent_name, opponent = build_training_opponent(
            episode
        )
        opponent_counter[opponent_name] += 1

        config.register_player(
            name=opponent_name,
            algorithm=opponent,
        )

        start_poker(config, verbose=0)

        if (episode + 1) % 100 == 0:
            print(
                f"Single-policy episode "
                f"{episode + 1}/{training_config.episodes}, "
                f"opponent={opponent_name}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"states={len(agent.q_table)}"
            )

    print(
        "Single-policy training distribution: "
        f"{dict(opponent_counter)}"
    )

    agent.save(
        training_config.single_policy_model_path
    )

    print(
        "Saved single-policy model to "
        f"{training_config.single_policy_model_path}"
    )


if __name__ == "__main__":
    run_single_policy_training()
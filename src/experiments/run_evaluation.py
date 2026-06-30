from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker

from src.agents.adaptive_player import AdaptivePlayer
from src.agents.aggressive_player import AggressivePlayer
from src.agents.fish_player import FishPlayer
from src.agents.q_learning_agent import QLearningAgent
from src.config import EvaluationConfig, GameConfig
from src.evaluation.result_logger import ResultLogger


def run_evaluation() -> None:
    game_config = GameConfig(max_round=100, initial_stack=100, small_blind_amount=5)
    eval_config = EvaluationConfig(games=30)

    big_blind = game_config.small_blind_amount * 2

    agent = QLearningAgent.load(eval_config.model_path)
    agent.eval()

    logger = ResultLogger(eval_config.output_path)

    for game_id in range(eval_config.games):
        config = setup_config(
            max_round=game_config.max_round,
            initial_stack=game_config.initial_stack,
            small_blind_amount=game_config.small_blind_amount,
        )

        adaptive_player = AdaptivePlayer(
            agent=agent,
            player_name="adaptive_rl",
        )

        config.register_player(
            name="adaptive_rl",
            algorithm=adaptive_player,
        )

        if game_id % 2 == 0:
            opponent_name = "fish"
            config.register_player(
                name=opponent_name,
                algorithm=FishPlayer(player_name=opponent_name),
            )
        else:
            opponent_name = "aggressive"
            config.register_player(
                name=opponent_name,
                algorithm=AggressivePlayer(player_name=opponent_name),
            )

        result = start_poker(config, verbose=0)

        for player in result["players"]:
            if player["name"] == "adaptive_rl":
                logger.log_game(
                    experiment_name=f"adaptive_vs_{opponent_name}",
                    game_id=game_id,
                    agent_name="adaptive_rl",
                    final_stack=player["stack"],
                    initial_stack=game_config.initial_stack,
                    hands_played=max(adaptive_player.hands_played, 1),
                    big_blind=big_blind,
                )

        print(
            f"Finished evaluation game {game_id + 1}/{eval_config.games}, "
            f"hands_played={adaptive_player.hands_played}, "
            f"total_reward_bb={adaptive_player.total_reward_bb:.2f}"
        )

    print(f"Saved results to {eval_config.output_path}")


if __name__ == "__main__":
    run_evaluation()
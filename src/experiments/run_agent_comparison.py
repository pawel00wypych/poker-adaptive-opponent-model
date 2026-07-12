from PyPokerEngine.pypokerengine.api.game import setup_config, start_poker

from src.players.adaptive_player import AdaptivePlayer
from src.players.aggressive_player import AggressivePlayer
from src.players.calling_player import CallingPlayer
from src.players.fish_player import FishPlayer
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.players.rule_based_player import RuleBasedPlayer
from src.players.single_policy_player import SinglePolicyPlayer
from src.config import GameConfig, TrainingConfig, EvaluationConfig
from src.evaluation.result_logger import ResultLogger


def build_opponent(opponent_name: str):
    if opponent_name == "fish":
        return FishPlayer(player_name="fish")

    if opponent_name == "aggressive":
        return AggressivePlayer(player_name="aggressive")

    if opponent_name == "calling":
        return CallingPlayer(player_name="calling")

    raise ValueError(f"Unknown opponent: {opponent_name}")


def build_tested_player(agent_name: str):
    training_config = TrainingConfig()

    if agent_name == "rule_based":
        return RuleBasedPlayer(
            player_name="rule_based"
        )

    if agent_name == "single_policy_mc":
        agent = MonteCarloAgent.load(
            training_config.single_policy_model_path
        )
        agent.eval()

        return SinglePolicyPlayer(
            agent=agent,
            player_name="single_policy_mc",
        )

    if agent_name == "adaptive_mc":
        agent = MonteCarloAgent.load(
            training_config.adaptive_model_path
        )
        agent.eval()

        return AdaptivePlayer(
            agent=agent,
            player_name="adaptive_mc",
        )

    raise ValueError(
        f"Unknown tested agent: {agent_name}"
    )


def get_hands_played(player) -> int:
    hands_played = getattr(player, "hands_played", None)

    if hands_played is None:
        raise AttributeError(
            f"Player {player.__class__.__name__} does not expose hands_played. "
            "Add TrackingPlayerMixin or implement receive_round_result_message."
        )

    return max(hands_played, 1)


def run_single_game(
    game_id: int,
    tested_agent_name: str,
    opponent_name: str,
    game_config: GameConfig,
    logger: ResultLogger,
) -> None:
    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=game_config.small_blind_amount,
    )

    tested_player = build_tested_player(tested_agent_name)
    opponent = build_opponent(opponent_name)

    config.register_player(name=tested_agent_name, algorithm=tested_player)
    config.register_player(name=opponent_name, algorithm=opponent)

    result = start_poker(config, verbose=0)

    big_blind = game_config.small_blind_amount * 2

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            logger.log_game(
                experiment_name=f"{tested_agent_name}_vs_{opponent_name}",
                game_id=game_id,
                agent_name=tested_agent_name,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=get_hands_played(tested_player),
                big_blind=big_blind,
            )


def run_agent_comparison() -> None:
    game_config = GameConfig()
    evaluation_config = EvaluationConfig()
    logger = ResultLogger(evaluation_config.output_path)

    tested_agents = [
        "rule_based",
        "single_policy_mc",
        "adaptive_mc",
    ]

    opponents = [
        "fish",
        "aggressive",
        "calling",
    ]

    games_per_matchup = evaluation_config.games_per_matchup

    game_id = 0

    for tested_agent in tested_agents:
        for opponent in opponents:
            for _ in range(games_per_matchup):
                run_single_game(
                    game_id=game_id,
                    tested_agent_name=tested_agent,
                    opponent_name=opponent,
                    game_config=game_config,
                    logger=logger,
                )

                game_id += 1

            print(f"Finished matchup: {tested_agent} vs {opponent}")

    print(f"Saved comparison results to {evaluation_config.output_path}")

def load_eval_agent(
    model_path: str,
) -> MonteCarloAgent:
    agent = MonteCarloAgent.load(
        model_path
    )
    agent.eval()
    return agent


def load_adaptive_agents(
    training_config: TrainingConfig,
) -> dict[str, MonteCarloAgent]:
    return {
        "unknown": load_eval_agent(
            training_config.single_policy_model_path
        ),
        "fish": load_eval_agent(
            training_config.fish_model_path
        ),
        "aggressive": load_eval_agent(
            training_config.aggressive_model_path
        ),
        "calling": load_eval_agent(
            training_config.calling_model_path
        ),
    }

if __name__ == "__main__":
    run_agent_comparison()
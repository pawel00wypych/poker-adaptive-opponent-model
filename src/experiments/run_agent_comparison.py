from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.players.adaptive_player import AdaptivePlayer
from src.agents.monte_carlo_agent import (
    MonteCarloAgent,
)
from src.players.rule_based_player import (
    RuleBasedPlayer,
)
from src.players.single_policy_player import (
    SinglePolicyPlayer,
)
from src.config import (
    EvaluationConfig,
    GameConfig,
    TrainingConfig,
)
from src.evaluation.result_logger import (
    ResultLogger,
)
from src.experiments.training_opponents import (
    build_opponent,
)


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


def build_tested_player(
    agent_name: str,
    opponent_name: str,
):
    training_config = TrainingConfig()

    if agent_name == "rule_based":
        return RuleBasedPlayer(
            player_name="rule_based"
        )

    if agent_name == "single_policy_mc":
        agent = load_eval_agent(
            training_config.single_policy_model_path
        )

        return SinglePolicyPlayer(
            agent=agent,
            player_name="single_policy_mc",
        )

    if agent_name == "adaptive_mc":
        return AdaptivePlayer(
            agents=load_adaptive_agents(
                training_config
            ),
            player_name="adaptive_mc",
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    raise ValueError(
        f"Unknown tested agent: {agent_name}"
    )


def get_hands_played(player) -> int:
    hands_played = getattr(
        player,
        "hands_played",
        None,
    )

    if hands_played is None:
        raise AttributeError(
            f"Player {player.__class__.__name__} "
            "does not expose hands_played."
        )

    return max(
        hands_played,
        1,
    )


def get_classifier_metrics(
    player,
) -> dict:
    if not isinstance(
        player,
        AdaptivePlayer,
    ):
        return {
            "classified_decisions": 0,
            "correct_classifications": 0,
            "incorrect_classifications": 0,
            "unknown_classifications": 0,
            "classifier_accuracy": 0.0,
            "classifier_coverage": 0.0,
            "policy_switches": 0,
            "first_classification_hand": None,
            "first_correct_classification_hand": None,
            "final_predicted_type": "",
        }

    return {
        "classified_decisions": (
            player.classified_decisions
        ),
        "correct_classifications": (
            player.correct_classifications
        ),
        "incorrect_classifications": (
            player.incorrect_classifications
        ),
        "unknown_classifications": (
            player.unknown_classifications
        ),
        "classifier_accuracy": (
            player.classifier_accuracy
        ),
        "classifier_coverage": (
            player.classifier_coverage
        ),
        "policy_switches": (
            player.policy_switches
        ),
        "first_classification_hand": (
            player.first_classification_hand
        ),
        "first_correct_classification_hand": (
            player.first_correct_classification_hand
        ),
        "final_predicted_type": (
            player.final_predicted_type
        ),
    }


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
        small_blind_amount=(
            game_config.small_blind_amount
        ),
    )

    tested_player = build_tested_player(
        tested_agent_name,
        opponent_name,
    )

    opponent = build_opponent(
        opponent_name
    )

    config.register_player(
        name=tested_agent_name,
        algorithm=tested_player,
    )

    config.register_player(
        name=opponent_name,
        algorithm=opponent,
    )

    result = start_poker(
        config,
        verbose=0,
    )

    hands_played = get_hands_played(
        tested_player
    )

    big_blind = (
        game_config.small_blind_amount * 2
    )

    ended_by_bust = any(
        player_result["stack"] == 0
        for player_result in result["players"]
    )

    ended_by_round_limit = (
        not ended_by_bust
        and hands_played >= game_config.max_round
    )

    classifier_metrics = (
        get_classifier_metrics(
            tested_player
        )
    )

    for player_result in result["players"]:
        if (
            player_result["name"]
            == tested_agent_name
        ):
            logger.log_game(
                experiment_name=(
                    f"{tested_agent_name}"
                    f"_vs_{opponent_name}"
                ),
                game_id=game_id,
                agent_name=tested_agent_name,
                opponent_name=opponent_name,
                final_stack=player_result["stack"],
                initial_stack=(
                    game_config.initial_stack
                ),
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=(
                    ended_by_round_limit
                ),
                **classifier_metrics,
            )

            break


def run_agent_comparison() -> None:
    game_config = GameConfig()
    evaluation_config = EvaluationConfig()

    logger = ResultLogger(
        evaluation_config.output_path
    )

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

    game_id = 0

    for tested_agent in tested_agents:
        for opponent in opponents:
            for _ in range(
                evaluation_config.games_per_matchup
            ):
                run_single_game(
                    game_id=game_id,
                    tested_agent_name=tested_agent,
                    opponent_name=opponent,
                    game_config=game_config,
                    logger=logger,
                )

                game_id += 1

            print(
                "Finished matchup: "
                f"{tested_agent} vs {opponent}"
            )

    print(
        "Saved comparison results to "
        f"{evaluation_config.output_path}"
    )


if __name__ == "__main__":
    run_agent_comparison()
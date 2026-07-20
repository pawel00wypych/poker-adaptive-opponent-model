import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PyPokerEngine.pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import GameConfig
from src.evaluation.constants import (
    CHECKPOINT_PREFIX_BY_POLICY_TYPE,
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    MODEL_DIRECTORY_BY_POLICY_TYPE,
    SUPPORTED_TESTED_AGENTS,
)
from src.experiments.training_opponents import build_opponent
from src.players.adaptive_player import AdaptivePlayer
from src.players.rule_based_player import RuleBasedPlayer
from src.players.single_policy_player import SinglePolicyPlayer
from src.players.fixed_policy_player import FixedPolicyPlayer
from src.players.oracle_adaptive_player import OracleAdaptivePlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
    OPPONENT_TYPE_UNKNOWN,
)


@dataclass(frozen=True)
class ModelBundle:
    """
    Complete set of models for one seed and one checkpoint episode.

    unknown -> general single-policy model
    fish/aggressive/calling -> specialist models
    """

    training_run_directory: Path
    seed: int
    checkpoint_episode: int
    unknown_model_path: Path
    fish_model_path: Path
    aggressive_model_path: Path
    calling_model_path: Path

    @property
    def experiment_id(self) -> str:
        return (
            f"seed_{self.seed}"
            f"_episodes_{self.checkpoint_episode}"
        )

    def agent_paths(self) -> dict[str, Path]:
        return {
            OPPONENT_TYPE_UNKNOWN: self.unknown_model_path,
            OPPONENT_TYPE_FISH: self.fish_model_path,
            OPPONENT_TYPE_AGGRESSIVE: self.aggressive_model_path,
            OPPONENT_TYPE_CALLING: self.calling_model_path,
        }


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path
    include_rule_based: bool = True


def checkpoint_filename(
    policy_type: str,
    checkpoint_episode: int,
    seed: int,
) -> str:
    prefix = CHECKPOINT_PREFIX_BY_POLICY_TYPE[policy_type]

    return (
        f"{prefix}"
        f"_episodes_{checkpoint_episode}"
        f"_seed_{seed}.pkl"
    )


def final_model_path(
    seed_directory: Path,
    policy_type: str,
) -> Path:
    return (
        seed_directory
        / MODEL_DIRECTORY_BY_POLICY_TYPE[policy_type]
        / "final.pkl"
    )


def checkpoint_model_path(
    seed_directory: Path,
    policy_type: str,
    checkpoint_episode: int,
    seed: int,
) -> Path:
    return (
        seed_directory
        / MODEL_DIRECTORY_BY_POLICY_TYPE[policy_type]
        / "checkpoints"
        / checkpoint_filename(
            policy_type=policy_type,
            checkpoint_episode=checkpoint_episode,
            seed=seed,
        )
    )


def parse_seed_from_directory(
    seed_directory: Path,
) -> int:
    name = seed_directory.name

    if not name.startswith("seed_"):
        raise ValueError(
            f"Invalid seed directory name: {name}"
        )

    return int(
        name.removeprefix("seed_")
    )


def discover_seed_directories(
    training_run_directory: str | Path,
) -> list[Path]:
    root = Path(training_run_directory)

    if not root.exists():
        raise FileNotFoundError(
            f"Training run directory does not exist: {root}"
        )

    seed_directories = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("seed_")
    ]

    return sorted(
        seed_directories,
        key=parse_seed_from_directory,
    )


def build_model_bundle(
    training_run_directory: str | Path,
    seed: int,
    checkpoint_episode: int,
    use_final_models: bool = False,
) -> ModelBundle:
    root = Path(training_run_directory)
    seed_directory = root / f"seed_{seed}"

    if use_final_models:
        paths = {
            policy_type: final_model_path(
                seed_directory=seed_directory,
                policy_type=policy_type,
            )
            for policy_type in MODEL_DIRECTORY_BY_POLICY_TYPE
        }
    else:
        paths = {
            policy_type: checkpoint_model_path(
                seed_directory=seed_directory,
                policy_type=policy_type,
                checkpoint_episode=checkpoint_episode,
                seed=seed,
            )
            for policy_type in MODEL_DIRECTORY_BY_POLICY_TYPE
        }

    missing_paths = [
        path
        for path in paths.values()
        if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Incomplete model bundle. Missing paths:\n"
            f"{missing}"
        )

    return ModelBundle(
        training_run_directory=root,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
        unknown_model_path=paths[OPPONENT_TYPE_UNKNOWN],
        fish_model_path=paths[OPPONENT_TYPE_FISH],
        aggressive_model_path=paths[OPPONENT_TYPE_AGGRESSIVE],
        calling_model_path=paths[OPPONENT_TYPE_CALLING],
    )


def discover_model_bundles(
    training_run_directory: str | Path,
    checkpoint_episodes: Iterable[int],
    seeds: Iterable[int] | None = None,
    use_final_models: bool = False,
    skip_incomplete: bool = True,
) -> list[ModelBundle]:
    root = Path(training_run_directory)

    if seeds is None:
        discovered_seeds = [
            parse_seed_from_directory(path)
            for path in discover_seed_directories(root)
        ]
    else:
        discovered_seeds = list(seeds)

    bundles: list[ModelBundle] = []

    for seed in discovered_seeds:
        for checkpoint_episode in checkpoint_episodes:
            try:
                bundle = build_model_bundle(
                    training_run_directory=root,
                    seed=seed,
                    checkpoint_episode=checkpoint_episode,
                    use_final_models=use_final_models,
                )
            except FileNotFoundError:
                if skip_incomplete:
                    continue

                raise

            bundles.append(bundle)

    return bundles


def load_eval_agent(
    model_path: str | Path,
) -> MonteCarloAgent:
    agent = MonteCarloAgent.load(
        str(model_path)
    )
    agent.eval()

    return agent


def load_adaptive_agents(
    bundle: ModelBundle,
) -> dict[str, MonteCarloAgent]:
    return {
        policy_type: load_eval_agent(path)
        for policy_type, path in bundle.agent_paths().items()
    }


def build_tested_player(
    tested_agent_name: str,
    opponent_name: str,
    bundle: ModelBundle,
):
    if tested_agent_name == "rule_based":
        return RuleBasedPlayer(
            player_name="rule_based"
        )

    if tested_agent_name == "single_policy_mc":
        agent = load_eval_agent(
            bundle.unknown_model_path
        )

        return SinglePolicyPlayer(
            agent=agent,
            player_name="single_policy_mc",
        )

    if tested_agent_name == "adaptive_mc":
        return AdaptivePlayer(
            agents=load_adaptive_agents(bundle),
            player_name="adaptive_mc",
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    if tested_agent_name == "oracle_adaptive":
        return OracleAdaptivePlayer(
            agents=load_adaptive_agents(bundle),
            oracle_opponent_type=opponent_name,
            player_name="oracle_adaptive",
            verbose=False,
        )

    if tested_agent_name in CROSS_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = CROSS_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_eval_agent(
            bundle.agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    raise ValueError(
        f"Unsupported tested agent: {tested_agent_name}"
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


def get_classifier_metrics(player) -> dict:
    if isinstance(player, AdaptivePlayer):
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
            "policy_switches": player.policy_switches,
            "first_classification_hand": (
                player.first_classification_hand
            ),
            "first_correct_classification_hand": (
                player.first_correct_classification_hand
            ),
            "first_classification_action_count": getattr(
                player,
                "first_classification_action_count",
                None,
            ),
            "first_correct_classification_action_count": getattr(
                player,
                "first_correct_classification_action_count",
                None,
            ),
            "final_predicted_type": (
                player.final_predicted_type
            ),
        }

    if isinstance(player, OracleAdaptivePlayer):
        return {
            "classified_decisions": 1,
            "correct_classifications": 1,
            "incorrect_classifications": 0,
            "unknown_classifications": 0,
            "classifier_accuracy": 1.0,
            "classifier_coverage": 1.0,
            "policy_switches": 0,
            "first_classification_hand": 1,
            "first_correct_classification_hand": 1,
            "first_classification_action_count": 0,
            "first_correct_classification_action_count": 0,
            "final_predicted_type": (
                player.final_predicted_type
            ),
        }

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
        "first_classification_action_count": None,
        "first_correct_classification_action_count": None,
        "final_predicted_type": "",
    }


def set_evaluation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_game_seed(
    eval_seed_base: int,
    model_seed: int,
    checkpoint_episode: int,
    game_id: int,
) -> int:
    return (
        eval_seed_base
        + model_seed * 1_000_000
        + checkpoint_episode * 1_000
        + game_id
    )


def build_result_row(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    final_stack: int,
    initial_stack: int,
    hands_played: int,
    big_blind: int,
    ended_by_bust: bool,
    ended_by_round_limit: bool,
    classifier_metrics: dict,
) -> dict:
    if big_blind <= 0:
        raise ValueError(
            "big_blind must be greater than zero"
        )

    profit = final_stack - initial_stack
    profit_bb = profit / big_blind

    return {
        "training_run": (
            bundle.training_run_directory.name
        ),
        "model_seed": bundle.seed,
        "checkpoint_episode": (
            bundle.checkpoint_episode
        ),
        "experiment_id": bundle.experiment_id,
        "experiment_name": (
            f"{tested_agent_name}_vs_{opponent_name}"
        ),
        "game_id": game_id,
        "agent_name": tested_agent_name,
        "opponent_name": opponent_name,
        "final_stack": final_stack,
        "initial_stack": initial_stack,
        "profit": profit,
        "profit_bb": profit_bb,
        "hands_played": hands_played,
        "won_game": int(
            final_stack > initial_stack
        ),
        "busted": int(
            final_stack == 0
        ),
        "ended_by_bust": int(
            ended_by_bust
        ),
        "ended_by_round_limit": int(
            ended_by_round_limit
        ),
        **classifier_metrics,
    }


def evaluate_single_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_game_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        checkpoint_episode=bundle.checkpoint_episode,
        game_id=game_id,
    )

    set_evaluation_seed(game_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=(
            game_config.small_blind_amount
        ),
    )

    tested_player = build_tested_player(
        tested_agent_name=tested_agent_name,
        opponent_name=opponent_name,
        bundle=bundle,
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

    ended_by_bust = any(
        player_result["stack"] == 0
        for player_result in result["players"]
    )

    ended_by_round_limit = (
        not ended_by_bust
        and hands_played >= game_config.max_round
    )

    classifier_metrics = get_classifier_metrics(
        tested_player
    )

    big_blind = (
        game_config.small_blind_amount * 2
    )

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            return build_result_row(
                bundle=bundle,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_name,
                game_id=game_id,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=(
                    ended_by_round_limit
                ),
                classifier_metrics=classifier_metrics,
            )

    raise RuntimeError(
        "Tested player result not found in game result."
    )


def evaluate_bundle(
    *,
    bundle: ModelBundle,
    config: CheckpointEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        for opponent_name in config.opponents:
            for _ in range(config.games_per_matchup):
                row = evaluate_single_game(
                    bundle=bundle,
                    tested_agent_name=(
                        tested_agent_name
                    ),
                    opponent_name=opponent_name,
                    game_id=game_id,
                    game_config=game_config,
                    eval_seed_base=config.eval_seed_base,
                )

                rows.append(row)
                game_id += 1

    return rows


CHECKPOINT_EVALUATION_FIELDNAMES = [
    "training_run",
    "model_seed",
    "checkpoint_episode",
    "experiment_id",
    "experiment_name",
    "game_id",
    "agent_name",
    "opponent_name",
    "final_stack",
    "initial_stack",
    "profit",
    "profit_bb",
    "hands_played",
    "won_game",
    "busted",
    "ended_by_bust",
    "ended_by_round_limit",
    "classified_decisions",
    "correct_classifications",
    "incorrect_classifications",
    "unknown_classifications",
    "classifier_accuracy",
    "classifier_coverage",
    "policy_switches",
    "first_classification_hand",
    "first_correct_classification_hand",
    "first_classification_action_count",
    "first_correct_classification_action_count",
    "final_predicted_type",
]


def write_rows(
    output_path: str | Path,
    rows: Iterable[dict],
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CHECKPOINT_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

import csv
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.config import GameConfig
from src.evaluation.constants import MODEL_DIRECTORIES
from src.evaluation.player_factory import (
    EvaluationAgentLoaders,
    build_evaluation_player,
)
from src.evaluation.runners.evaluation_seed import (
    build_paired_evaluation_seed,
)
from src.players.learned.adaptive_player import AdaptivePlayer
from src.players.learned.oracle_player import OraclePlayer
from src.players.opponents.factory import build_opponent
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


@dataclass(frozen=True)
class ModelBundle:
    """
    Complete set of model paths for one seed and one model source.

    Final benchmarks use ``model_source="final"`` and contain exactly one
    bundle per training seed. ``model_source="checkpoint"`` is reserved for
    the separate learning-curve diagnostic.
    """

    training_run_directory: Path
    seed: int
    episode: int
    model_source: Literal["final", "checkpoint"]
    unknown_model_path: Path
    tight_model_path: Path
    aggressive_model_path: Path
    calling_model_path: Path
    q_learning_training_run_directory: Path | None = None
    q_learning_unknown_model_path: Path | None = None
    q_learning_tight_model_path: Path | None = None
    q_learning_aggressive_model_path: Path | None = None
    q_learning_calling_model_path: Path | None = None
    sarsa_training_run_directory: Path | None = None
    sarsa_unknown_model_path: Path | None = None
    sarsa_tight_model_path: Path | None = None
    sarsa_aggressive_model_path: Path | None = None
    sarsa_calling_model_path: Path | None = None
    double_q_learning_training_run_directory: Path | None = None
    double_q_learning_unknown_model_path: Path | None = None
    double_q_learning_tight_model_path: Path | None = None
    double_q_learning_aggressive_model_path: Path | None = None
    double_q_learning_calling_model_path: Path | None = None

    def __post_init__(self) -> None:
        if self.episode <= 0:
            raise ValueError("episode must be greater than zero")
        if self.model_source not in {"final", "checkpoint"}:
            raise ValueError("model_source must be either 'final' or 'checkpoint'")

    @property
    def training_episode(self) -> int | None:
        return self.episode if self.model_source == "final" else None

    @property
    def model_episode(self) -> int:
        return self.episode

    @property
    def checkpoint_episode(self) -> int | None:
        return self.episode if self.model_source == "checkpoint" else None

    @property
    def experiment_id(self) -> str:
        return f"seed_{self.seed}_{self.model_source}_episode_{self.episode}"

    def agent_paths(self) -> dict[str, Path]:
        return {
            OPPONENT_TYPE_UNKNOWN: self.unknown_model_path,
            OPPONENT_TYPE_TIGHT: self.tight_model_path,
            OPPONENT_TYPE_AGGRESSIVE: self.aggressive_model_path,
            OPPONENT_TYPE_CALLING: self.calling_model_path,
        }

    def has_q_learning_models(self) -> bool:
        return all(
            path is not None
            for path in (
                self.q_learning_unknown_model_path,
                self.q_learning_tight_model_path,
                self.q_learning_aggressive_model_path,
                self.q_learning_calling_model_path,
            )
        )

    def has_sarsa_models(self) -> bool:
        return all(
            path is not None
            for path in (
                self.sarsa_unknown_model_path,
                self.sarsa_tight_model_path,
                self.sarsa_aggressive_model_path,
                self.sarsa_calling_model_path,
            )
        )

    def has_double_q_learning_models(self) -> bool:
        return all(
            path is not None
            for path in (
                self.double_q_learning_unknown_model_path,
                self.double_q_learning_tight_model_path,
                self.double_q_learning_aggressive_model_path,
                self.double_q_learning_calling_model_path,
            )
        )

    def double_q_learning_agent_paths(self) -> dict[str, Path]:
        if not self.has_double_q_learning_models():
            raise ValueError(
                "Double Q-learning model paths are not available for this bundle. "
                "Pass --double-q-learning-run-dir to evaluate Double Q-learning agents."
            )

        assert self.double_q_learning_unknown_model_path is not None
        assert self.double_q_learning_tight_model_path is not None
        assert self.double_q_learning_aggressive_model_path is not None
        assert self.double_q_learning_calling_model_path is not None

        return {
            OPPONENT_TYPE_UNKNOWN: self.double_q_learning_unknown_model_path,
            OPPONENT_TYPE_TIGHT: self.double_q_learning_tight_model_path,
            OPPONENT_TYPE_AGGRESSIVE: self.double_q_learning_aggressive_model_path,
            OPPONENT_TYPE_CALLING: self.double_q_learning_calling_model_path,
        }

    def sarsa_agent_paths(self) -> dict[str, Path]:
        if not self.has_sarsa_models():
            raise ValueError(
                "SARSA model paths are not available for this bundle. "
                "Pass --sarsa-run-dir to evaluate SARSA agents."
            )

        assert self.sarsa_unknown_model_path is not None
        assert self.sarsa_tight_model_path is not None
        assert self.sarsa_aggressive_model_path is not None
        assert self.sarsa_calling_model_path is not None

        return {
            OPPONENT_TYPE_UNKNOWN: self.sarsa_unknown_model_path,
            OPPONENT_TYPE_TIGHT: self.sarsa_tight_model_path,
            OPPONENT_TYPE_AGGRESSIVE: self.sarsa_aggressive_model_path,
            OPPONENT_TYPE_CALLING: self.sarsa_calling_model_path,
        }

    def q_learning_agent_paths(self) -> dict[str, Path]:
        if not self.has_q_learning_models():
            raise ValueError(
                "Q-learning model paths are not available for this bundle. "
                "Pass --q-learning-run-dir to evaluate Q-learning agents."
            )

        assert self.q_learning_unknown_model_path is not None
        assert self.q_learning_tight_model_path is not None
        assert self.q_learning_aggressive_model_path is not None
        assert self.q_learning_calling_model_path is not None

        return {
            OPPONENT_TYPE_UNKNOWN: self.q_learning_unknown_model_path,
            OPPONENT_TYPE_TIGHT: self.q_learning_tight_model_path,
            OPPONENT_TYPE_AGGRESSIVE: self.q_learning_aggressive_model_path,
            OPPONENT_TYPE_CALLING: self.q_learning_calling_model_path,
        }


@dataclass(frozen=True)
class TrainingOpponentEvaluationConfig:
    games_per_matchup: int
    opponents: tuple[str, ...]
    tested_agents: tuple[str, ...]
    eval_seed_base: int
    output_path: Path
    include_rule_based: bool = True


def final_model_path(
    seed_directory: Path,
    policy_type: str,
) -> Path:
    return seed_directory / MODEL_DIRECTORIES[policy_type] / "final.pkl"


def parse_seed_from_directory(
    seed_directory: Path,
) -> int:
    name = seed_directory.name

    if not name.startswith("seed_"):
        raise ValueError(f"Invalid seed directory name: {name}")

    return int(name.removeprefix("seed_"))


def discover_seed_directories(
    training_run_directory: str | Path,
) -> list[Path]:
    root = Path(training_run_directory)

    if not root.exists():
        raise FileNotFoundError(f"Training run directory does not exist: {root}")

    seed_directories = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("seed_")
    ]

    return sorted(
        seed_directories,
        key=parse_seed_from_directory,
    )


def build_final_policy_paths(
    *,
    seed_directory: Path,
) -> dict[str, Path]:
    return {
        policy_type: final_model_path(
            seed_directory=seed_directory,
            policy_type=policy_type,
        )
        for policy_type in MODEL_DIRECTORIES
    }


def validate_model_paths(
    paths: Iterable[Path],
    *,
    bundle_name: str,
) -> None:
    missing_paths = [path for path in paths if not path.exists()]

    if not missing_paths:
        return

    missing = "\n".join(str(path) for path in missing_paths)

    raise FileNotFoundError(
        f"Incomplete model bundle ({bundle_name}). Missing paths:\n{missing}"
    )


def load_final_model_metadata(model_path: str | Path) -> dict:
    metadata_path = Path(model_path).with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Final model metadata does not exist: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise TypeError(f"Final model metadata must be a JSON object: {metadata_path}")
    return metadata


def final_training_episode(
    paths: Iterable[Path],
    *,
    expected_seed: int,
    bundle_name: str,
) -> int:
    model_paths = tuple(paths)
    validate_model_paths(model_paths, bundle_name=bundle_name)

    completed_episodes: set[int] = set()
    for path in model_paths:
        metadata = load_final_model_metadata(path)
        completed_episode = metadata.get("completed_episodes")
        metadata_seed = metadata.get("seed")

        if (
            isinstance(completed_episode, bool)
            or not isinstance(completed_episode, int)
            or completed_episode <= 0
        ):
            raise ValueError(
                "Final model metadata must contain a positive integer "
                f"completed_episodes: {path.with_suffix('.json')}"
            )
        if metadata_seed != expected_seed:
            raise ValueError(
                f"Final model metadata seed mismatch for {path}: "
                f"expected {expected_seed}, found {metadata_seed!r}."
            )
        completed_episodes.add(completed_episode)

    if len(completed_episodes) != 1:
        raise ValueError(
            f"Final {bundle_name} models for seed {expected_seed} were "
            "saved after different numbers of training episodes: "
            f"{sorted(completed_episodes)}."
        )

    return completed_episodes.pop()


def build_model_bundle_from_paths(
    *,
    training_run_directory: str | Path,
    seed: int,
    episode: int,
    model_source: Literal["final", "checkpoint"],
    paths: dict[str, Path],
    q_learning_run_directory: str | Path | None = None,
    q_learning_paths: dict[str, Path] | None = None,
    sarsa_run_directory: str | Path | None = None,
    sarsa_paths: dict[str, Path] | None = None,
    double_q_learning_run_directory: str | Path | None = None,
    double_q_learning_paths: dict[str, Path] | None = None,
) -> ModelBundle:
    root = Path(training_run_directory)
    q_learning_root = (
        Path(q_learning_run_directory) if q_learning_run_directory is not None else None
    )
    sarsa_root = Path(sarsa_run_directory) if sarsa_run_directory is not None else None
    double_q_learning_root = (
        Path(double_q_learning_run_directory)
        if double_q_learning_run_directory is not None
        else None
    )

    return ModelBundle(
        training_run_directory=root,
        seed=seed,
        episode=episode,
        model_source=model_source,
        unknown_model_path=paths[OPPONENT_TYPE_UNKNOWN],
        tight_model_path=paths[OPPONENT_TYPE_TIGHT],
        aggressive_model_path=paths[OPPONENT_TYPE_AGGRESSIVE],
        calling_model_path=paths[OPPONENT_TYPE_CALLING],
        q_learning_training_run_directory=q_learning_root,
        q_learning_unknown_model_path=(
            q_learning_paths[OPPONENT_TYPE_UNKNOWN]
            if q_learning_paths is not None
            else None
        ),
        q_learning_tight_model_path=(
            q_learning_paths[OPPONENT_TYPE_TIGHT]
            if q_learning_paths is not None
            else None
        ),
        q_learning_aggressive_model_path=(
            q_learning_paths[OPPONENT_TYPE_AGGRESSIVE]
            if q_learning_paths is not None
            else None
        ),
        q_learning_calling_model_path=(
            q_learning_paths[OPPONENT_TYPE_CALLING]
            if q_learning_paths is not None
            else None
        ),
        sarsa_training_run_directory=sarsa_root,
        sarsa_unknown_model_path=(
            sarsa_paths[OPPONENT_TYPE_UNKNOWN] if sarsa_paths is not None else None
        ),
        sarsa_tight_model_path=(
            sarsa_paths[OPPONENT_TYPE_TIGHT] if sarsa_paths is not None else None
        ),
        sarsa_aggressive_model_path=(
            sarsa_paths[OPPONENT_TYPE_AGGRESSIVE] if sarsa_paths is not None else None
        ),
        sarsa_calling_model_path=(
            sarsa_paths[OPPONENT_TYPE_CALLING] if sarsa_paths is not None else None
        ),
        double_q_learning_training_run_directory=double_q_learning_root,
        double_q_learning_unknown_model_path=(
            double_q_learning_paths[OPPONENT_TYPE_UNKNOWN]
            if double_q_learning_paths is not None
            else None
        ),
        double_q_learning_tight_model_path=(
            double_q_learning_paths[OPPONENT_TYPE_TIGHT]
            if double_q_learning_paths is not None
            else None
        ),
        double_q_learning_aggressive_model_path=(
            double_q_learning_paths[OPPONENT_TYPE_AGGRESSIVE]
            if double_q_learning_paths is not None
            else None
        ),
        double_q_learning_calling_model_path=(
            double_q_learning_paths[OPPONENT_TYPE_CALLING]
            if double_q_learning_paths is not None
            else None
        ),
    )


def build_final_model_bundle(
    training_run_directory: str | Path,
    seed: int,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
) -> ModelBundle:
    root = Path(training_run_directory)
    paths = build_final_policy_paths(seed_directory=root / f"seed_{seed}")
    episodes = {
        final_training_episode(
            paths.values(),
            expected_seed=seed,
            bundle_name="Monte Carlo",
        )
    }

    optional_specs = (
        ("Q-learning", q_learning_run_directory),
        ("SARSA", sarsa_run_directory),
        ("Double Q-learning", double_q_learning_run_directory),
    )
    optional_paths: list[dict[str, Path] | None] = []
    for bundle_name, run_directory in optional_specs:
        if run_directory is None:
            optional_paths.append(None)
            continue

        algorithm_paths = build_final_policy_paths(
            seed_directory=Path(run_directory) / f"seed_{seed}"
        )
        episodes.add(
            final_training_episode(
                algorithm_paths.values(),
                expected_seed=seed,
                bundle_name=bundle_name,
            )
        )
        optional_paths.append(algorithm_paths)

    if len(episodes) != 1:
        raise ValueError(
            "Final algorithm bundles must use the same training budget for "
            f"seed {seed}; found completed_episodes={sorted(episodes)}."
        )

    q_learning_paths, sarsa_paths, double_q_learning_paths = optional_paths
    return build_model_bundle_from_paths(
        training_run_directory=root,
        seed=seed,
        episode=episodes.pop(),
        model_source="final",
        paths=paths,
        q_learning_run_directory=q_learning_run_directory,
        q_learning_paths=q_learning_paths,
        sarsa_run_directory=sarsa_run_directory,
        sarsa_paths=sarsa_paths,
        double_q_learning_run_directory=double_q_learning_run_directory,
        double_q_learning_paths=double_q_learning_paths,
    )


def discover_final_model_bundles(
    training_run_directory: str | Path,
    seeds: Iterable[int] | None = None,
    skip_incomplete: bool = True,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
) -> list[ModelBundle]:
    root = Path(training_run_directory)

    if seeds is None:
        discovered_seeds = [
            parse_seed_from_directory(path) for path in discover_seed_directories(root)
        ]
    else:
        discovered_seeds = list(seeds)

    bundles: list[ModelBundle] = []

    for seed in discovered_seeds:
        try:
            bundle = build_final_model_bundle(
                training_run_directory=root,
                seed=seed,
                q_learning_run_directory=q_learning_run_directory,
                sarsa_run_directory=sarsa_run_directory,
                double_q_learning_run_directory=double_q_learning_run_directory,
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
    agent = MonteCarloAgent.load(str(model_path))
    agent.eval()

    return agent


def load_q_learning_eval_agent(
    model_path: str | Path,
) -> QLearningAgent:
    agent = QLearningAgent.load(str(model_path))
    agent.eval()

    return agent


def load_sarsa_eval_agent(
    model_path: str | Path,
) -> SarsaAgent:
    agent = SarsaAgent.load(str(model_path))
    agent.eval()

    return agent


def load_double_q_learning_eval_agent(
    model_path: str | Path,
) -> DoubleQLearningAgent:
    agent = DoubleQLearningAgent.load(str(model_path))
    agent.eval()

    return agent


def load_adaptive_agents(
    bundle: ModelBundle,
) -> dict[str, MonteCarloAgent]:
    return {
        policy_type: load_eval_agent(path)
        for policy_type, path in bundle.agent_paths().items()
    }


def load_q_learning_adaptive_agents(
    bundle: ModelBundle,
) -> dict[str, QLearningAgent]:
    return {
        policy_type: load_q_learning_eval_agent(path)
        for policy_type, path in bundle.q_learning_agent_paths().items()
    }


def load_sarsa_adaptive_agents(
    bundle: ModelBundle,
) -> dict[str, SarsaAgent]:
    return {
        policy_type: load_sarsa_eval_agent(path)
        for policy_type, path in bundle.sarsa_agent_paths().items()
    }


def load_double_q_learning_adaptive_agents(
    bundle: ModelBundle,
) -> dict[str, DoubleQLearningAgent]:
    return {
        policy_type: load_double_q_learning_eval_agent(path)
        for policy_type, path in bundle.double_q_learning_agent_paths().items()
    }


def build_model_player_loaders() -> EvaluationAgentLoaders:
    return EvaluationAgentLoaders(
        load_monte_carlo_agent=load_eval_agent,
        load_q_learning_agent=load_q_learning_eval_agent,
        load_sarsa_agent=load_sarsa_eval_agent,
        load_double_q_learning_agent=load_double_q_learning_eval_agent,
        load_monte_carlo_agents=load_adaptive_agents,
        load_q_learning_agents=load_q_learning_adaptive_agents,
        load_sarsa_agents=load_sarsa_adaptive_agents,
        load_double_q_learning_agents=load_double_q_learning_adaptive_agents,
    )


def build_tested_player(
    tested_agent_name: str,
    opponent_name: str,
    bundle: ModelBundle,
):
    return build_evaluation_player(
        tested_agent_name=tested_agent_name,
        bundle=bundle,
        loaders=build_model_player_loaders(),
        expected_opponent_type=opponent_name,
        oracle_opponent_type=opponent_name,
        unsupported_context="tested agent",
    )


def get_hands_played(player) -> int:
    hands_played = getattr(
        player,
        "hands_played",
        None,
    )

    if hands_played is None:
        raise AttributeError(
            f"Player {player.__class__.__name__} does not expose hands_played."
        )

    return max(
        hands_played,
        1,
    )


def get_classifier_metrics(player) -> dict:
    if isinstance(player, AdaptivePlayer):
        return {
            "classified_decisions": (player.classified_decisions),
            "correct_classifications": (player.correct_classifications),
            "incorrect_classifications": (player.incorrect_classifications),
            "unknown_classifications": (player.unknown_classifications),
            "classifier_accuracy": (player.classifier_accuracy),
            "classifier_coverage": (player.classifier_coverage),
            "policy_switches": player.policy_switches,
            "first_classification_hand": (player.first_classification_hand),
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
            "final_predicted_type": (player.final_predicted_type),
        }

    if isinstance(player, OraclePlayer):
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
            "final_predicted_type": (player.final_predicted_type),
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
    model_episode: int,
    matchup_game_index: int,
) -> int:
    return build_paired_evaluation_seed(
        eval_seed_base=eval_seed_base,
        model_seed=model_seed,
        model_episode=model_episode,
        matchup_game_index=matchup_game_index,
    )


def build_result_row(
    *,
    bundle: ModelBundle | None,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    matchup_game_index: int,
    evaluation_seed: int,
    evaluation_replicate_id: int | None = None,
    final_stack: int,
    initial_stack: int,
    hands_played: int,
    big_blind: int,
    ended_by_bust: bool,
    ended_by_round_limit: bool,
    classifier_metrics: dict,
) -> dict:
    if big_blind <= 0:
        raise ValueError("big_blind must be greater than zero")

    profit = final_stack - initial_stack
    profit_bb = profit / big_blind

    if bundle is None and evaluation_replicate_id is None:
        raise ValueError(
            "evaluation_replicate_id is required when bundle is not provided"
        )
    if bundle is not None and evaluation_replicate_id is not None:
        raise ValueError(
            "evaluation_replicate_id must be empty for trained model bundles"
        )

    training_run = bundle.training_run_directory.name if bundle is not None else None
    model_seed = bundle.seed if bundle is not None else None
    model_source = bundle.model_source if bundle is not None else None
    training_episode = bundle.training_episode if bundle is not None else None
    checkpoint_episode = bundle.checkpoint_episode if bundle is not None else None
    experiment_id = (
        bundle.experiment_id
        if bundle is not None
        else f"evaluation_replicate_{evaluation_replicate_id}"
    )

    return {
        "training_run": training_run,
        "model_seed": model_seed,
        "model_source": model_source,
        "training_episode": training_episode,
        "checkpoint_episode": checkpoint_episode,
        "evaluation_replicate_id": evaluation_replicate_id,
        "experiment_id": experiment_id,
        "experiment_name": (f"{tested_agent_name}_vs_{opponent_name}"),
        "game_id": game_id,
        "matchup_game_index": matchup_game_index,
        "evaluation_seed": evaluation_seed,
        "agent_name": tested_agent_name,
        "opponent_name": opponent_name,
        "final_stack": final_stack,
        "initial_stack": initial_stack,
        "profit": profit,
        "profit_bb": profit_bb,
        "hands_played": hands_played,
        "won_game": int(final_stack > initial_stack),
        "busted": int(final_stack == 0),
        "ended_by_bust": int(ended_by_bust),
        "ended_by_round_limit": int(ended_by_round_limit),
        **classifier_metrics,
    }


def evaluate_single_game(
    *,
    bundle: ModelBundle,
    tested_agent_name: str,
    opponent_name: str,
    game_id: int,
    matchup_game_index: int,
    game_config: GameConfig,
    eval_seed_base: int,
) -> dict:
    game_seed = build_game_seed(
        eval_seed_base=eval_seed_base,
        model_seed=bundle.seed,
        model_episode=bundle.episode,
        matchup_game_index=matchup_game_index,
    )

    set_evaluation_seed(game_seed)

    config = setup_config(
        max_round=game_config.max_round,
        initial_stack=game_config.initial_stack,
        small_blind_amount=(game_config.small_blind_amount),
    )

    tested_player = build_tested_player(
        tested_agent_name=tested_agent_name,
        opponent_name=opponent_name,
        bundle=bundle,
    )

    opponent = build_opponent(opponent_name)

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

    hands_played = get_hands_played(tested_player)

    ended_by_bust = any(
        player_result["stack"] == 0 for player_result in result["players"]
    )

    ended_by_round_limit = not ended_by_bust and hands_played >= game_config.max_round

    classifier_metrics = get_classifier_metrics(tested_player)

    big_blind = game_config.small_blind_amount * 2

    for player_result in result["players"]:
        if player_result["name"] == tested_agent_name:
            return build_result_row(
                bundle=bundle,
                tested_agent_name=tested_agent_name,
                opponent_name=opponent_name,
                game_id=game_id,
                matchup_game_index=matchup_game_index,
                evaluation_seed=game_seed,
                final_stack=player_result["stack"],
                initial_stack=game_config.initial_stack,
                hands_played=hands_played,
                big_blind=big_blind,
                ended_by_bust=ended_by_bust,
                ended_by_round_limit=(ended_by_round_limit),
                classifier_metrics=classifier_metrics,
            )

    raise RuntimeError("Tested player result not found in game result.")


def evaluate_training_opponent_bundle(
    *,
    bundle: ModelBundle,
    config: TrainingOpponentEvaluationConfig,
) -> list[dict]:
    game_config = GameConfig()
    rows: list[dict] = []

    game_id = 0

    for tested_agent_name in config.tested_agents:
        for opponent_name in config.opponents:
            for matchup_game_index in range(config.games_per_matchup):
                row = evaluate_single_game(
                    bundle=bundle,
                    tested_agent_name=(tested_agent_name),
                    opponent_name=opponent_name,
                    game_id=game_id,
                    matchup_game_index=matchup_game_index,
                    game_config=game_config,
                    eval_seed_base=config.eval_seed_base,
                )

                rows.append(row)
                game_id += 1

    return rows


MODEL_EVALUATION_FIELDNAMES = [
    "training_run",
    "model_seed",
    "model_source",
    "training_episode",
    "checkpoint_episode",
    "evaluation_replicate_id",
    "experiment_id",
    "experiment_name",
    "game_id",
    "matchup_game_index",
    "evaluation_seed",
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
            fieldnames=MODEL_EVALUATION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

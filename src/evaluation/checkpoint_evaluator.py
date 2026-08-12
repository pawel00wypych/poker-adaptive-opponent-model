import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pypokerengine.api.game import (
    setup_config,
    start_poker,
)

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.config import GameConfig
from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    CHECKPOINT_PREFIXES,
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    MODEL_DIRECTORIES,
    ORACLE_MC_AGENT,
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    SARSA_POLICY_AGENT_TO_POLICY_TYPE,
    DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE,
    RULE_BASED_AGENT,
)
from src.experiments.training.training_opponents import build_opponent
from src.players.adaptive_player import AdaptivePlayer
from src.players.always_call_player import AlwaysCallPlayer
from src.players.always_raise_player import AlwaysRaisePlayer
from src.players.rule_based_player import RuleBasedPlayer
from src.players.fixed_policy_player import FixedPolicyPlayer
from src.players.oracle_player import OraclePlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


@dataclass(frozen=True)
class ModelBundle:
    """
    Complete set of model paths for one seed and one checkpoint episode.

    The Monte Carlo paths are always required and preserve the previous
    evaluation behavior. Q-learning, SARSA, and Double Q-learning paths are optional and are
    present only when the evaluator is given separate training run directories.
    """

    training_run_directory: Path
    seed: int
    checkpoint_episode: int
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

    @property
    def experiment_id(self) -> str:
        return (
            f"seed_{self.seed}"
            f"_episodes_{self.checkpoint_episode}"
        )

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
    prefix = CHECKPOINT_PREFIXES[policy_type]

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
        / MODEL_DIRECTORIES[policy_type]
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
        / MODEL_DIRECTORIES[policy_type]
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


def build_policy_paths(
    *,
    seed_directory: Path,
    checkpoint_episode: int,
    seed: int,
    use_final_models: bool,
) -> dict[str, Path]:
    if use_final_models:
        return {
            policy_type: final_model_path(
                seed_directory=seed_directory,
                policy_type=policy_type,
            )
            for policy_type in MODEL_DIRECTORIES
        }

    return {
        policy_type: checkpoint_model_path(
            seed_directory=seed_directory,
            policy_type=policy_type,
            checkpoint_episode=checkpoint_episode,
            seed=seed,
        )
        for policy_type in MODEL_DIRECTORIES
    }


def validate_model_paths(
    paths: Iterable[Path],
    *,
    bundle_name: str,
) -> None:
    missing_paths = [
        path
        for path in paths
        if not path.exists()
    ]

    if not missing_paths:
        return

    missing = "\n".join(
        str(path)
        for path in missing_paths
    )

    raise FileNotFoundError(
        f"Incomplete model bundle ({bundle_name}). Missing paths:\n"
        f"{missing}"
    )


def build_model_bundle(
    training_run_directory: str | Path,
    seed: int,
    checkpoint_episode: int,
    use_final_models: bool = False,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
) -> ModelBundle:
    root = Path(training_run_directory)
    seed_directory = root / f"seed_{seed}"

    paths = build_policy_paths(
        seed_directory=seed_directory,
        checkpoint_episode=checkpoint_episode,
        seed=seed,
        use_final_models=use_final_models,
    )

    validate_model_paths(
        paths.values(),
        bundle_name="Monte Carlo",
    )

    q_learning_root = (
        Path(q_learning_run_directory)
        if q_learning_run_directory is not None
        else None
    )
    q_learning_paths: dict[str, Path] | None = None

    if q_learning_root is not None:
        q_learning_paths = build_policy_paths(
            seed_directory=q_learning_root / f"seed_{seed}",
            checkpoint_episode=checkpoint_episode,
            seed=seed,
            use_final_models=use_final_models,
        )

        validate_model_paths(
            q_learning_paths.values(),
            bundle_name="Q-learning",
        )

    sarsa_root = (
        Path(sarsa_run_directory)
        if sarsa_run_directory is not None
        else None
    )
    sarsa_paths: dict[str, Path] | None = None

    if sarsa_root is not None:
        sarsa_paths = build_policy_paths(
            seed_directory=sarsa_root / f"seed_{seed}",
            checkpoint_episode=checkpoint_episode,
            seed=seed,
            use_final_models=use_final_models,
        )

        validate_model_paths(
            sarsa_paths.values(),
            bundle_name="SARSA",
        )

    double_q_learning_root = (
        Path(double_q_learning_run_directory)
        if double_q_learning_run_directory is not None
        else None
    )
    double_q_learning_paths: dict[str, Path] | None = None

    if double_q_learning_root is not None:
        double_q_learning_paths = build_policy_paths(
            seed_directory=double_q_learning_root / f"seed_{seed}",
            checkpoint_episode=checkpoint_episode,
            seed=seed,
            use_final_models=use_final_models,
        )

        validate_model_paths(
            double_q_learning_paths.values(),
            bundle_name="Double Q-learning",
        )

    return ModelBundle(
        training_run_directory=root,
        seed=seed,
        checkpoint_episode=checkpoint_episode,
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
            sarsa_paths[OPPONENT_TYPE_UNKNOWN]
            if sarsa_paths is not None
            else None
        ),
        sarsa_tight_model_path=(
            sarsa_paths[OPPONENT_TYPE_TIGHT]
            if sarsa_paths is not None
            else None
        ),
        sarsa_aggressive_model_path=(
            sarsa_paths[OPPONENT_TYPE_AGGRESSIVE]
            if sarsa_paths is not None
            else None
        ),
        sarsa_calling_model_path=(
            sarsa_paths[OPPONENT_TYPE_CALLING]
            if sarsa_paths is not None
            else None
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


def discover_model_bundles(
    training_run_directory: str | Path,
    checkpoint_episodes: Iterable[int],
    seeds: Iterable[int] | None = None,
    use_final_models: bool = False,
    skip_incomplete: bool = True,
    q_learning_run_directory: str | Path | None = None,
    sarsa_run_directory: str | Path | None = None,
    double_q_learning_run_directory: str | Path | None = None,
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
    agent = MonteCarloAgent.load(
        str(model_path)
    )
    agent.eval()

    return agent


def load_q_learning_eval_agent(
    model_path: str | Path,
) -> QLearningAgent:
    agent = QLearningAgent.load(
        str(model_path)
    )
    agent.eval()

    return agent


def load_sarsa_eval_agent(
    model_path: str | Path,
) -> SarsaAgent:
    agent = SarsaAgent.load(
        str(model_path)
    )
    agent.eval()

    return agent


def load_double_q_learning_eval_agent(
    model_path: str | Path,
) -> DoubleQLearningAgent:
    agent = DoubleQLearningAgent.load(
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


def build_tested_player(
    tested_agent_name: str,
    opponent_name: str,
    bundle: ModelBundle,
):
    if tested_agent_name == RULE_BASED_AGENT:
        return RuleBasedPlayer(
            player_name=RULE_BASED_AGENT
        )

    if tested_agent_name == ALWAYS_RAISE_AGENT:
        return AlwaysRaisePlayer(
            player_name=ALWAYS_RAISE_AGENT
        )

    if tested_agent_name == ALWAYS_CALL_AGENT:
        return AlwaysCallPlayer(
            player_name=ALWAYS_CALL_AGENT
        )

    if tested_agent_name == ADAPTIVE_MC_AGENT:
        return AdaptivePlayer(
            agents=load_adaptive_agents(bundle),
            player_name=ADAPTIVE_MC_AGENT,
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_Q_LEARNING_AGENT:
        return AdaptivePlayer(
            agents=load_q_learning_adaptive_agents(bundle),
            player_name=ADAPTIVE_Q_LEARNING_AGENT,
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_SARSA_AGENT:
        return AdaptivePlayer(
            agents=load_sarsa_adaptive_agents(bundle),
            player_name=ADAPTIVE_SARSA_AGENT,
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    if tested_agent_name == ADAPTIVE_DOUBLE_Q_LEARNING_AGENT:
        return AdaptivePlayer(
            agents=load_double_q_learning_adaptive_agents(bundle),
            player_name=ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
            expected_opponent_type=opponent_name,
            verbose=False,
        )

    if tested_agent_name == ORACLE_MC_AGENT:
        return OraclePlayer(
            agents=load_adaptive_agents(bundle),
            oracle_opponent_type=opponent_name,
            player_name=ORACLE_MC_AGENT,
            verbose=False,
        )

    if tested_agent_name == ORACLE_Q_LEARNING_AGENT:
        return OraclePlayer(
            agents=load_q_learning_adaptive_agents(bundle),
            oracle_opponent_type=opponent_name,
            player_name=ORACLE_Q_LEARNING_AGENT,
            verbose=False,
        )

    if tested_agent_name == ORACLE_SARSA_AGENT:
        return OraclePlayer(
            agents=load_sarsa_adaptive_agents(bundle),
            oracle_opponent_type=opponent_name,
            player_name=ORACLE_SARSA_AGENT,
            verbose=False,
        )

    if tested_agent_name == ORACLE_DOUBLE_Q_LEARNING_AGENT:
        return OraclePlayer(
            agents=load_double_q_learning_adaptive_agents(bundle),
            oracle_opponent_type=opponent_name,
            player_name=ORACLE_DOUBLE_Q_LEARNING_AGENT,
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

    if tested_agent_name in Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_q_learning_eval_agent(
            bundle.q_learning_agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    if tested_agent_name in SARSA_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = SARSA_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_sarsa_eval_agent(
            bundle.sarsa_agent_paths()[policy_type]
        )

        return FixedPolicyPlayer(
            agent=agent,
            policy_type=policy_type,
            player_name=tested_agent_name,
            verbose=False,
        )

    if tested_agent_name in DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE:
        policy_type = DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE[
            tested_agent_name
        ]

        agent = load_double_q_learning_eval_agent(
            bundle.double_q_learning_agent_paths()[policy_type]
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

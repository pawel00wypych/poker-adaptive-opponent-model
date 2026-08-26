from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from src.config import (
    FINAL_CONFIG,
    VERIFICATION_CONFIG,
    GameConfig,
    TrainingConfig,
)
from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
)
from src.features.state_encoder import STATE_FIELDS, STATE_VERSION
from src.players.base.player_template import REWARD_FORMULA, REWARD_VERSION
from src.players.constants import GENERALIZATION_OPPONENTS
from src.poker.action_mapper import (
    ACTION_NAMES,
    ACTION_VERSION,
    FREE_FOLD_FALLBACK,
    INVALID_RAISE_FALLBACK,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES
from src.training.constants import ALGORITHM_KEYS

FINAL_PRESET = "final"
VERIFICATION_PRESET = "verification"
EXTENDED_PRESET = "extended"
CUSTOM_PRESET = "custom"

TRAINING_OPPONENT_EVALUATION = "training-opponent"
LEARNING_CURVE_EVALUATION = "learning-curve"
HEAD_TO_HEAD_EVALUATION = "head-to-head"
GENERALIZATION_EVALUATION = "generalization"
STRESS_TEST_EVALUATION = "stress-test"
CROSS_PLAY_EVALUATION = "cross-play"

EVALUATION_SEED_NAMESPACES = {
    TRAINING_OPPONENT_EVALUATION: 1,
    LEARNING_CURVE_EVALUATION: 1,
    HEAD_TO_HEAD_EVALUATION: 2,
    GENERALIZATION_EVALUATION: 3,
    STRESS_TEST_EVALUATION: 4,
    CROSS_PLAY_EVALUATION: 5,
}


@dataclass(frozen=True)
class EvaluationConfig:
    games_per_matchup: int = 500
    learning_curve_games_per_matchup: int = 200
    baseline_evaluation_replicates: int = 5
    seed_namespaces: tuple[tuple[str, int], ...] = tuple(
        EVALUATION_SEED_NAMESPACES.items()
    )

    def __post_init__(self) -> None:
        if self.games_per_matchup <= 0:
            raise ValueError("games_per_matchup must be greater than zero")
        if self.learning_curve_games_per_matchup <= 0:
            raise ValueError(
                "learning_curve_games_per_matchup must be greater than zero"
            )
        if self.baseline_evaluation_replicates <= 0:
            raise ValueError(
                "baseline_evaluation_replicates must be greater than zero"
            )

        namespace_values = [value for _, value in self.seed_namespaces]
        if any(value <= 0 for value in namespace_values):
            raise ValueError("Evaluation seed namespaces must be positive")

    def seed_namespace(self, evaluation_type: str) -> int:
        namespaces = dict(self.seed_namespaces)
        try:
            return namespaces[evaluation_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluation type: {evaluation_type!r}"
            ) from error

    def games_for(self, evaluation_type: str) -> int:
        if evaluation_type == LEARNING_CURVE_EVALUATION:
            return self.learning_curve_games_per_matchup
        return self.games_per_matchup


@dataclass(frozen=True)
class RepresentationConfig:
    state_version: str = STATE_VERSION
    state_fields: tuple[str, ...] = STATE_FIELDS
    action_version: str = ACTION_VERSION
    actions: tuple[str, ...] = ACTION_NAMES
    invalid_raise_fallback: str = INVALID_RAISE_FALLBACK
    free_fold_fallback: str = FREE_FOLD_FALLBACK
    reward_version: str = REWARD_VERSION
    reward_formula: str = REWARD_FORMULA


@dataclass(frozen=True)
class OpponentConfig:
    training: tuple[str, ...] = TRAINING_OPPONENT_TYPES
    generalization: tuple[str, ...] = GENERALIZATION_OPPONENTS
    stress_test: tuple[str, ...] = (
        ALWAYS_CALL_AGENT,
        ALWAYS_RAISE_AGENT,
        RULE_BASED_AGENT,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    preset_name: str
    protocol_id: str
    game: GameConfig = field(default_factory=GameConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    representation: RepresentationConfig = field(
        default_factory=RepresentationConfig
    )
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    algorithms: tuple[str, ...] = ALGORITHM_KEYS

    def scientific_snapshot(self) -> dict[str, object]:
        training = asdict(self.training)
        training.pop("model_root_directory", None)
        return {
            "game": {
                **asdict(self.game),
                "big_blind_amount": self.game.big_blind_amount,
            },
            "training": training,
            "evaluation": asdict(self.evaluation),
            "representation": asdict(self.representation),
            "opponents": asdict(self.opponents),
            "algorithms": list(self.algorithms),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "preset_name": self.preset_name,
            "protocol_id": self.protocol_id,
            **self.scientific_snapshot(),
        }

    @property
    def config_hash(self) -> str:
        return _hash_snapshot(self.scientific_snapshot())

    @property
    def training_config_hash(self) -> str:
        scientific = self.scientific_snapshot()
        return _hash_snapshot(
            {
                "game": scientific["game"],
                "training": scientific["training"],
                "representation": scientific["representation"],
                "training_opponents": scientific["opponents"]["training"],
                "algorithms": scientific["algorithms"],
            }
        )


@dataclass(frozen=True)
class ProtocolProvenance:
    protocol_id: str
    preset_name: str
    experiment_config_hash: str
    training_config_hash: str
    experiment_config: dict[str, object]
    source_revision: str
    source_dirty: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FINAL_EXPERIMENT_CONFIG = ExperimentConfig(
    preset_name=FINAL_PRESET,
    protocol_id="thesis-final-v2",
    training=FINAL_CONFIG,
)

VERIFICATION_EXPERIMENT_CONFIG = ExperimentConfig(
    preset_name=VERIFICATION_PRESET,
    protocol_id="thesis-verification-v2",
    training=VERIFICATION_CONFIG,
    evaluation=EvaluationConfig(
        games_per_matchup=200,
        learning_curve_games_per_matchup=200,
        baseline_evaluation_replicates=3,
    ),
)

EXTENDED_EXPERIMENT_CONFIG = ExperimentConfig(
    preset_name=EXTENDED_PRESET,
    protocol_id="thesis-extended-v2",
    training=FINAL_CONFIG,
    evaluation=EvaluationConfig(
        games_per_matchup=1_000,
        learning_curve_games_per_matchup=200,
        baseline_evaluation_replicates=5,
    ),
)

EXPERIMENT_CONFIG_PRESETS = {
    FINAL_PRESET: FINAL_EXPERIMENT_CONFIG,
    VERIFICATION_PRESET: VERIFICATION_EXPERIMENT_CONFIG,
    EXTENDED_PRESET: EXTENDED_EXPERIMENT_CONFIG,
}

DEFAULT_EXPERIMENT_CONFIG_PRESET = FINAL_PRESET


def experiment_config_for(preset: str) -> ExperimentConfig:
    try:
        return EXPERIMENT_CONFIG_PRESETS[preset]
    except KeyError as error:
        raise ValueError(
            f"Unsupported experiment config preset: {preset!r}. "
            f"Choose one of {sorted(EXPERIMENT_CONFIG_PRESETS)}."
        ) from error


def experiment_config_from_snapshot(
    snapshot: dict[str, object],
) -> ExperimentConfig:
    game_values = dict(snapshot["game"])
    game_values.pop("big_blind_amount", None)
    training_values = dict(snapshot["training"])
    training_values["seeds"] = tuple(training_values["seeds"])
    training_values["checkpoint_episodes"] = tuple(
        training_values["checkpoint_episodes"]
    )
    evaluation_values = dict(snapshot["evaluation"])
    evaluation_values["seed_namespaces"] = tuple(
        tuple(item) for item in evaluation_values["seed_namespaces"]
    )
    representation_values = dict(snapshot["representation"])
    representation_values["state_fields"] = tuple(
        representation_values["state_fields"]
    )
    representation_values["actions"] = tuple(
        representation_values["actions"]
    )
    opponent_values = dict(snapshot["opponents"])
    for key in ("training", "generalization", "stress_test"):
        opponent_values[key] = tuple(opponent_values[key])

    return ExperimentConfig(
        preset_name=str(snapshot["preset_name"]),
        protocol_id=str(snapshot["protocol_id"]),
        game=GameConfig(**game_values),
        training=TrainingConfig(**training_values),
        evaluation=EvaluationConfig(**evaluation_values),
        representation=RepresentationConfig(**representation_values),
        opponents=OpponentConfig(**opponent_values),
        algorithms=tuple(snapshot["algorithms"]),
    )


def resolve_effective_config(
    preset: str,
    *,
    game: GameConfig | None = None,
    training: TrainingConfig | None = None,
    evaluation: EvaluationConfig | None = None,
    representation: RepresentationConfig | None = None,
    opponents: OpponentConfig | None = None,
    algorithms: tuple[str, ...] | None = None,
) -> ExperimentConfig:
    base = experiment_config_for(preset)
    candidate = replace(
        base,
        game=game or base.game,
        training=training or base.training,
        evaluation=evaluation or base.evaluation,
        representation=representation or base.representation,
        opponents=opponents or base.opponents,
        algorithms=algorithms or base.algorithms,
    )
    if candidate.scientific_snapshot() == base.scientific_snapshot():
        return base
    return replace(
        candidate,
        preset_name=CUSTOM_PRESET,
        protocol_id=f"custom-from-{base.protocol_id}",
    )


def resolve_training_run_config(
    preset: str,
    *,
    game: GameConfig,
    episodes: int,
    seed: int,
    alpha: float,
    alpha_mode: str,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_schedule: str,
    checkpoint_episodes: tuple[int, ...],
) -> ExperimentConfig:
    base = experiment_config_for(preset)
    seed_set = (
        base.training.seeds
        if seed in base.training.seeds
        else (seed,)
    )
    training = replace(
        base.training,
        episodes=episodes,
        seeds=seed_set,
        alpha=alpha,
        alpha_mode=alpha_mode,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_schedule=epsilon_schedule,
        checkpoint_episodes=checkpoint_episodes,
    )
    return resolve_effective_config(
        preset,
        game=game,
        training=training,
    )


def build_protocol_provenance(
    config: ExperimentConfig,
    *,
    source_revision: str | None = None,
    source_dirty: bool | None = None,
) -> ProtocolProvenance:
    revision, dirty = (
        (source_revision, source_dirty)
        if source_revision is not None
        else resolve_source_revision()
    )
    return ProtocolProvenance(
        protocol_id=config.protocol_id,
        preset_name=config.preset_name,
        experiment_config_hash=config.config_hash,
        training_config_hash=config.training_config_hash,
        experiment_config=_json_compatible(config.snapshot()),
        source_revision=revision or "unknown",
        source_dirty=dirty,
    )


def validate_protocol_provenance(
    provenance: ProtocolProvenance,
    config: ExperimentConfig,
    *,
    verify_source: bool = True,
) -> None:
    expected_snapshot = _json_compatible(config.snapshot())
    mismatches: list[str] = []
    if provenance.experiment_config_hash != config.config_hash:
        mismatches.append("experiment_config_hash")
    if provenance.training_config_hash != config.training_config_hash:
        mismatches.append("training_config_hash")
    if provenance.experiment_config != expected_snapshot:
        mismatches.append("experiment_config")
    if verify_source:
        revision, dirty = resolve_source_revision(ignore_environment=True)
        if revision != "unknown" and provenance.source_revision != revision:
            mismatches.append("source_revision")
        if dirty is not None and provenance.source_dirty != dirty:
            mismatches.append("source_dirty")
    if mismatches:
        raise ValueError(
            "Injected protocol provenance does not match the effective run: "
            f"{sorted(mismatches)}."
        )


def protocol_metadata(
    provenance: ProtocolProvenance,
) -> dict[str, object]:
    return provenance.to_dict()


def experiment_config_hash_from_snapshot(
    snapshot: dict[str, object],
) -> str:
    scientific = {
        key: snapshot[key]
        for key in (
            "game",
            "training",
            "evaluation",
            "representation",
            "opponents",
            "algorithms",
        )
    }
    return _hash_snapshot(scientific)


def training_config_hash_from_snapshot(
    snapshot: dict[str, object],
) -> str:
    opponents = snapshot["opponents"]
    if not isinstance(opponents, dict):
        raise TypeError("experiment_config.opponents must be an object")
    return _hash_snapshot(
        {
            "game": snapshot["game"],
            "training": snapshot["training"],
            "representation": snapshot["representation"],
            "training_opponents": opponents["training"],
            "algorithms": snapshot["algorithms"],
        }
    )


def protocol_provenance_from_environment() -> ProtocolProvenance | None:
    raw = os.environ.get("EXPERIMENT_PROTOCOL_PROVENANCE")
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("EXPERIMENT_PROTOCOL_PROVENANCE must contain a JSON object")
    return ProtocolProvenance(
        protocol_id=str(payload["protocol_id"]),
        preset_name=str(payload["preset_name"]),
        experiment_config_hash=str(payload["experiment_config_hash"]),
        training_config_hash=str(payload["training_config_hash"]),
        experiment_config=dict(payload["experiment_config"]),
        source_revision=str(payload["source_revision"]),
        source_dirty=payload.get("source_dirty"),
    )


def resolve_source_revision(
    repository_root: str | Path | None = None,
    *,
    ignore_environment: bool = False,
) -> tuple[str, bool | None]:
    environment_revision = (
        None if ignore_environment else os.environ.get("GIT_COMMIT")
    )
    if environment_revision:
        raw_dirty = os.environ.get("EXPERIMENT_SOURCE_DIRTY")
        dirty = (
            raw_dirty.lower() == "true"
            if raw_dirty is not None
            else None
        )
        return environment_revision, dirty

    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return revision, bool(status.strip())


def _hash_snapshot(snapshot: dict[str, object]) -> str:
    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_compatible(value: object) -> dict[str, object]:
    normalized = json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
        )
    )
    if not isinstance(normalized, dict):
        raise TypeError("Experiment configuration snapshot must be an object")
    return normalized

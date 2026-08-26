from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from src.evaluation.algorithm_metadata import (
    ADAPTIVE_AGENTS,
    GENERAL_POLICY_AGENTS,
    ORACLE_ALGORITHM_AGENTS,
)
from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    CHECKPOINT_PREFIXES,
    MODEL_DIRECTORIES,
    RULE_BASED_AGENT,
)
from src.evaluation.runners.model_evaluator import discover_final_model_bundles
from src.experiment_protocol import (
    EXTENDED_PRESET,
    FINAL_PRESET,
    VERIFICATION_PRESET,
    ExperimentConfig,
    experiment_config_for,
)
from src.training.constants import ALGORITHM_KEYS

PIPELINE_SCHEMA_VERSION = 1
PIPELINE_CONFIGS = (
    VERIFICATION_PRESET,
    FINAL_PRESET,
    EXTENDED_PRESET,
)
POLICY_MODELS = ("general_policy", "tight", "aggressive", "calling")
TRAINING_OPPONENTS = ("tight", "aggressive", "calling")
GENERALIZATION_OPPONENTS = (
    "tight_extreme",
    "aggressive_extreme",
    "calling_extreme",
)
STRESS_OPPONENTS = (ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT, RULE_BASED_AGENT)
BASELINE_AGENTS = (ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT, RULE_BASED_AGENT)


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    models: Path
    evaluations: Path
    reports: Path
    log: Path
    manifest: Path
    summary: Path

    @classmethod
    def from_root(cls, root: str | Path) -> PipelinePaths:
        resolved = Path(root).resolve()
        return cls(
            root=resolved,
            models=resolved / "models",
            evaluations=resolved / "evaluations",
            reports=resolved / "reports",
            log=resolved / "pipeline.log",
            manifest=resolved / "pipeline_manifest.json",
            summary=resolved / "pipeline_summary.json",
        )


@dataclass(frozen=True)
class ModelPaths:
    monte_carlo: Path
    q_learning: Path
    sarsa: Path
    double_q_learning: Path


@dataclass(frozen=True)
class Stage:
    stage_id: str
    title: str
    dependencies: tuple[str, ...]
    outputs: tuple[Path, ...]
    kind: str
    work_units: float
    command: tuple[str, ...] = ()
    action: str | None = None
    progress_glob: str | None = None

    def fingerprint(self, config_hash: str) -> str:
        payload = {
            "stage_id": self.stage_id,
            "dependencies": self.dependencies,
            "outputs": [str(path) for path in self.outputs],
            "command": self.command,
            "action": self.action,
            "config_hash": config_hash,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class PipelineContext:
    config_name: str
    experiment_config: ExperimentConfig
    paths: PipelinePaths
    model_paths: ModelPaths
    final_pipeline_dir: Path | None
    workers: int
    heartbeat_seconds: int
    resume: bool
    dry_run: bool
    repository_root: Path


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._lock = threading.Lock()

    def __enter__(self) -> TeeLogger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()

    def emit(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message.rstrip()}"
        with self._lock:
            print(line, flush=True)
            if self._file is not None:
                self._file.write(line + "\n")


class ProgressEstimator:
    DEFAULT_SECONDS_PER_UNIT = {
        "training": 0.083,
        "evaluation": 0.03,
        "report": 2.0,
        "internal": 0.2,
    }

    def __init__(self, stages: Sequence[Stage], stage_records: dict[str, dict]):
        self.stages = stages
        self.stage_records = stage_records

    def estimate_remaining(self, remaining_ids: set[str]) -> float:
        rates: dict[str, list[float]] = {}
        for stage in self.stages:
            record = self.stage_records.get(stage.stage_id, {})
            duration = record.get("duration_seconds")
            if (
                record.get("status") == "success"
                and isinstance(duration, (int, float))
                and duration >= 0
                and stage.work_units > 0
            ):
                rates.setdefault(stage.kind, []).append(
                    float(duration) / stage.work_units
                )

        total = 0.0
        for stage in self.stages:
            if stage.stage_id not in remaining_ids:
                continue
            samples = rates.get(stage.kind, [])
            seconds_per_unit = (
                sum(samples) / len(samples)
                if samples
                else self.DEFAULT_SECONDS_PER_UNIT[stage.kind]
            )
            total += seconds_per_unit * max(stage.work_units, 1.0)
        return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete thesis experiment pipeline with resumable state, "
            "console/file logging, validations, and reports."
        )
    )
    parser.add_argument(
        "--config",
        choices=PIPELINE_CONFIGS,
        default=VERIFICATION_PRESET,
        help=(
            "verification runs the short rehearsal; final trains and evaluates "
            "the thesis models; extended reuses a completed final pipeline and "
            "reruns final-model evaluations with 1000 games per matchup."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Pipeline root. Default: results/pipelines/<config>.",
    )
    parser.add_argument(
        "--final-pipeline-dir",
        type=str,
        default=None,
        help=(
            "Completed final pipeline used by --config extended. "
            "Default: results/pipelines/final."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum workers passed to parallel-capable training/evaluation CLIs.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Emit a pipeline heartbeat when a child stage is otherwise quiet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful stages whose fingerprints and outputs remain valid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stage plan and commands without running anything.",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be greater than zero")
    return args


def build_context(args: argparse.Namespace) -> PipelineContext:
    repository_root = Path(__file__).resolve().parents[2]
    root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else repository_root / "results" / "pipelines" / args.config
    )
    paths = PipelinePaths.from_root(root)
    final_pipeline_dir = None
    if args.config == EXTENDED_PRESET:
        final_pipeline_dir = (
            Path(args.final_pipeline_dir).resolve()
            if args.final_pipeline_dir is not None
            else (
                repository_root / "results" / "pipelines" / FINAL_PRESET
            ).resolve()
        )
        model_root = final_pipeline_dir / "models"
    else:
        model_root = paths.models
    model_paths = ModelPaths(
        monte_carlo=model_root / "monte_carlo",
        q_learning=model_root / "q_learning",
        sarsa=model_root / "sarsa",
        double_q_learning=model_root / "double_q_learning",
    )
    return PipelineContext(
        config_name=args.config,
        experiment_config=experiment_config_for(args.config),
        paths=paths,
        model_paths=model_paths,
        final_pipeline_dir=final_pipeline_dir,
        workers=args.workers,
        heartbeat_seconds=args.heartbeat_seconds,
        resume=args.resume,
        dry_run=args.dry_run,
        repository_root=repository_root,
    )


def _python_module(module: str, *arguments: object) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        "-m",
        module,
        *(str(argument) for argument in arguments),
    )


def _summary_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".summary.json")


def _model_artifacts(root: Path, config: ExperimentConfig) -> tuple[Path, ...]:
    outputs: list[Path] = [
        root / "experiment_config.json",
    ]
    for seed in config.training.seeds:
        seed_dir = root / f"seed_{seed}"
        for policy_type, directory in MODEL_DIRECTORIES.items():
            policy_dir = seed_dir / directory
            outputs.extend(
                [
                    policy_dir / "final.pkl",
                    policy_dir / "final.json",
                ]
            )
            prefix = CHECKPOINT_PREFIXES[policy_type]
            for episode in config.training.checkpoint_episodes:
                checkpoint = (
                    policy_dir
                    / "checkpoints"
                    / f"{prefix}_episodes_{episode}_seed_{seed}.pkl"
                )
                outputs.extend([checkpoint, checkpoint.with_suffix(".json")])
    return tuple(outputs)


def _training_command(
    module: str,
    config: ExperimentConfig,
    output_dir: Path,
    log_interval: int,
) -> tuple[str, ...]:
    return _python_module(
        module,
        "--config",
        config.preset_name,
        "--episodes",
        config.training.episodes,
        "--seeds",
        *config.training.seeds,
        "--models",
        *POLICY_MODELS,
        "--epsilon-schedule",
        config.training.epsilon_schedule,
        "--alpha",
        config.training.alpha,
        "--alpha-mode",
        config.training.alpha_mode,
        "--gamma",
        config.training.gamma,
        "--output-dir",
        output_dir,
        "--checkpoint-episodes",
        *config.training.checkpoint_episodes,
        "--checkpoints",
        "--progress",
        "--no-player-verbose",
        "--no-engine-verbose",
        "--log-interval",
        log_interval,
    )


def _common_model_evaluation_arguments(
    context: PipelineContext,
    output_path: Path,
) -> tuple[object, ...]:
    config = context.experiment_config
    return (
        "--config",
        context.config_name,
        "--training-run-dir",
        context.model_paths.monte_carlo,
        "--q-learning-run-dir",
        context.model_paths.q_learning,
        "--sarsa-run-dir",
        context.model_paths.sarsa,
        "--double-q-learning-run-dir",
        context.model_paths.double_q_learning,
        "--seeds",
        *config.training.seeds,
        "--games",
        config.evaluation.games_per_matchup,
        "--workers",
        context.workers,
        "--output-path",
        output_path,
        "--fail-on-incomplete",
    )


def _evaluation_outputs(path: Path) -> tuple[Path, ...]:
    return (path, _summary_path(path))


def _validation_command(
    context: PipelineContext,
    *,
    input_path: Path,
    output_dir: Path,
    mode: str,
    baseline_only: bool = False,
) -> tuple[str, ...]:
    config = context.experiment_config
    arguments: list[object] = [
        "--input-path",
        input_path,
        "--output-dir",
        output_dir,
        "--format",
        "both",
        "--validation-mode",
        mode,
        "--expected-games-per-matchup",
        config.evaluation.games_per_matchup,
        "--require-manifest",
    ]
    if baseline_only:
        arguments.extend(
            [
                "--expected-evaluation-replicates",
                config.evaluation.baseline_evaluation_replicates,
            ]
        )
    else:
        arguments.extend(
            [
                "--algorithms",
                *ALGORITHM_KEYS,
                "--require-all-algorithms",
                "--min-seeds-per-matchup",
                len(config.training.seeds),
                "--expected-model-seeds",
                *config.training.seeds,
            ]
        )
    if context.config_name == FINAL_PRESET:
        arguments.append("--enforce-frozen-final-protocol")
    return _python_module(
        "src.experiments.validation.validate_evaluation_results",
        *arguments,
    )


def build_stages(context: PipelineContext) -> tuple[Stage, ...]:
    config = context.experiment_config
    paths = context.paths
    log_interval = 100 if context.config_name == VERIFICATION_PRESET else 1_000
    training_units = (
        config.training.episodes
        * len(config.training.seeds)
        * len(POLICY_MODELS)
    )
    stages: list[Stage] = []

    if context.config_name == EXTENDED_PRESET:
        reuse_output = paths.root / "reused_final_models.json"
        stages.append(
            Stage(
                "verify_final_models",
                "Verify and reuse final trained models",
                (),
                (reuse_output,),
                "internal",
                1,
                action="verify_final_models",
            )
        )
        model_dependencies = ("verify_final_models",)
    else:
        mc_outputs = (
            context.model_paths.monte_carlo / "summary.json",
            *_model_artifacts(context.model_paths.monte_carlo, config),
        )
        mc_command = _python_module(
            "src.experiments.training.run_monte_carlo_suite",
            "--config",
            context.config_name,
            "--episodes",
            config.training.episodes,
            "--seeds",
            *config.training.seeds,
            "--models",
            *POLICY_MODELS,
            "--checkpoint-episodes",
            *config.training.checkpoint_episodes,
            "--epsilon-schedule",
            config.training.epsilon_schedule,
            "--alpha-mode",
            config.training.alpha_mode,
            "--output-root",
            paths.models,
            "--experiment-name",
            "monte_carlo",
            "--workers",
            context.workers,
            "--log-interval",
            log_interval,
            "--rerun-existing",
        )
        stages.append(
            Stage(
                "train_monte_carlo",
                "Train Monte Carlo policies",
                (),
                tuple(mc_outputs),
                "training",
                training_units,
                command=mc_command,
                progress_glob="models/monte_carlo/seed_*/**/training.log",
            )
        )
        previous = "train_monte_carlo"
        for stage_id, title, module, output_dir in (
            (
                "train_q_learning",
                "Train Q-learning policies",
                "src.experiments.training.run_q_learning_training",
                context.model_paths.q_learning,
            ),
            (
                "train_sarsa",
                "Train SARSA policies",
                "src.experiments.training.run_sarsa_training",
                context.model_paths.sarsa,
            ),
            (
                "train_double_q_learning",
                "Train Double Q-learning policies",
                "src.experiments.training.run_double_q_learning_training",
                context.model_paths.double_q_learning,
            ),
        ):
            stages.append(
                Stage(
                    stage_id,
                    title,
                    (previous,),
                    (
                        output_dir / "training_summary.json",
                        *_model_artifacts(output_dir, config),
                    ),
                    "training",
                    training_units,
                    command=_training_command(
                        module,
                        config,
                        output_dir,
                        log_interval,
                    ),
                )
            )
            previous = stage_id
        model_dependencies = ("train_double_q_learning",)

        learning_csv = paths.evaluations / "learning_curve_evaluation.csv"
        learning_agents = (
            *GENERAL_POLICY_AGENTS,
            *ADAPTIVE_AGENTS,
            RULE_BASED_AGENT,
        )
        learning_games = config.evaluation.learning_curve_games_per_matchup
        learning_command = _python_module(
            "src.experiments.evaluation.run_learning_curve_evaluation",
            "--config",
            context.config_name,
            "--training-run-dir",
            context.model_paths.monte_carlo,
            "--q-learning-run-dir",
            context.model_paths.q_learning,
            "--sarsa-run-dir",
            context.model_paths.sarsa,
            "--double-q-learning-run-dir",
            context.model_paths.double_q_learning,
            "--checkpoint-episodes",
            *config.training.checkpoint_episodes,
            "--seeds",
            *config.training.seeds,
            "--games",
            learning_games,
            "--agents",
            *learning_agents,
            "--opponents",
            *TRAINING_OPPONENTS,
            "--workers",
            context.workers,
            "--output-path",
            learning_csv,
            "--fail-on-incomplete",
        )
        stages.append(
            Stage(
                "evaluate_learning_curve",
                "Evaluate learning-curve checkpoints",
                model_dependencies,
                _evaluation_outputs(learning_csv),
                "evaluation",
                (
                    learning_games
                    * len(config.training.seeds)
                    * len(config.training.checkpoint_episodes)
                    * len(learning_agents)
                    * len(TRAINING_OPPONENTS)
                ),
                command=learning_command,
            )
        )

    training_csv = paths.evaluations / "training_opponent.csv"
    training_agents = (
        *ADAPTIVE_AGENTS,
        *ORACLE_ALGORITHM_AGENTS,
        *GENERAL_POLICY_AGENTS,
        *BASELINE_AGENTS,
    )
    stages.append(
        Stage(
            "evaluate_training_opponents",
            "Evaluate final models against training opponents",
            model_dependencies,
            _evaluation_outputs(training_csv),
            "evaluation",
            (
                config.evaluation.games_per_matchup
                * len(config.training.seeds)
                * len(training_agents)
                * len(TRAINING_OPPONENTS)
            ),
            command=_python_module(
                "src.experiments.evaluation.run_training_opponent_evaluation",
                *_common_model_evaluation_arguments(context, training_csv),
                "--agents",
                *training_agents,
                "--opponents",
                *TRAINING_OPPONENTS,
            ),
        )
    )

    generalization_csv = paths.evaluations / "generalization.csv"
    stages.append(
        Stage(
            "evaluate_generalization",
            "Evaluate generalization variants",
            ("evaluate_training_opponents",),
            _evaluation_outputs(generalization_csv),
            "evaluation",
            (
                config.evaluation.games_per_matchup
                * len(config.training.seeds)
                * len(training_agents)
                * len(GENERALIZATION_OPPONENTS)
            ),
            command=_python_module(
                "src.experiments.evaluation.run_generalization_evaluation",
                *_common_model_evaluation_arguments(context, generalization_csv),
                "--agents",
                *training_agents,
                "--opponents",
                *GENERALIZATION_OPPONENTS,
            ),
        )
    )

    stress_csv = paths.evaluations / "stress_test.csv"
    stress_agents = (*ADAPTIVE_AGENTS, *GENERAL_POLICY_AGENTS)
    stages.append(
        Stage(
            "evaluate_stress_test",
            "Evaluate stress-test opponents",
            ("evaluate_generalization",),
            _evaluation_outputs(stress_csv),
            "evaluation",
            (
                config.evaluation.games_per_matchup
                * len(config.training.seeds)
                * len(stress_agents)
                * len(STRESS_OPPONENTS)
            ),
            command=_python_module(
                "src.experiments.evaluation.run_stress_test_evaluation",
                *_common_model_evaluation_arguments(context, stress_csv),
                "--agents",
                *stress_agents,
                "--opponents",
                *STRESS_OPPONENTS,
            ),
        )
    )

    cross_dir = paths.evaluations / "cross_play_fragments"
    cross_fragments: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("adaptive", tuple(ADAPTIVE_AGENTS), tuple(ADAPTIVE_AGENTS)),
        ("general", tuple(GENERAL_POLICY_AGENTS), tuple(GENERAL_POLICY_AGENTS)),
    ]
    for algorithm, adaptive, general in zip(
        ALGORITHM_KEYS,
        ADAPTIVE_AGENTS,
        GENERAL_POLICY_AGENTS,
        strict=True,
    ):
        pair = (adaptive, general)
        cross_fragments.append((f"paired_{algorithm}", pair, pair))

    cross_stage_ids: list[str] = []
    cross_csvs: list[Path] = []
    previous_cross_dependency = "evaluate_stress_test"
    for fragment_name, agents, opponents in cross_fragments:
        stage_id = f"evaluate_cross_play_{fragment_name}"
        output = cross_dir / f"{fragment_name}.csv"
        cross_stage_ids.append(stage_id)
        cross_csvs.append(output)
        stages.append(
            Stage(
                stage_id,
                f"Evaluate cross-play fragment: {fragment_name}",
                (previous_cross_dependency,),
                _evaluation_outputs(output),
                "evaluation",
                (
                    config.evaluation.games_per_matchup
                    * len(config.training.seeds)
                    * (len(agents) * len(opponents) - len(set(agents) & set(opponents)))
                ),
                command=_python_module(
                    "src.experiments.evaluation.run_cross_play_evaluation",
                    *_common_model_evaluation_arguments(context, output),
                    "--agents",
                    *agents,
                    "--opponent-agents",
                    *opponents,
                    "--no-include-self-play",
                ),
            )
        )
        previous_cross_dependency = stage_id

    cross_csv = paths.evaluations / "cross_play.csv"
    stages.append(
        Stage(
            "merge_cross_play",
            "Merge complete cross-play matrix",
            tuple(cross_stage_ids),
            _evaluation_outputs(cross_csv),
            "internal",
            1,
            action="merge_cross_play",
        )
    )

    baseline_csv = paths.evaluations / "baseline_head_to_head.csv"
    stages.append(
        Stage(
            "evaluate_baseline_sanity",
            "Evaluate baseline head-to-head matrix",
            ("merge_cross_play",),
            _evaluation_outputs(baseline_csv),
            "evaluation",
            (
                config.evaluation.games_per_matchup
                * config.evaluation.baseline_evaluation_replicates
                * len(BASELINE_AGENTS)
                * len(BASELINE_AGENTS)
            ),
            command=_python_module(
                "src.experiments.evaluation.run_head_to_head_evaluation",
                "--config",
                context.config_name,
                "--games",
                config.evaluation.games_per_matchup,
                "--evaluation-replicates",
                config.evaluation.baseline_evaluation_replicates,
                "--agents",
                *BASELINE_AGENTS,
                "--opponents",
                *BASELINE_AGENTS,
                "--workers",
                context.workers,
                "--output-path",
                baseline_csv,
            ),
        )
    )

    primary_csv = paths.evaluations / "primary_evaluation.csv"
    stages.append(
        Stage(
            "merge_primary_evaluation",
            "Merge training and generalization evaluations",
            ("evaluate_training_opponents", "evaluate_generalization"),
            _evaluation_outputs(primary_csv),
            "internal",
            1,
            action="merge_primary",
        )
    )

    validation_specs = (
        ("training", training_csv, "training-opponent", False),
        ("generalization", generalization_csv, "generalization", False),
        ("stress", stress_csv, "stress-test", False),
        ("cross_play", cross_csv, "cross-play", False),
        ("baseline", baseline_csv, "baseline-sanity", True),
    )
    validation_stage_ids: list[str] = []
    validation_dependency = "evaluate_baseline_sanity"
    for name, input_path, mode, baseline_only in validation_specs:
        stage_id = f"validate_{name}"
        output_dir = paths.reports / "validation" / name
        validation_stage_ids.append(stage_id)
        stages.append(
            Stage(
                stage_id,
                f"Validate {name} evaluation",
                (validation_dependency,),
                (
                    output_dir / "experiment_validation.md",
                    output_dir / "experiment_validation.json",
                ),
                "report",
                1,
                command=_validation_command(
                    context,
                    input_path=input_path,
                    output_dir=output_dir,
                    mode=mode,
                    baseline_only=baseline_only,
                ),
            )
        )
        validation_dependency = stage_id

    report_stage_ids: list[str] = []
    if context.config_name != EXTENDED_PRESET:
        learning_report_dir = paths.reports / "learning_curve"
        stages.append(
            Stage(
                "report_learning_curve",
                "Generate learning-curve reports",
                ("evaluate_learning_curve",),
                (
                    learning_report_dir / "learning_curve_report.md",
                    learning_report_dir / "learning_curve_report.html",
                    learning_report_dir / "plots" / "checkpoint_mean_profit_bb.png",
                    learning_report_dir / "plots" / "checkpoint_bb_per_100.png",
                    learning_report_dir / "plots" / "checkpoint_win_rate.png",
                    learning_report_dir / "plots" / "checkpoint_bust_rate.png",
                    learning_report_dir
                    / "plots"
                    / "checkpoint_global_classifier_accuracy.png",
                    learning_report_dir
                    / "plots"
                    / "checkpoint_global_classifier_coverage.png",
                ),
                "report",
                1,
                command=_python_module(
                    "src.experiments.reporting.create_learning_curve_report",
                    "--input-path",
                    paths.evaluations / "learning_curve_evaluation.csv",
                    "--output-dir",
                    learning_report_dir,
                    "--format",
                    "both",
                ),
            )
        )
        report_stage_ids.append("report_learning_curve")

    training_report_dir = paths.reports / "training_opponent"
    stages.append(
        Stage(
            "report_training_opponents",
            "Generate training-opponent reports",
            ("evaluate_training_opponents",),
            (
                training_report_dir / "training_opponent_report.md",
                training_report_dir / "training_opponent_report.html",
            ),
            "report",
            1,
            command=_python_module(
                "src.experiments.reporting.create_training_opponent_report",
                "--input-path",
                training_csv,
                "--output-dir",
                training_report_dir,
                "--format",
                "both",
            ),
        )
    )
    report_stage_ids.append("report_training_opponents")

    for stage_id, title, input_path, report_dir in (
        (
            "report_generalization_overview",
            "Generalization Evaluation Overview",
            generalization_csv,
            paths.reports / "generalization",
        ),
        (
            "report_stress_overview",
            "Stress-test Evaluation Overview",
            stress_csv,
            paths.reports / "stress_test",
        ),
        (
            "report_cross_play_overview",
            "Cross-play Evaluation Overview",
            cross_csv,
            paths.reports / "cross_play",
        ),
        (
            "report_baseline_overview",
            "Baseline Head-to-head Overview",
            baseline_csv,
            paths.reports / "baseline_head_to_head",
        ),
    ):
        stages.append(
            Stage(
                stage_id,
                f"Generate {title.lower()}",
                (
                    "merge_cross_play"
                    if input_path == cross_csv
                    else (
                        "evaluate_baseline_sanity"
                        if input_path == baseline_csv
                        else (
                            "evaluate_generalization"
                            if input_path == generalization_csv
                            else "evaluate_stress_test"
                        )
                    )
                ,),
                (
                    report_dir / "evaluation_overview.md",
                    report_dir / "evaluation_overview.json",
                    report_dir / "evaluation_overview.csv",
                ),
                "report",
                1,
                command=_python_module(
                    "src.experiments.reporting.create_evaluation_overview",
                    "--input-path",
                    input_path,
                    "--output-dir",
                    report_dir,
                    "--title",
                    title,
                ),
            )
        )
        report_stage_ids.append(stage_id)

    for stage_id, title, module, report_dir, expected_files in (
        (
            "report_algorithm_comparison",
            "Generate algorithm comparison report",
            "src.experiments.reporting.create_algorithm_comparison",
            paths.reports / "algorithm_comparison",
            (
                "algorithm_comparison.md",
                "algorithm_comparison.json",
                "algorithm_global_ranking.csv",
                "algorithm_by_opponent.csv",
                "algorithm_deltas.csv",
                "algorithm_global_ranking.tex",
                "algorithm_by_opponent.tex",
                "algorithm_deltas.tex",
                "charts/algorithm_mean_profit_by_opponent.png",
                "charts/algorithm_seed_stability_by_opponent.png",
                "charts/algorithm_global_mean_profit.png",
            ),
        ),
        (
            "report_seed_stability",
            "Generate seed stability report",
            "src.experiments.reporting.create_seed_stability_report",
            paths.reports / "seed_stability",
            (
                "seed_stability.md",
                "seed_stability.json",
                "seed_performance.csv",
                "seed_stability_summary.csv",
                "seed_rankings.csv",
                "ranking_stability.csv",
                "seed_stability_summary.tex",
                "seed_rankings.tex",
                "ranking_stability.tex",
            ),
        ),
        (
            "report_classifier_quality",
            "Generate classifier quality report",
            "src.experiments.reporting.create_classifier_quality_report",
            paths.reports / "classifier_quality",
            (
                "classifier_quality.md",
                "classifier_quality.json",
                "classifier_quality_summary.csv",
                "classifier_confusion_matrix.csv",
                "classifier_quality_summary.tex",
                "classifier_confusion_matrix.tex",
            ),
        ),
        (
            "report_experiment_summary",
            "Generate consolidated experiment summary",
            "src.experiments.reporting.create_experiment_summary",
            paths.reports / "experiment_summary",
            (
                "experiment_summary.md",
                "experiment_summary.json",
                "agent_ranking.csv",
                "deltas.csv",
                "quality_flags.csv",
                "agent_ranking.tex",
                "deltas.tex",
                "quality_flags.tex",
                "charts/mean_profit_ci_by_opponent.png",
                "charts/seed_stability_by_opponent.png",
            ),
        ),
    ):
        stages.append(
            Stage(
                stage_id,
                title,
                ("merge_primary_evaluation",),
                tuple(report_dir / filename for filename in expected_files),
                "report",
                1,
                command=_python_module(
                    module,
                    "--input-path",
                    primary_csv,
                    "--output-dir",
                    report_dir,
                    "--format",
                    "all",
                ),
            )
        )
        report_stage_ids.append(stage_id)

    final_summary = paths.reports / (
        f"final_thesis_summary_g{config.evaluation.games_per_matchup}.md"
    )
    stages.append(
        Stage(
            "finalize_pipeline",
            "Write final pipeline and thesis artifact summary",
            tuple(validation_stage_ids + report_stage_ids),
            (final_summary,),
            "internal",
            1,
            action="finalize_pipeline",
        )
    )
    _validate_stage_graph(stages)
    return tuple(stages)


def _validate_stage_graph(stages: Sequence[Stage]) -> None:
    seen: set[str] = set()
    for stage in stages:
        if stage.stage_id in seen:
            raise ValueError(f"Duplicate pipeline stage: {stage.stage_id}")
        missing = set(stage.dependencies) - seen
        if missing:
            raise ValueError(
                f"Stage {stage.stage_id} depends on later/unknown stages: "
                f"{sorted(missing)}"
            )
        seen.add(stage.stage_id)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _initial_manifest(
    context: PipelineContext,
    stages: Sequence[Stage],
) -> dict:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_status": "pending",
        "config": context.config_name,
        "protocol_id": context.experiment_config.protocol_id,
        "experiment_config_hash": context.experiment_config.config_hash,
        "training_config_hash": context.experiment_config.training_config_hash,
        "experiment_config": context.experiment_config.snapshot(),
        "pipeline_root": str(context.paths.root),
        "final_pipeline_dir": (
            str(context.final_pipeline_dir)
            if context.final_pipeline_dir is not None
            else None
        ),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "stages": {
            stage.stage_id: {
                "title": stage.title,
                "status": "pending",
                "dependencies": list(stage.dependencies),
                "outputs": [str(path) for path in stage.outputs],
                "command": list(stage.command),
                "action": stage.action,
                "kind": stage.kind,
                "work_units": stage.work_units,
                "fingerprint": stage.fingerprint(
                    context.experiment_config.config_hash
                ),
            }
            for stage in stages
        },
    }


def _write_pipeline_summary(
    context: PipelineContext,
    manifest: dict,
) -> None:
    stage_records = manifest["stages"]
    durations = [
        float(record.get("duration_seconds", 0.0))
        for record in stage_records.values()
        if record.get("status") == "success"
    ]
    summary = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_status": manifest["pipeline_status"],
        "config": context.config_name,
        "protocol_id": context.experiment_config.protocol_id,
        "experiment_config_hash": context.experiment_config.config_hash,
        "pipeline_root": str(context.paths.root),
        "model_source": (
            str(context.final_pipeline_dir)
            if context.config_name == EXTENDED_PRESET
            else str(context.paths.models)
        ),
        "completed_stages": sum(
            record.get("status") == "success"
            for record in stage_records.values()
        ),
        "failed_stages": [
            stage_id
            for stage_id, record in stage_records.items()
            if record.get("status") == "failed"
        ],
        "total_stages": len(stage_records),
        "recorded_stage_duration_seconds": sum(durations),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "updated_at": _utc_now(),
        "manifest_path": str(context.paths.manifest),
        "log_path": str(context.paths.log),
        "reports_dir": str(context.paths.reports),
        "stages": stage_records,
    }
    _atomic_write_json(context.paths.summary, summary)


def _outputs_exist(stage: Stage) -> bool:
    return bool(stage.outputs) and all(path.exists() for path in stage.outputs)


def _read_process_output(
    process: subprocess.Popen,
    logger: TeeLogger,
    stage_id: str,
    last_output: list[float],
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        if line.strip():
            last_output[0] = time.monotonic()
            logger.emit(f"[{stage_id}] {line.rstrip()}")


def _tail_progress_logs(
    context: PipelineContext,
    pattern: str,
    offsets: dict[Path, int],
    logger: TeeLogger,
    stage_id: str,
) -> None:
    for path in context.paths.root.glob(pattern):
        offset = offsets.get(path, 0)
        try:
            if path.stat().st_size < offset:
                offset = 0
            with path.open(encoding="utf-8", errors="replace") as file:
                file.seek(offset)
                lines = file.readlines()
                offsets[path] = file.tell()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if stripped and (
                "episode=" in stripped
                or "estimated_remaining=" in stripped
                or "finished" in stripped.lower()
                or "saved checkpoint" in stripped.lower()
            ):
                logger.emit(f"[{stage_id}] {path.parent.name}: {stripped}")


def run_subprocess_stage(
    context: PipelineContext,
    stage: Stage,
    logger: TeeLogger,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(stage.command),
        cwd=context.repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    last_output = [time.monotonic()]
    reader = threading.Thread(
        target=_read_process_output,
        args=(process, logger, stage.stage_id, last_output),
        daemon=True,
    )
    reader.start()
    offsets: dict[Path, int] = {}
    while process.poll() is None:
        if stage.progress_glob is not None:
            _tail_progress_logs(
                context,
                stage.progress_glob,
                offsets,
                logger,
                stage.stage_id,
            )
        now = time.monotonic()
        if now - last_output[0] >= context.heartbeat_seconds:
            logger.emit(
                f"[{stage.stage_id}] heartbeat: stage still running"
            )
            last_output[0] = now
        time.sleep(0.5)
    reader.join(timeout=5)
    if stage.progress_glob is not None:
        _tail_progress_logs(
            context,
            stage.progress_glob,
            offsets,
            logger,
            stage.stage_id,
        )
    if process.returncode != 0:
        raise PipelineError(
            f"Stage {stage.stage_id} failed with exit code "
            f"{process.returncode}."
        )


def _merge_csv_files(input_paths: Sequence[Path], output_path: Path) -> int:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in input_paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            for fieldname in reader.fieldnames:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            rows.extend(dict(row) for row in reader)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _merge_summary_files(
    input_paths: Sequence[Path],
    output_csv: Path,
    *,
    evaluation_type: str,
) -> None:
    summaries = [_load_json(path) for path in input_paths]
    first = summaries[0]
    provenance_fields = (
        "protocol_id",
        "preset_name",
        "experiment_config_hash",
        "training_config_hash",
        "experiment_config",
        "source_revision",
        "source_dirty",
        "model_training_config_hash",
        "model_source_revisions",
        "model_source_dirty",
    )
    for summary in summaries[1:]:
        for field in provenance_fields:
            if summary.get(field) != first.get(field):
                raise ValueError(
                    f"Cannot merge summaries with different {field}."
                )
        if summary.get("games") != first.get("games"):
            raise ValueError("Cannot merge summaries with different game budgets.")
    fragment_namespaces = {
        str(path): summary.get(
            "evaluation_seed_namespace",
            summary.get("eval_seed_base"),
        )
        for path, summary in zip(input_paths, summaries, strict=True)
    }
    namespaces = sorted(
        {
            namespace
            for namespace in fragment_namespaces.values()
            if namespace is not None
        }
    )
    scalar_namespace = namespaces[0] if len(namespaces) == 1 else None
    merged = {
        **{
            field: first.get(field)
            for field in provenance_fields
            if field in first
        },
        "evaluation_type": evaluation_type,
        "output_path": str(output_csv),
        "model_source": first.get("model_source"),
        "training_episodes": sorted(
            {
                episode
                for summary in summaries
                for episode in summary.get("training_episodes", [])
            }
        ),
        "seeds": first.get("seeds"),
        "games": first.get("games"),
        "eval_seed_base": scalar_namespace,
        "evaluation_seed_namespace": scalar_namespace,
        "evaluation_seed_namespaces": namespaces,
        "fragment_seed_namespaces": fragment_namespaces,
        "bundle_count": max(
            int(summary.get("bundle_count", 0)) for summary in summaries
        ),
        "row_count": sum(
            int(summary.get("row_count", 0)) for summary in summaries
        ),
        "duration_seconds": sum(
            float(summary.get("duration_seconds", 0.0))
            for summary in summaries
        ),
        "fragments": [str(path) for path in input_paths],
    }
    _atomic_write_json(_summary_path(output_csv), merged)


def _verify_final_models(context: PipelineContext) -> None:
    if context.final_pipeline_dir is None:
        raise PipelineError("Extended pipeline requires --final-pipeline-dir.")
    summary_path = context.final_pipeline_dir / "pipeline_summary.json"
    if not summary_path.exists():
        raise PipelineError(
            f"Final pipeline summary does not exist: {summary_path}"
        )
    final_summary = _load_json(summary_path)
    if final_summary.get("pipeline_status") != "success":
        raise PipelineError("The referenced final pipeline did not finish successfully.")
    bundles = discover_final_model_bundles(
        training_run_directory=context.model_paths.monte_carlo,
        seeds=context.experiment_config.training.seeds,
        skip_incomplete=False,
        q_learning_run_directory=context.model_paths.q_learning,
        sarsa_run_directory=context.model_paths.sarsa,
        double_q_learning_run_directory=context.model_paths.double_q_learning,
    )
    if len(bundles) != len(context.experiment_config.training.seeds):
        raise PipelineError("The final pipeline does not contain every required seed.")
    expected_hash = context.experiment_config.training_config_hash
    for bundle in bundles:
        if bundle.training_config_hash != expected_hash:
            raise PipelineError(
                "A reused model bundle does not match the extended training "
                f"configuration: seed={bundle.seed}."
            )
    _atomic_write_json(
        context.paths.root / "reused_final_models.json",
        {
            "source_pipeline": str(context.final_pipeline_dir),
            "training_config_hash": expected_hash,
            "seeds": [bundle.seed for bundle in bundles],
            "verified_at": _utc_now(),
        },
    )


def _merge_cross_play(context: PipelineContext) -> None:
    fragment_dir = context.paths.evaluations / "cross_play_fragments"
    csv_paths = [
        fragment_dir / "adaptive.csv",
        fragment_dir / "general.csv",
        *[
            fragment_dir / f"paired_{algorithm}.csv"
            for algorithm in ALGORITHM_KEYS
        ],
    ]
    output = context.paths.evaluations / "cross_play.csv"
    row_count = _merge_csv_files(csv_paths, output)
    _merge_summary_files(
        [_summary_path(path) for path in csv_paths],
        output,
        evaluation_type="cross_play",
    )
    summary = _load_json(_summary_path(output))
    if int(summary["row_count"]) != row_count:
        raise ValueError("Cross-play summary row count does not match merged CSV.")


def _merge_primary(context: PipelineContext) -> None:
    csv_paths = [
        context.paths.evaluations / "training_opponent.csv",
        context.paths.evaluations / "generalization.csv",
    ]
    output = context.paths.evaluations / "primary_evaluation.csv"
    row_count = _merge_csv_files(csv_paths, output)
    _merge_summary_files(
        [_summary_path(path) for path in csv_paths],
        output,
        evaluation_type="primary_evaluation",
    )
    summary = _load_json(_summary_path(output))
    if int(summary["row_count"]) != row_count:
        raise ValueError("Primary summary row count does not match merged CSV.")


def _finalize_pipeline(
    context: PipelineContext,
    manifest: dict,
) -> None:
    output = context.paths.reports / (
        "final_thesis_summary_"
        f"g{context.experiment_config.evaluation.games_per_matchup}.md"
    )
    report_entries = []
    for stage_id, record in manifest["stages"].items():
        for raw_output in record.get("outputs", []):
            path = Path(raw_output)
            if path.suffix.lower() in {".md", ".html", ".json", ".csv"}:
                report_entries.append((stage_id, path))
    lines = [
        "# Final Thesis Experiment Pipeline Summary",
        "",
        f"- **Preset:** `{context.config_name}`",
        f"- **Protocol:** `{context.experiment_config.protocol_id}`",
        f"- **Games per matchup:** "
        f"`{context.experiment_config.evaluation.games_per_matchup}`",
        f"- **Training seeds:** "
        f"`{list(context.experiment_config.training.seeds)}`",
        f"- **Model source:** "
        f"`{context.final_pipeline_dir or context.paths.models}`",
        "",
        "## Stage Durations",
        "",
        "| Stage | Status | Duration |",
        "| --- | --- | ---: |",
    ]
    for stage_id, record in manifest["stages"].items():
        display_status = (
            "success"
            if stage_id == "finalize_pipeline"
            else record.get("status")
        )
        duration = _format_duration(float(record.get("duration_seconds", 0.0)))
        lines.append(
            f"| `{stage_id}` | {display_status} | {duration} |"
        )
    lines.extend(["", "## Generated Artifacts", ""])
    for stage_id, path in report_entries:
        try:
            relative = path.relative_to(context.paths.root)
        except ValueError:
            relative = path
        lines.append(f"- `{stage_id}`: `{relative}`")
    lines.extend(
        [
            "",
            "Scientific conclusions must be taken from the generated experiment, "
            "algorithm, stability, classifier, and validation reports.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


INTERNAL_ACTIONS: dict[
    str,
    Callable[[PipelineContext, dict], None],
] = {
    "verify_final_models": lambda context, manifest: _verify_final_models(context),
    "merge_cross_play": lambda context, manifest: _merge_cross_play(context),
    "merge_primary": lambda context, manifest: _merge_primary(context),
    "finalize_pipeline": _finalize_pipeline,
}


def _prepare_manifest(
    context: PipelineContext,
    stages: Sequence[Stage],
) -> dict:
    if context.paths.manifest.exists():
        if not context.resume:
            raise PipelineError(
                f"Pipeline manifest already exists: {context.paths.manifest}. "
                "Use --resume or choose another --output-dir."
            )
        manifest = _load_json(context.paths.manifest)
        if manifest.get("experiment_config_hash") != (
            context.experiment_config.config_hash
        ):
            raise PipelineError(
                "Cannot resume a pipeline created with a different configuration."
            )
        current = _initial_manifest(context, stages)
        previous_stages = manifest.get("stages", {})
        for stage_id, record in current["stages"].items():
            previous = previous_stages.get(stage_id)
            if isinstance(previous, dict):
                for key in (
                    "status",
                    "started_at",
                    "finished_at",
                    "duration_seconds",
                    "error",
                ):
                    if key in previous:
                        record[key] = previous[key]
                record["fingerprint"] = previous.get("fingerprint")
        manifest = {
            **current,
            "created_at": manifest.get("created_at", current["created_at"]),
        }
    else:
        manifest = _initial_manifest(context, stages)
    manifest["updated_at"] = _utc_now()
    return manifest


def _can_resume_stage(
    context: PipelineContext,
    stage: Stage,
    record: dict,
    rerun_stages: set[str],
) -> bool:
    if not context.resume:
        return False
    if any(dependency in rerun_stages for dependency in stage.dependencies):
        return False
    return (
        record.get("status") == "success"
        and record.get("fingerprint")
        == stage.fingerprint(context.experiment_config.config_hash)
        and _outputs_exist(stage)
    )


def run_pipeline(
    context: PipelineContext,
    stages: Sequence[Stage] | None = None,
) -> int:
    selected_stages = tuple(stages or build_stages(context))
    if context.dry_run:
        for index, stage in enumerate(selected_stages, start=1):
            print(f"{index:02d}. {stage.stage_id}: {stage.title}")
            if stage.command:
                print("    " + subprocess.list2cmdline(list(stage.command)))
        return 0

    context.paths.root.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_manifest(context, selected_stages)
    manifest["pipeline_status"] = "running"
    manifest["started_at"] = manifest.get("started_at") or _utc_now()
    _atomic_write_json(context.paths.manifest, manifest)
    _write_pipeline_summary(context, manifest)

    overall_start = time.monotonic()
    rerun_stages: set[str] = set()
    with TeeLogger(context.paths.log) as logger:
        logger.emit(
            "Pipeline started: "
            f"config={context.config_name}, stages={len(selected_stages)}, "
            f"root={context.paths.root}"
        )
        for index, stage in enumerate(selected_stages, start=1):
            record = manifest["stages"][stage.stage_id]
            if _can_resume_stage(context, stage, record, rerun_stages):
                logger.emit(
                    f"[stage {index}/{len(selected_stages)}] RESUME-SKIP "
                    f"{stage.stage_id}: outputs are complete"
                )
                continue

            rerun_stages.add(stage.stage_id)
            remaining_ids = {
                candidate.stage_id
                for candidate in selected_stages[index - 1 :]
            }
            estimator = ProgressEstimator(
                selected_stages,
                manifest["stages"],
            )
            eta = estimator.estimate_remaining(remaining_ids)
            logger.emit(
                f"[stage {index}/{len(selected_stages)}] START "
                f"{stage.stage_id}: {stage.title}; elapsed="
                f"{_format_duration(time.monotonic() - overall_start)}, "
                f"estimated_remaining~{_format_duration(eta)}"
            )
            record.update(
                {
                    "status": "running",
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "error": None,
                    "fingerprint": stage.fingerprint(
                        context.experiment_config.config_hash
                    ),
                }
            )
            _atomic_write_json(context.paths.manifest, manifest)
            stage_start = time.monotonic()
            try:
                if stage.command:
                    run_subprocess_stage(context, stage, logger)
                elif stage.action is not None:
                    INTERNAL_ACTIONS[stage.action](context, manifest)
                else:
                    raise PipelineError(
                        f"Stage {stage.stage_id} has no command or action."
                    )
                missing_outputs = [
                    str(path) for path in stage.outputs if not path.exists()
                ]
                if missing_outputs:
                    raise PipelineError(
                        f"Stage {stage.stage_id} did not create outputs: "
                        f"{missing_outputs}"
                    )
            except Exception as error:
                duration = time.monotonic() - stage_start
                record.update(
                    {
                        "status": "failed",
                        "finished_at": _utc_now(),
                        "duration_seconds": duration,
                        "error": repr(error),
                    }
                )
                manifest["pipeline_status"] = "failed"
                manifest["failed_stage"] = stage.stage_id
                manifest["finished_at"] = _utc_now()
                manifest["updated_at"] = _utc_now()
                _atomic_write_json(context.paths.manifest, manifest)
                _write_pipeline_summary(context, manifest)
                logger.emit(
                    f"FAILED {stage.stage_id} after "
                    f"{_format_duration(duration)}: {error}"
                )
                return 1

            duration = time.monotonic() - stage_start
            record.update(
                {
                    "status": "success",
                    "finished_at": _utc_now(),
                    "duration_seconds": duration,
                    "error": None,
                }
            )
            manifest["updated_at"] = _utc_now()
            _atomic_write_json(context.paths.manifest, manifest)
            _write_pipeline_summary(context, manifest)
            remaining_ids.discard(stage.stage_id)
            eta = ProgressEstimator(
                selected_stages,
                manifest["stages"],
            ).estimate_remaining(remaining_ids)
            logger.emit(
                f"[stage {index}/{len(selected_stages)}] DONE "
                f"{stage.stage_id}; duration={_format_duration(duration)}, "
                f"elapsed={_format_duration(time.monotonic() - overall_start)}, "
                f"estimated_remaining~{_format_duration(eta)}"
            )

        manifest["pipeline_status"] = "success"
        manifest["finished_at"] = _utc_now()
        manifest["updated_at"] = _utc_now()
        _atomic_write_json(context.paths.manifest, manifest)
        _write_pipeline_summary(context, manifest)
        logger.emit(
            "Pipeline finished successfully; total_elapsed="
            f"{_format_duration(time.monotonic() - overall_start)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = build_context(args)
    try:
        return run_pipeline(context)
    except (OSError, TypeError, ValueError, PipelineError) as error:
        print(f"Pipeline configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

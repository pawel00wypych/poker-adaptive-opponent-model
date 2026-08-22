import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence

from src.config import TrainingConfig
from src.experiments.constants import (
    MODEL_TYPE_GENERAL_POLICY,
    MODEL_TYPES,
)
from src.training.constants import (
    ALPHA_MODE_CONSTANT,
    SUPPORTED_ALPHA_MODES,
    SUPPORTED_EPSILON_SCHEDULES,
)


@dataclass(frozen=True)
class TrainingJob:
    model_type: str
    seed: int
    episodes: int
    epsilon_schedule: str
    checkpoint_episodes: tuple[int, ...]
    experiment_directory: str
    log_interval: int
    alpha_mode: str = ALPHA_MODE_CONSTANT

    @property
    def run_name(self) -> str:
        if self.model_type == MODEL_TYPE_GENERAL_POLICY:
            return MODEL_TYPE_GENERAL_POLICY

        return f"specialist_{self.model_type}"

    @property
    def run_directory(self) -> Path:
        return (
            Path(self.experiment_directory)
            / f"seed_{self.seed}"
            / self.run_name
        )

    @property
    def final_model_path(self) -> Path:
        return self.run_directory / "final.pkl"

    @property
    def checkpoint_directory(self) -> Path:
        return self.run_directory / "checkpoints"

    @property
    def log_path(self) -> Path:
        return self.run_directory / "training.log"

    @property
    def result_path(self) -> Path:
        return self.run_directory / "job_result.json"


@dataclass
class JobResult:
    model_type: str
    seed: int
    episodes: int
    status: str
    return_code: int
    duration_seconds: float
    final_model_path: str
    checkpoint_directory: str
    log_path: str
    command: list[str]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible training experiments for "
            "the general policy and all specialist policies."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2026],
        help="Random seeds used for independent runs.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=TrainingConfig.episodes,
        help=(
            "Maximum number of episodes in each training run. Shared with the "
            "temporal-difference trainers so budgets stay comparable."
        ),
    )

    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        nargs="+",
        default=list(TrainingConfig.checkpoint_episodes),
        help=(
            "Episode counts saved as model checkpoints. Defaults stay inside "
            "the default episode budget so the suite starts without flags."
        ),
    )

    parser.add_argument(
        "--epsilon-schedule",
        choices=SUPPORTED_EPSILON_SCHEDULES,
        default=SUPPORTED_EPSILON_SCHEDULES[0],
    )

    parser.add_argument(
        "--alpha-mode",
        choices=SUPPORTED_ALPHA_MODES,
        default=ALPHA_MODE_CONSTANT,
        help=(
            "Monte Carlo learning-rate mode. constant uses fixed alpha, "
            "visit_count uses 1/N(s,a), and sqrt_visit uses "
            "1/sqrt(N(s,a))."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            4,
            os.cpu_count() or 1,
        ),
        help=(
            "Maximum number of training processes "
            "running in parallel."
        ),
    )

    parser.add_argument(
        "--models",
        choices=MODEL_TYPES,
        nargs="+",
        default=list(MODEL_TYPES),
        help=(
            "Model families included in the experiment."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="results/training_runs",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Optional experiment directory name. "
            "A timestamped name is generated when omitted."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--rerun-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run jobs even when their final model already exists."
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error(
            "--episodes must be greater than zero"
        )

    if args.workers <= 0:
        parser.error(
            "--workers must be greater than zero"
        )

    if args.log_interval <= 0:
        parser.error(
            "--log-interval must be greater than zero"
        )

    if any(
        checkpoint <= 0
        for checkpoint in args.checkpoint_episodes
    ):
        parser.error(
            "Checkpoint episodes must be positive"
        )

    if any(
        checkpoint > args.episodes
        for checkpoint in args.checkpoint_episodes
    ):
        parser.error(
            "Checkpoint episodes cannot exceed "
            "--episodes"
        )

    if len(set(args.seeds)) != len(args.seeds):
        parser.error(
            "--seeds must not contain duplicates"
        )

    return args


def build_command(job: TrainingJob) -> list[str]:
    common_arguments = [
        "--episodes",
        str(job.episodes),
        "--seed",
        str(job.seed),
        "--epsilon-schedule",
        job.epsilon_schedule,
        "--alpha-mode",
        job.alpha_mode,
        "--output-path",
        str(job.final_model_path),
        "--checkpoint-directory",
        str(job.checkpoint_directory),
        "--checkpoint-episodes",
        *[
            str(value)
            for value in job.checkpoint_episodes
        ],
        "--progress",
        "--log-interval",
        str(job.log_interval),
        "--no-player-verbose",
        "--no-engine-verbose",
    ]

    if job.model_type == MODEL_TYPE_GENERAL_POLICY:
        return [
            sys.executable,
            "-m",
            "src.experiments.training.run_general_policy_training",
            *common_arguments,
        ]

    return [
        sys.executable,
        "-m",
        "src.experiments.training.run_specialist_training",
        "--opponent",
        job.model_type,
        *common_arguments,
    ]


def run_job(job: TrainingJob) -> JobResult:
    job.run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    job.checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = build_command(job)

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(job.seed)

    start = perf_counter()

    with job.log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            "COMMAND:\n"
            + subprocess.list2cmdline(command)
            + "\n\n"
        )
        log_file.flush()

        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )

    duration = perf_counter() - start

    status = (
        "success"
        if process.returncode == 0
        else "failed"
    )

    result = JobResult(
        model_type=job.model_type,
        seed=job.seed,
        episodes=job.episodes,
        status=status,
        return_code=process.returncode,
        duration_seconds=duration,
        final_model_path=str(job.final_model_path),
        checkpoint_directory=str(
            job.checkpoint_directory
        ),
        log_path=str(job.log_path),
        command=command,
        error=(
            None
            if process.returncode == 0
            else (
                "Training process returned "
                f"code {process.returncode}"
            )
        ),
    )

    with job.result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(result),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return result


def build_jobs(
    models: Sequence[str],
    seeds: Sequence[int],
    episodes: int,
    epsilon_schedule: str,
    checkpoint_episodes: Sequence[int],
    experiment_directory: str,
    log_interval: int,
    rerun_existing: bool,
    alpha_mode: str = ALPHA_MODE_CONSTANT,
) -> tuple[list[TrainingJob], list[TrainingJob]]:
    runnable: list[TrainingJob] = []
    skipped: list[TrainingJob] = []

    for seed in seeds:
        for model_type in models:
            job = TrainingJob(
                model_type=model_type,
                seed=seed,
                episodes=episodes,
                epsilon_schedule=epsilon_schedule,
                alpha_mode=alpha_mode,
                checkpoint_episodes=tuple(
                    sorted(
                        set(checkpoint_episodes)
                        | {episodes}
                    )
                ),
                experiment_directory=(
                    experiment_directory
                ),
                log_interval=log_interval,
            )

            if (
                job.final_model_path.exists()
                and not rerun_existing
            ):
                skipped.append(job)
            else:
                runnable.append(job)

    return runnable, skipped


def save_manifest(
    path: Path,
    arguments: argparse.Namespace,
    jobs: Sequence[TrainingJob],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "python_executable": sys.executable,
        "arguments": vars(arguments),
        "jobs": [
            {
                "model_type": job.model_type,
                "seed": job.seed,
                "episodes": job.episodes,
                "epsilon_schedule": job.epsilon_schedule,
                "alpha_mode": job.alpha_mode,
                "final_model_path": str(
                    job.final_model_path
                ),
                "checkpoint_directory": str(
                    job.checkpoint_directory
                ),
                "log_path": str(job.log_path),
            }
            for job in jobs
        ],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_result(result: JobResult) -> None:
    duration_minutes = (
        result.duration_seconds / 60
    )

    print(
        f"[{result.status.upper()}] "
        f"model={result.model_type}, "
        f"seed={result.seed}, "
        f"episodes={result.episodes}, "
        f"duration={duration_minutes:.2f} min, "
        f"log={result.log_path}"
    )


def main() -> None:
    args = parse_args()

    experiment_name = (
        args.experiment_name
        or (
            datetime.now()
            .strftime("%Y%m%d_%H%M%S")
            + f"_{args.epsilon_schedule}"
            + f"_ep{args.episodes}"
        )
    )

    experiment_directory = str(
        Path(args.output_root)
        / experiment_name
    )

    runnable_jobs, skipped_jobs = build_jobs(
        models=args.models,
        seeds=args.seeds,
        episodes=args.episodes,
        epsilon_schedule=args.epsilon_schedule,
        alpha_mode=args.alpha_mode,
        checkpoint_episodes=(
            args.checkpoint_episodes
        ),
        experiment_directory=(
            experiment_directory
        ),
        log_interval=args.log_interval,
        rerun_existing=args.rerun_existing,
    )

    all_jobs = [
        *runnable_jobs,
        *skipped_jobs,
    ]

    save_manifest(
        path=(
            Path(experiment_directory)
            / "manifest.json"
        ),
        arguments=args,
        jobs=all_jobs,
    )

    print(
        f"Experiment: {experiment_directory}\n"
        f"Models: {args.models}\n"
        f"Seeds: {args.seeds}\n"
        f"Episodes: {args.episodes}\n"
        f"Alpha mode: {args.alpha_mode}\n"
        f"Checkpoints: "
        f"{sorted(set(args.checkpoint_episodes) | {args.episodes})}\n"
        f"Workers: {args.workers}\n"
        f"Runnable jobs: {len(runnable_jobs)}\n"
        f"Skipped jobs: {len(skipped_jobs)}"
    )

    for job in skipped_jobs:
        print(
            "[SKIPPED] "
            f"model={job.model_type}, "
            f"seed={job.seed}, "
            f"model already exists: "
            f"{job.final_model_path}"
        )

    if not runnable_jobs:
        print(
            "No training jobs to run."
        )
        return

    suite_start = perf_counter()
    results: list[JobResult] = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_to_job: dict[
            Future[JobResult],
            TrainingJob,
        ] = {
            executor.submit(
                run_job,
                job,
            ): job
            for job in runnable_jobs
        }

        for future in as_completed(
            future_to_job
        ):
            job = future_to_job[future]

            try:
                result = future.result()
            except Exception as error:
                result = JobResult(
                    model_type=job.model_type,
                    seed=job.seed,
                    episodes=job.episodes,
                    status="failed",
                    return_code=-1,
                    duration_seconds=0.0,
                    final_model_path=str(
                        job.final_model_path
                    ),
                    checkpoint_directory=str(
                        job.checkpoint_directory
                    ),
                    log_path=str(job.log_path),
                    command=build_command(job),
                    error=repr(error),
                )

            results.append(result)
            print_result(result)

    suite_duration = (
        perf_counter() - suite_start
    )

    summary = {
        "experiment_directory": (
            experiment_directory
        ),
        "duration_seconds": suite_duration,
        "successful_jobs": sum(
            result.status == "success"
            for result in results
        ),
        "failed_jobs": sum(
            result.status == "failed"
            for result in results
        ),
        "results": [
            asdict(result)
            for result in results
        ],
    }

    summary_path = (
        Path(experiment_directory)
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nTraining suite finished\n"
        f"duration_minutes="
        f"{suite_duration / 60:.2f}\n"
        f"successful_jobs="
        f"{summary['successful_jobs']}\n"
        f"failed_jobs="
        f"{summary['failed_jobs']}\n"
        f"summary={summary_path}"
    )

    if summary["failed_jobs"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

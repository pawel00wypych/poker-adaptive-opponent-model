from pathlib import Path

from src.experiments.run_training_suite import (
    TrainingJob,
    build_command,
    build_jobs,
)


def test_single_policy_command_uses_correct_module(
    tmp_path,
):
    job = TrainingJob(
        model_type="single_policy",
        seed=42,
        episodes=5_000,
        epsilon_schedule="linear",
        checkpoint_episodes=(
            1_000,
            2_500,
            5_000,
        ),
        experiment_directory=str(tmp_path),
        log_interval=1_000,
    )

    command = build_command(job)

    assert (
        "src.experiments.run_single_policy_training"
        in command
    )

    assert "--seed" in command
    assert "42" in command
    assert "--output-path" in command


def test_specialist_command_contains_opponent(
    tmp_path,
):
    job = TrainingJob(
        model_type="calling",
        seed=123,
        episodes=10_000,
        epsilon_schedule="linear",
        checkpoint_episodes=(
            5_000,
            10_000,
        ),
        experiment_directory=str(tmp_path),
        log_interval=1_000,
    )

    command = build_command(job)

    assert (
        "src.experiments.run_specialist_training"
        in command
    )

    opponent_index = command.index(
        "--opponent"
    )

    assert (
        command[opponent_index + 1]
        == "calling"
    )


def test_job_paths_are_unique_for_seeds(
    tmp_path,
):
    first = TrainingJob(
        model_type="tight",
        seed=42,
        episodes=5_000,
        epsilon_schedule="linear",
        checkpoint_episodes=(5_000,),
        experiment_directory=str(tmp_path),
        log_interval=1_000,
    )

    second = TrainingJob(
        model_type="tight",
        seed=123,
        episodes=5_000,
        epsilon_schedule="linear",
        checkpoint_episodes=(5_000,),
        experiment_directory=str(tmp_path),
        log_interval=1_000,
    )

    assert (
        first.final_model_path
        != second.final_model_path
    )

    assert (
        first.checkpoint_directory
        != second.checkpoint_directory
    )


def test_build_jobs_creates_every_model_seed_pair(
    tmp_path,
):
    runnable, skipped = build_jobs(
        models=[
            "single_policy",
            "tight",
            "aggressive",
            "calling",
        ],
        seeds=[42, 123],
        episodes=5_000,
        epsilon_schedule="linear",
        checkpoint_episodes=[
            1_000,
            5_000,
        ],
        experiment_directory=str(tmp_path),
        log_interval=1_000,
        rerun_existing=False,
    )

    assert len(runnable) == 8
    assert skipped == []


def test_existing_final_model_is_skipped(
    tmp_path,
):
    job_directory = (
        Path(tmp_path)
        / "seed_42"
        / "single_policy"
    )

    job_directory.mkdir(
        parents=True
    )

    (
        job_directory / "final.pkl"
    ).write_bytes(b"existing")

    runnable, skipped = build_jobs(
        models=["single_policy"],
        seeds=[42],
        episodes=5_000,
        epsilon_schedule="linear",
        checkpoint_episodes=[5_000],
        experiment_directory=str(tmp_path),
        log_interval=1_000,
        rerun_existing=False,
    )

    assert runnable == []
    assert len(skipped) == 1

def test_training_command_contains_alpha_mode(tmp_path):
    job = TrainingJob(
        model_type="calling",
        seed=42,
        episodes=4_000,
        epsilon_schedule="linear",
        alpha_mode="sqrt_visit",
        checkpoint_episodes=(
            500,
            4_000,
        ),
        experiment_directory=str(tmp_path),
        log_interval=1_000,
    )

    command = build_command(job)

    alpha_mode_index = command.index("--alpha-mode")
    assert command[alpha_mode_index + 1] == "sqrt_visit"

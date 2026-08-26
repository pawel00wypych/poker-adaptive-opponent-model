from pathlib import Path
import json

import pytest

from src.experiment_protocol import (
    FINAL_EXPERIMENT_CONFIG,
    build_protocol_provenance,
)
from src.experiments.training.run_monte_carlo_suite import (
    TrainingJob,
    build_command,
    build_jobs,
)


def test_general_policy_command_uses_correct_module(
    tmp_path,
):
    job = TrainingJob(
        model_type="general_policy",
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
        "src.experiments.training.run_general_policy_training"
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
        "src.experiments.training.run_specialist_training"
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
            "general_policy",
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
        / "general_policy"
    )

    job_directory.mkdir(
        parents=True
    )

    (
        job_directory / "final.pkl"
    ).write_bytes(b"existing")

    runnable, skipped = build_jobs(
        models=["general_policy"],
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


def test_protocol_aware_suite_refuses_to_skip_legacy_model(tmp_path):
    job_directory = tmp_path / "seed_42" / "general_policy"
    job_directory.mkdir(parents=True)
    (job_directory / "final.pkl").write_bytes(b"existing")
    provenance = build_protocol_provenance(
        FINAL_EXPERIMENT_CONFIG,
        source_revision="revision",
        source_dirty=False,
    )

    with pytest.raises(ValueError, match="incomplete protocol-aware"):
        build_jobs(
            models=["general_policy"],
            seeds=[42],
            episodes=10_000,
            epsilon_schedule="linear",
            checkpoint_episodes=[1_000, 10_000],
            experiment_directory=str(tmp_path),
            log_interval=1_000,
            rerun_existing=False,
            protocol_provenance=provenance,
        )


def test_protocol_aware_suite_skips_matching_existing_model(tmp_path):
    job_directory = tmp_path / "seed_42" / "general_policy"
    job_directory.mkdir(parents=True)
    (job_directory / "final.pkl").write_bytes(b"existing")
    provenance = build_protocol_provenance(
        FINAL_EXPERIMENT_CONFIG,
        source_revision="revision",
        source_dirty=False,
    )
    (job_directory / "final.json").write_text(
        json.dumps(
            {
                **provenance.to_dict(),
                "seed": 42,
                "completed_episodes": 10_000,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_directory = job_directory / "checkpoints"
    checkpoint_directory.mkdir()
    for episode in (1_000, 10_000):
        checkpoint_path = checkpoint_directory / (
            f"general_policy_episodes_{episode}_seed_42.pkl"
        )
        checkpoint_path.write_bytes(b"checkpoint")
        checkpoint_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    **provenance.to_dict(),
                    "seed": 42,
                    "completed_episodes": episode,
                }
            ),
            encoding="utf-8",
        )

    runnable, skipped = build_jobs(
        models=["general_policy"],
        seeds=[42],
        episodes=10_000,
        epsilon_schedule="linear",
        checkpoint_episodes=[1_000, 10_000],
        experiment_directory=str(tmp_path),
        log_interval=1_000,
        rerun_existing=False,
        protocol_provenance=provenance,
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

from pathlib import Path

from src.training.checkpoint_utils import (
    build_checkpoint_episodes,
    build_checkpoint_path,
    resolve_checkpoint_episodes,
)


def test_build_configured_checkpoint_episodes():
    checkpoints = build_checkpoint_episodes(
        total_episodes=10_000,
        configured_checkpoints=(
            1_000,
            2_500,
            5_000,
            10_000,
            25_000,
        ),
        checkpoint_interval=None,
    )

    assert checkpoints == {
        1_000,
        2_500,
        5_000,
        10_000,
    }


def test_final_episode_is_always_checkpoint():
    checkpoints = build_checkpoint_episodes(
        total_episodes=8_000,
        configured_checkpoints=(
            1_000,
            5_000,
        ),
        checkpoint_interval=None,
    )

    assert 8_000 in checkpoints


def test_checkpoint_interval():
    checkpoints = build_checkpoint_episodes(
        total_episodes=10_000,
        configured_checkpoints=(),
        checkpoint_interval=2_500,
    )

    assert checkpoints == {
        2_500,
        5_000,
        7_500,
        10_000,
    }


def test_effective_checkpoint_schedule_filters_the_preset_to_a_short_run():
    checkpoints = resolve_checkpoint_episodes(
        total_episodes=2,
        configured_checkpoints=(1_000, 2_500, 10_000),
        checkpoints_enabled=True,
        checkpoint_interval=None,
    )

    assert checkpoints == (2,)


def test_disabled_checkpoints_produce_an_empty_protocol_schedule():
    assert resolve_checkpoint_episodes(
        total_episodes=10_000,
        configured_checkpoints=(1_000, 10_000),
        checkpoints_enabled=False,
        checkpoint_interval=None,
    ) == ()


def test_checkpoint_path_contains_parameters():
    path = build_checkpoint_path(
        checkpoint_directory=(
            "results/models/checkpoints"
        ),
        model_name=(
            "monte_carlo_vs_calling"
        ),
        completed_episodes=5_000,
        seed=42,
    )

    assert Path(path).name == (
        "monte_carlo_vs_calling"
        "_episodes_5000"
        "_seed_42.pkl"
    )

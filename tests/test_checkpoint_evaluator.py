from pathlib import Path

import pytest

from src.evaluation.checkpoint_evaluator import (
    ModelBundle,
    build_game_seed,
    build_model_bundle,
    build_result_row,
    checkpoint_filename,
    checkpoint_model_path,
    discover_model_bundles,
    discover_seed_directories,
    parse_seed_from_directory,
)


def create_checkpoint_bundle_files(
    root: Path,
    seed: int,
    episode: int,
) -> None:
    files = [
        (
            "single_policy",
            "single_policy",
        ),
        (
            "specialist_fish",
            "specialist_fish",
        ),
        (
            "specialist_aggressive",
            "specialist_aggressive",
        ),
        (
            "specialist_calling",
            "specialist_calling",
        ),
    ]

    for directory_name, prefix in files:
        checkpoint_directory = (
            root
            / f"seed_{seed}"
            / directory_name
            / "checkpoints"
        )

        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            checkpoint_directory
            / (
                f"{prefix}"
                f"_episodes_{episode}"
                f"_seed_{seed}.pkl"
            )
        )

        checkpoint_path.write_bytes(
            b"dummy-model"
        )


def test_checkpoint_filename_for_single_policy():
    filename = checkpoint_filename(
        policy_type="unknown",
        checkpoint_episode=5000,
        seed=42,
    )

    assert filename == (
        "single_policy"
        "_episodes_5000"
        "_seed_42.pkl"
    )


def test_checkpoint_filename_for_specialist():
    filename = checkpoint_filename(
        policy_type="calling",
        checkpoint_episode=7500,
        seed=123,
    )

    assert filename == (
        "specialist_calling"
        "_episodes_7500"
        "_seed_123.pkl"
    )


def test_parse_seed_from_directory():
    seed = parse_seed_from_directory(
        Path("seed_2026")
    )

    assert seed == 2026


def test_parse_seed_from_directory_rejects_invalid_name():
    with pytest.raises(
        ValueError,
        match="Invalid seed directory name",
    ):
        parse_seed_from_directory(
            Path("model_42")
        )


def test_discover_seed_directories_sorts_by_seed(
    tmp_path,
):
    (tmp_path / "seed_123").mkdir()
    (tmp_path / "seed_42").mkdir()
    (tmp_path / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    discovered = discover_seed_directories(
        tmp_path
    )

    assert [
        path.name
        for path in discovered
    ] == [
        "seed_42",
        "seed_123",
    ]


def test_checkpoint_model_path_for_calling(
    tmp_path,
):
    seed_directory = tmp_path / "seed_42"

    path = checkpoint_model_path(
        seed_directory=seed_directory,
        policy_type="calling",
        checkpoint_episode=5000,
        seed=42,
    )

    assert path == (
        seed_directory
        / "specialist_calling"
        / "checkpoints"
        / (
            "specialist_calling"
            "_episodes_5000"
            "_seed_42.pkl"
        )
    )


def test_build_model_bundle_from_complete_files(
    tmp_path,
):
    create_checkpoint_bundle_files(
        root=tmp_path,
        seed=42,
        episode=5000,
    )

    bundle = build_model_bundle(
        training_run_directory=tmp_path,
        seed=42,
        checkpoint_episode=5000,
    )

    assert isinstance(
        bundle,
        ModelBundle,
    )
    assert bundle.seed == 42
    assert bundle.checkpoint_episode == 5000
    assert bundle.unknown_model_path.exists()
    assert bundle.fish_model_path.exists()
    assert bundle.aggressive_model_path.exists()
    assert bundle.calling_model_path.exists()


def test_build_model_bundle_fails_when_incomplete(
    tmp_path,
):
    create_checkpoint_bundle_files(
        root=tmp_path,
        seed=42,
        episode=5000,
    )

    missing = (
        tmp_path
        / "seed_42"
        / "specialist_calling"
        / "checkpoints"
        / (
            "specialist_calling"
            "_episodes_5000"
            "_seed_42.pkl"
        )
    )

    missing.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Incomplete model bundle",
    ):
        build_model_bundle(
            training_run_directory=tmp_path,
            seed=42,
            checkpoint_episode=5000,
        )


def test_discover_model_bundles_skips_incomplete(
    tmp_path,
):
    create_checkpoint_bundle_files(
        root=tmp_path,
        seed=42,
        episode=1000,
    )

    create_checkpoint_bundle_files(
        root=tmp_path,
        seed=42,
        episode=5000,
    )

    missing = (
        tmp_path
        / "seed_42"
        / "specialist_fish"
        / "checkpoints"
        / (
            "specialist_fish"
            "_episodes_5000"
            "_seed_42.pkl"
        )
    )

    missing.unlink()

    bundles = discover_model_bundles(
        training_run_directory=tmp_path,
        checkpoint_episodes=[
            1000,
            5000,
        ],
        seeds=[
            42,
        ],
        skip_incomplete=True,
    )

    assert len(bundles) == 1
    assert bundles[0].checkpoint_episode == 1000


def test_build_game_seed_is_deterministic():
    first = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        game_id=7,
    )

    second = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        game_id=7,
    )

    assert first == second


def test_build_game_seed_changes_for_different_game():
    first = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        game_id=7,
    )

    second = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        game_id=8,
    )

    assert first != second


def test_build_result_row_calculates_profit_fields(
    tmp_path,
):
    bundle = ModelBundle(
        training_run_directory=tmp_path / "run",
        seed=42,
        checkpoint_episode=5000,
        unknown_model_path=Path("unknown.pkl"),
        fish_model_path=Path("fish.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )

    row = build_result_row(
        bundle=bundle,
        tested_agent_name="adaptive_mc",
        opponent_name="calling",
        game_id=3,
        final_stack=250,
        initial_stack=200,
        hands_played=20,
        big_blind=10,
        ended_by_bust=True,
        ended_by_round_limit=False,
        classifier_metrics={
            "classified_decisions": 10,
            "correct_classifications": 8,
            "incorrect_classifications": 2,
            "unknown_classifications": 1,
            "classifier_accuracy": 0.8,
            "classifier_coverage": 0.9,
            "policy_switches": 2,
            "first_classification_hand": 3,
            "first_correct_classification_hand": 4,
            "first_classification_action_count": 5,
            "first_correct_classification_action_count": 6,
            "final_predicted_type": "calling",
        },
    )

    assert row["profit"] == 50
    assert row["profit_bb"] == 5.0
    assert row["won_game"] == 1
    assert row["busted"] == 0
    assert row["ended_by_bust"] == 1
    assert row["ended_by_round_limit"] == 0
    assert row["model_seed"] == 42
    assert row["checkpoint_episode"] == 5000
    assert row["final_predicted_type"] == "calling"
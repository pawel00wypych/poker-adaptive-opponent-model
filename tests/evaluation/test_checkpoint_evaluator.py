from pathlib import Path

import pytest

from src.evaluation.runners.checkpoint_evaluator import (
    CheckpointEvaluationConfig,
    ModelBundle,
    build_game_seed,
    build_model_bundle,
    build_result_row,
    checkpoint_filename,
    checkpoint_model_path,
    discover_model_bundles,
    discover_seed_directories,
    evaluate_bundle,
    parse_seed_from_directory,
)


def create_checkpoint_bundle_files(
    root: Path,
    seed: int,
    episode: int,
) -> None:
    files = [
        (
            "general_policy",
            "general_policy",
        ),
        (
            "specialist_tight",
            "specialist_tight",
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


def test_checkpoint_filename_for_general_policy():
    filename = checkpoint_filename(
        policy_type="unknown",
        checkpoint_episode=5000,
        seed=42,
    )

    assert filename == (
        "general_policy"
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
    assert bundle.tight_model_path.exists()
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
        / "specialist_tight"
        / "checkpoints"
        / (
            "specialist_tight"
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
        matchup_game_index=7,
    )

    second = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        matchup_game_index=7,
    )

    assert first == second


def test_build_game_seed_changes_for_different_game():
    first = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        matchup_game_index=7,
    )

    second = build_game_seed(
        eval_seed_base=100_000,
        model_seed=42,
        checkpoint_episode=5000,
        matchup_game_index=8,
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
        tight_model_path=Path("tight.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )

    row = build_result_row(
        bundle=bundle,
        tested_agent_name="adaptive_mc",
        opponent_name="calling",
        game_id=3,
        matchup_game_index=1,
        evaluation_seed=123_456,
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
    assert row["game_id"] == 3
    assert row["matchup_game_index"] == 1
    assert row["evaluation_seed"] == 123_456
    assert row["final_predicted_type"] == "calling"


def test_evaluate_bundle_restarts_game_index_for_each_matchup(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_evaluate_single_game(**kwargs):
        calls.append(
            (
                kwargs["tested_agent_name"],
                kwargs["opponent_name"],
                kwargs["game_id"],
                kwargs["matchup_game_index"],
            )
        )
        return kwargs

    monkeypatch.setattr(
        "src.evaluation.runners.checkpoint_evaluator.evaluate_single_game",
        fake_evaluate_single_game,
    )

    bundle = ModelBundle(
        training_run_directory=tmp_path / "run",
        seed=42,
        checkpoint_episode=5000,
        unknown_model_path=Path("unknown.pkl"),
        tight_model_path=Path("tight.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )
    config = CheckpointEvaluationConfig(
        games_per_matchup=2,
        opponents=("calling", "tight"),
        tested_agents=("adaptive_mc", "oracle_mc"),
        eval_seed_base=100_000,
        output_path=tmp_path / "evaluation.csv",
    )

    rows = evaluate_bundle(
        bundle=bundle,
        config=config,
    )

    assert len(rows) == 8
    assert [call[2] for call in calls] == list(range(8))
    assert [call[3] for call in calls] == [0, 1] * 4

from src.evaluation.constants import (
    CROSS_POLICY_AGENT_TO_POLICY_TYPE,
    SUPPORTED_TESTED_AGENTS,
)
from src.evaluation.runners.checkpoint_evaluator import (
    get_classifier_metrics,
)
from src.players.learned.fixed_policy_player import FixedPolicyPlayer
from src.players.learned.oracle_player import OraclePlayer


def test_supported_agents_include_oracle_and_cross_policy_agents():
    assert "oracle_mc" in SUPPORTED_TESTED_AGENTS
    assert "oracle_q_learning" in SUPPORTED_TESTED_AGENTS
    assert "oracle_sarsa" in SUPPORTED_TESTED_AGENTS
    assert "oracle_double_q_learning" in SUPPORTED_TESTED_AGENTS
    assert "policy_general_mc" in SUPPORTED_TESTED_AGENTS
    assert "policy_tight" in SUPPORTED_TESTED_AGENTS
    assert "policy_aggressive" in SUPPORTED_TESTED_AGENTS
    assert "policy_calling" in SUPPORTED_TESTED_AGENTS


def test_cross_policy_agent_mapping():
    assert (
        CROSS_POLICY_AGENT_TO_POLICY_TYPE["policy_general_mc"]
        == "unknown"
    )
    assert (
        CROSS_POLICY_AGENT_TO_POLICY_TYPE["policy_tight"]
        == "tight"
    )
    assert (
        CROSS_POLICY_AGENT_TO_POLICY_TYPE["policy_aggressive"]
        == "aggressive"
    )
    assert (
        CROSS_POLICY_AGENT_TO_POLICY_TYPE["policy_calling"]
        == "calling"
    )


def test_oracle_classifier_metrics_are_perfect(
    adaptive_agents,
):
    player = OraclePlayer(
        agents=adaptive_agents,
        oracle_opponent_type="calling",
    )

    metrics = get_classifier_metrics(
        player
    )

    assert metrics["classified_decisions"] == 1
    assert metrics["correct_classifications"] == 1
    assert metrics["incorrect_classifications"] == 0
    assert metrics["unknown_classifications"] == 0
    assert metrics["classifier_accuracy"] == 1.0
    assert metrics["classifier_coverage"] == 1.0
    assert metrics["policy_switches"] == 0
    assert metrics["final_predicted_type"] == "calling"


def test_fixed_policy_classifier_metrics_are_empty(
    eval_agent,
):
    player = FixedPolicyPlayer(
        agent=eval_agent,
        policy_type="calling",
    )

    metrics = get_classifier_metrics(
        player
    )

    assert metrics["classified_decisions"] == 0
    assert metrics["correct_classifications"] == 0
    assert metrics["incorrect_classifications"] == 0
    assert metrics["unknown_classifications"] == 0
    assert metrics["classifier_accuracy"] == 0.0
    assert metrics["classifier_coverage"] == 0.0
    assert metrics["final_predicted_type"] == ""

import json
from pathlib import Path

import pytest

from src.evaluation.runners.learning_curve_evaluator import (
    LearningCurveEvaluationConfig,
    build_checkpoint_model_bundle,
    checkpoint_filename,
    checkpoint_model_path,
    discover_checkpoint_model_bundles,
    evaluate_learning_curve_bundle,
)
from src.evaluation.runners.model_evaluator import build_result_row

POLICIES = (
    ("general_policy", "general_policy"),
    ("specialist_tight", "specialist_tight"),
    ("specialist_aggressive", "specialist_aggressive"),
    ("specialist_calling", "specialist_calling"),
)


def create_checkpoint_files(root: Path, *, seed: int, episode: int) -> None:
    for directory_name, prefix in POLICIES:
        path = (
            root
            / f"seed_{seed}"
            / directory_name
            / "checkpoints"
            / f"{prefix}_episodes_{episode}_seed_{seed}.pkl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy-checkpoint")


def empty_classifier_metrics() -> dict:
    return {
        "classified_decisions": 0,
        "correct_classifications": 0,
        "incorrect_classifications": 0,
        "unknown_classifications": 0,
        "other_classifications": 0,
        "classifier_accuracy": 0.0,
        "classifier_coverage": 0.0,
        "policy_switches": 0,
        "first_classification_hand": None,
        "first_correct_classification_hand": None,
        "first_classification_action_count": None,
        "first_correct_classification_action_count": None,
        "final_predicted_type": "",
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }


def test_checkpoint_naming_is_scoped_to_learning_curve():
    assert checkpoint_filename("unknown", 5000, 42) == (
        "general_policy_episodes_5000_seed_42.pkl"
    )
    assert checkpoint_filename("calling", 7500, 123) == (
        "specialist_calling_episodes_7500_seed_123.pkl"
    )


def test_checkpoint_model_path_uses_checkpoint_directory(tmp_path):
    path = checkpoint_model_path(tmp_path / "seed_42", "calling", 5000, 42)

    assert path == (
        tmp_path
        / "seed_42"
        / "specialist_calling"
        / "checkpoints"
        / "specialist_calling_episodes_5000_seed_42.pkl"
    )


def test_build_checkpoint_bundle_marks_diagnostic_source(tmp_path):
    create_checkpoint_files(tmp_path, seed=42, episode=5000)

    bundle = build_checkpoint_model_bundle(tmp_path, seed=42, checkpoint_episode=5000)

    assert bundle.model_source == "checkpoint"
    assert bundle.training_episode is None
    assert bundle.checkpoint_episode == 5000


def test_partial_legacy_checkpoint_sidecars_remain_readable(tmp_path):
    create_checkpoint_files(tmp_path, seed=42, episode=5000)
    sidecar = (
        tmp_path
        / "seed_42"
        / "specialist_tight"
        / "checkpoints"
        / "specialist_tight_episodes_5000_seed_42.json"
    )
    sidecar.write_text(
        json.dumps({"seed": 42, "completed_episodes": 5000}),
        encoding="utf-8",
    )

    bundle = build_checkpoint_model_bundle(
        tmp_path,
        seed=42,
        checkpoint_episode=5000,
    )

    assert bundle.protocol_id is None


def test_discovery_keeps_multiple_learning_curve_points(tmp_path):
    create_checkpoint_files(tmp_path, seed=42, episode=1000)
    create_checkpoint_files(tmp_path, seed=42, episode=5000)

    bundles = discover_checkpoint_model_bundles(
        tmp_path,
        checkpoint_episodes=[1000, 5000],
        seeds=[42],
    )

    assert [bundle.checkpoint_episode for bundle in bundles] == [1000, 5000]
    assert {bundle.model_source for bundle in bundles} == {"checkpoint"}


def test_discovery_can_fail_on_incomplete_checkpoint(tmp_path):
    create_checkpoint_files(tmp_path, seed=42, episode=1000)
    missing = (
        tmp_path
        / "seed_42"
        / "specialist_calling"
        / "checkpoints"
        / "specialist_calling_episodes_1000_seed_42.pkl"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Incomplete model bundle"):
        discover_checkpoint_model_bundles(
            tmp_path,
            checkpoint_episodes=[1000],
            seeds=[42],
            skip_incomplete=False,
        )


def test_learning_curve_result_row_has_checkpoint_source(tmp_path):
    create_checkpoint_files(tmp_path, seed=42, episode=1000)
    bundle = build_checkpoint_model_bundle(tmp_path, 42, 1000)

    row = build_result_row(
        bundle=bundle,
        tested_agent_name="adaptive_mc",
        opponent_name="calling",
        game_id=0,
        matchup_game_index=0,
        evaluation_seed=123,
        final_stack=200,
        initial_stack=200,
        hands_played=10,
        big_blind=10,
        ended_by_bust=False,
        ended_by_round_limit=True,
        classifier_metrics=empty_classifier_metrics(),
    )

    assert row["model_source"] == "checkpoint"
    assert row["training_episode"] is None
    assert row["checkpoint_episode"] == 1000


def test_learning_curve_evaluator_delegates_to_shared_game_loop(
    tmp_path,
    monkeypatch,
):
    create_checkpoint_files(tmp_path, seed=42, episode=1000)
    bundle = build_checkpoint_model_bundle(tmp_path, 42, 1000)
    captured = {}

    def fake_evaluate(*, bundle, config):
        captured["bundle"] = bundle
        captured["config"] = config
        return [{"ok": True}]

    monkeypatch.setattr(
        "src.evaluation.runners.learning_curve_evaluator."
        "evaluate_training_opponent_bundle",
        fake_evaluate,
    )
    config = LearningCurveEvaluationConfig(
        games_per_matchup=1,
        opponents=("calling",),
        tested_agents=("adaptive_mc",),
        eval_seed_base=100_000,
        output_path=tmp_path / "learning.csv",
    )

    assert evaluate_learning_curve_bundle(bundle=bundle, config=config) == [
        {"ok": True}
    ]
    assert captured["bundle"].model_source == "checkpoint"

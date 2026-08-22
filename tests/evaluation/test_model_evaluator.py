import json
from pathlib import Path

import pytest

from src.evaluation.runners.model_evaluator import (
    ModelBundle,
    TrainingOpponentEvaluationConfig,
    build_final_model_bundle,
    build_game_seed,
    build_result_row,
    discover_final_model_bundles,
    discover_seed_directories,
    evaluate_training_opponent_bundle,
    final_model_path,
    parse_seed_from_directory,
)

POLICIES = (
    "general_policy",
    "specialist_tight",
    "specialist_aggressive",
    "specialist_calling",
)


def create_final_bundle_files(
    root: Path,
    *,
    seed: int,
    completed_episodes: int,
) -> None:
    for directory_name in POLICIES:
        model_path = root / f"seed_{seed}" / directory_name / "final.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"dummy-model")
        model_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "completed_episodes": completed_episodes,
                }
            ),
            encoding="utf-8",
        )


def final_bundle(tmp_path: Path, *, episode: int = 5000) -> ModelBundle:
    return ModelBundle(
        training_run_directory=tmp_path / "run",
        seed=42,
        episode=episode,
        model_source="final",
        unknown_model_path=Path("unknown.pkl"),
        tight_model_path=Path("tight.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )


def classifier_metrics() -> dict:
    return {
        "classified_decisions": 10,
        "correct_classifications": 8,
        "incorrect_classifications": 2,
        "unknown_classifications": 1,
        "other_classifications": 0,
        "classifier_accuracy": 0.8,
        "classifier_coverage": 0.9,
        "policy_switches": 2,
        "first_classification_hand": 3,
        "first_correct_classification_hand": 4,
        "first_classification_action_count": 5,
        "first_correct_classification_action_count": 6,
        "final_predicted_type": "calling",
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }


def test_final_model_path_never_points_at_training_episode(tmp_path):
    path = final_model_path(tmp_path / "seed_42", "calling")

    assert path == tmp_path / "seed_42" / "specialist_calling" / "final.pkl"
    assert "training_episodes" not in path.parts


def test_parse_and_discover_seed_directories(tmp_path):
    (tmp_path / "seed_123").mkdir()
    (tmp_path / "seed_42").mkdir()

    assert parse_seed_from_directory(Path("seed_2026")) == 2026
    assert [path.name for path in discover_seed_directories(tmp_path)] == [
        "seed_42",
        "seed_123",
    ]


def test_parse_seed_rejects_invalid_directory_name():
    with pytest.raises(ValueError, match="Invalid seed directory name"):
        parse_seed_from_directory(Path("model_42"))


def test_build_final_bundle_uses_metadata_training_episode(tmp_path):
    create_final_bundle_files(tmp_path, seed=42, completed_episodes=5000)

    bundle = build_final_model_bundle(tmp_path, seed=42)

    assert bundle.model_source == "final"
    assert bundle.training_episode == 5000
    assert bundle.checkpoint_episode is None
    assert all(path.name == "final.pkl" for path in bundle.agent_paths().values())


def test_final_discovery_returns_one_bundle_per_seed_even_with_training_episodes(
    tmp_path,
):
    create_final_bundle_files(tmp_path, seed=42, completed_episodes=5000)
    training_episode_dir = tmp_path / "seed_42" / "general_policy" / "training_episodes"
    training_episode_dir.mkdir()
    (training_episode_dir / "general_policy_episodes_1000_seed_42.pkl").write_bytes(
        b"training_episode"
    )
    (training_episode_dir / "general_policy_episodes_3000_seed_42.pkl").write_bytes(
        b"training_episode"
    )

    bundles = discover_final_model_bundles(tmp_path)

    assert len(bundles) == 1
    assert bundles[0].training_episode == 5000


def test_final_bundle_rejects_mismatched_metadata_seed(tmp_path):
    create_final_bundle_files(tmp_path, seed=42, completed_episodes=5000)
    metadata_path = tmp_path / "seed_42" / "general_policy" / "final.json"
    metadata_path.write_text(
        json.dumps({"seed": 123, "completed_episodes": 5000}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata seed mismatch"):
        build_final_model_bundle(tmp_path, seed=42)


def test_final_bundle_rejects_mixed_training_budgets(tmp_path):
    create_final_bundle_files(tmp_path, seed=42, completed_episodes=5000)
    metadata_path = tmp_path / "seed_42" / "specialist_calling" / "final.json"
    metadata_path.write_text(
        json.dumps({"seed": 42, "completed_episodes": 4000}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different numbers"):
        build_final_model_bundle(tmp_path, seed=42)


def test_final_discovery_skips_incomplete_seed_by_default(tmp_path):
    create_final_bundle_files(tmp_path, seed=42, completed_episodes=5000)
    (tmp_path / "seed_42" / "specialist_calling" / "final.pkl").unlink()

    assert discover_final_model_bundles(tmp_path) == []


def test_build_game_seed_uses_final_training_episode():
    first = build_game_seed(100_000, 42, 5000, 7)
    second = build_game_seed(100_000, 42, 5000, 7)
    different_episode = build_game_seed(100_000, 42, 5001, 7)

    assert first == second
    assert first != different_episode


def test_final_result_row_has_final_source_and_no_checkpoint(tmp_path):
    row = build_result_row(
        bundle=final_bundle(tmp_path),
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
        classifier_metrics=classifier_metrics(),
    )

    assert row["profit_bb"] == 5.0
    assert row["model_source"] == "final"
    assert row["training_episode"] == 5000
    assert row["checkpoint_episode"] is None


def test_evaluate_final_bundle_restarts_game_index_per_matchup(
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
        "src.evaluation.runners.model_evaluator.evaluate_single_game",
        fake_evaluate_single_game,
    )
    config = TrainingOpponentEvaluationConfig(
        games_per_matchup=2,
        opponents=("calling", "tight"),
        tested_agents=("adaptive_mc", "oracle_mc"),
        eval_seed_base=100_000,
        output_path=tmp_path / "evaluation.csv",
    )

    rows = evaluate_training_opponent_bundle(
        bundle=final_bundle(tmp_path),
        config=config,
    )

    assert len(rows) == 8
    assert [call[2] for call in calls] == list(range(8))
    assert [call[3] for call in calls] == [0, 1] * 4


def test_decision_diagnostics_sum_across_adaptive_policies():
    from src.agents.monte_carlo_agent import MonteCarloAgent
    from src.evaluation.runners.model_evaluator import get_decision_diagnostics
    from src.players.learned.adaptive_player import AdaptivePlayer

    agents = {}
    for policy_type in ("unknown", "tight", "aggressive", "calling"):
        agent = MonteCarloAgent(alpha=0.1, epsilon=0.0, epsilon_min=0.0)
        agent.eval()
        agents[policy_type] = agent

    player = AdaptivePlayer(agents=agents, player_name="adaptive_mc")

    valid_actions = [
        {"action": "fold", "amount": 0},
        {"action": "call", "amount": 10},
    ]
    agents["unknown"].act((1, 1, 1, 0, 0, 0), valid_actions)
    agents["tight"].act((2, 2, 2, 0, 0, 0), valid_actions)

    diagnostics = get_decision_diagnostics(player)

    assert diagnostics["policy_decisions"] == 2
    assert diagnostics["unseen_state_decisions"] == 2
    assert diagnostics["unseen_state_decision_rate"] == 1.0


def test_decision_diagnostics_are_empty_for_scripted_players():
    from src.evaluation.runners.model_evaluator import get_decision_diagnostics
    from src.players.baselines.rule_based_player import RuleBasedPlayer

    diagnostics = get_decision_diagnostics(RuleBasedPlayer())

    assert diagnostics["policy_decisions"] == 0
    assert diagnostics["unseen_state_decision_rate"] == 0.0

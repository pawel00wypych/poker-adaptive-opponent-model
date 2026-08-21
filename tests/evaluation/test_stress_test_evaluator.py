from pathlib import Path

import pytest

from src.evaluation.constants import (
    ADAPTIVE_Q_LEARNING_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.runners.model_evaluator import ModelBundle
from src.evaluation.runners.stress_test_evaluator import (
    DEFAULT_STRESS_TEST_AGENTS,
    DEFAULT_STRESS_TEST_OPPONENTS,
    STRESS_TEST_EVALUATION_FIELDNAMES,
    StressTestEvaluationConfig,
    build_stress_test_seed,
    build_stress_tested_player,
    evaluate_stress_test_bundle,
    stress_test_opponent_registration_name,
    validate_stress_test_agent,
    validate_stress_test_opponent,
)
from src.players.baselines.always_call_player import AlwaysCallPlayer
from src.players.learned.adaptive_player import AdaptivePlayer
from src.players.learned.fixed_policy_player import FixedPolicyPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_TIGHT,
    OPPONENT_TYPE_UNKNOWN,
)


class DummyAgent:
    training = False

    def eval(self):
        self.training = False

    def act(self, state, valid_actions):
        return 1

    def remember(self, state, action_id, valid_actions=None):
        pass

    def learn_from_episode(self, reward):
        pass


def dummy_agents() -> dict[str, DummyAgent]:
    return {
        OPPONENT_TYPE_UNKNOWN: DummyAgent(),
        OPPONENT_TYPE_TIGHT: DummyAgent(),
        OPPONENT_TYPE_AGGRESSIVE: DummyAgent(),
        OPPONENT_TYPE_CALLING: DummyAgent(),
    }


def make_bundle_with_q_paths(tmp_path: Path) -> ModelBundle:
    return ModelBundle(
        training_run_directory=tmp_path / "mc_run",
        seed=42,
        episode=1000,
        model_source="final",
        unknown_model_path=Path("mc_unknown.pkl"),
        tight_model_path=Path("mc_tight.pkl"),
        aggressive_model_path=Path("mc_aggressive.pkl"),
        calling_model_path=Path("mc_calling.pkl"),
        q_learning_training_run_directory=tmp_path / "q_run",
        q_learning_unknown_model_path=Path("q_unknown.pkl"),
        q_learning_tight_model_path=Path("q_tight.pkl"),
        q_learning_aggressive_model_path=Path("q_aggressive.pkl"),
        q_learning_calling_model_path=Path("q_calling.pkl"),
    )


def test_default_stress_test_scope_contains_learned_agents_and_scripted_opponents():
    assert ADAPTIVE_Q_LEARNING_AGENT in DEFAULT_STRESS_TEST_AGENTS
    assert POLICY_GENERAL_Q_LEARNING_AGENT in DEFAULT_STRESS_TEST_AGENTS
    assert DEFAULT_STRESS_TEST_OPPONENTS == (
        ALWAYS_CALL_AGENT,
        ALWAYS_RAISE_AGENT,
        RULE_BASED_AGENT,
    )


def test_validate_stress_test_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unsupported stress-test agent"):
        validate_stress_test_agent("unknown_agent")

    with pytest.raises(ValueError, match="Unsupported stress-test opponent"):
        validate_stress_test_opponent("tight")


def test_build_stress_tested_q_learning_general_policy(tmp_path, monkeypatch):
    loaded_paths = []

    def fake_load_q_learning_eval_agent(path):
        loaded_paths.append(path)
        return DummyAgent()

    monkeypatch.setattr(
        "src.evaluation.runners.stress_test_evaluator.load_q_learning_eval_agent",
        fake_load_q_learning_eval_agent,
    )

    player = build_stress_tested_player(
        tested_agent_name=POLICY_GENERAL_Q_LEARNING_AGENT,
        bundle=make_bundle_with_q_paths(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_UNKNOWN
    assert player.player_name == POLICY_GENERAL_Q_LEARNING_AGENT
    assert loaded_paths == [Path("q_unknown.pkl")]


def test_build_stress_tested_q_learning_adaptive_has_no_expected_type(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.runners.stress_test_evaluator.load_q_learning_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_stress_tested_player(
        tested_agent_name=ADAPTIVE_Q_LEARNING_AGENT,
        bundle=make_bundle_with_q_paths(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.player_name == ADAPTIVE_Q_LEARNING_AGENT
    assert player.expected_opponent_type is None


def test_build_stress_tested_scripted_baseline(tmp_path):
    player = build_stress_tested_player(
        tested_agent_name=ALWAYS_CALL_AGENT,
        bundle=make_bundle_with_q_paths(tmp_path),
    )

    assert isinstance(player, AlwaysCallPlayer)
    assert player.player_name == ALWAYS_CALL_AGENT


def test_duplicate_agent_and_opponent_names_get_distinct_registration_name():
    assert (
        stress_test_opponent_registration_name(
            tested_agent_name=ALWAYS_CALL_AGENT,
            opponent_name=ALWAYS_CALL_AGENT,
        )
        == "always_call_opponent"
    )

    assert (
        stress_test_opponent_registration_name(
            tested_agent_name=ADAPTIVE_Q_LEARNING_AGENT,
            opponent_name=ALWAYS_CALL_AGENT,
        )
        == ALWAYS_CALL_AGENT
    )


def test_build_stress_test_seed_is_deterministic_and_distinct():
    first = build_stress_test_seed(
        eval_seed_base=600_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=0,
    )
    repeated = build_stress_test_seed(
        eval_seed_base=600_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=0,
    )
    second = build_stress_test_seed(
        eval_seed_base=600_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=1,
    )

    assert first == repeated
    assert second != first


def test_evaluate_stress_test_bundle_runs_all_matchups(tmp_path, monkeypatch):
    calls = []

    def fake_evaluate_single_stress_test_game(
        *,
        bundle,
        tested_agent_name,
        opponent_name,
        game_id,
        matchup_game_index,
        game_config,
        eval_seed_base,
    ):
        calls.append(
            (
                tested_agent_name,
                opponent_name,
                game_id,
                matchup_game_index,
            )
        )
        return {
            "training_run": bundle.training_run_directory.name,
            "model_seed": bundle.seed,
            "model_source": "final",
            "training_episode": bundle.training_episode,
            "experiment_id": bundle.experiment_id,
            "experiment_name": f"{tested_agent_name}_vs_{opponent_name}",
            "game_id": game_id,
            "matchup_game_index": matchup_game_index,
            "evaluation_seed": 123_456,
            "agent_name": tested_agent_name,
            "opponent_name": opponent_name,
            "final_stack": 110,
            "initial_stack": 100,
            "profit": 10,
            "profit_bb": 1.0,
            "hands_played": 5,
            "won_game": 1,
            "busted": 0,
            "ended_by_bust": 0,
            "ended_by_round_limit": 1,
            "classified_decisions": 0,
            "correct_classifications": 0,
            "incorrect_classifications": 0,
            "unknown_classifications": 0,
            "classifier_accuracy": 0.0,
            "classifier_coverage": 0.0,
            "policy_switches": 0,
            "first_classification_hand": None,
            "first_correct_classification_hand": None,
            "first_classification_action_count": None,
            "first_correct_classification_action_count": None,
            "final_predicted_type": "",
            "evaluation_type": "stress_test",
            "stress_opponent_type": opponent_name,
        }

    monkeypatch.setattr(
        "src.evaluation.runners.stress_test_evaluator.evaluate_single_stress_test_game",
        fake_evaluate_single_stress_test_game,
    )

    bundle = make_bundle_with_q_paths(tmp_path)
    config = StressTestEvaluationConfig(
        games_per_matchup=2,
        opponents=(ALWAYS_CALL_AGENT, RULE_BASED_AGENT),
        tested_agents=(ADAPTIVE_Q_LEARNING_AGENT, POLICY_GENERAL_Q_LEARNING_AGENT),
        eval_seed_base=600_000,
        output_path=tmp_path / "stress.csv",
    )

    rows = evaluate_stress_test_bundle(
        bundle=bundle,
        config=config,
    )

    assert len(rows) == 8
    assert len(calls) == 8
    assert [call[2] for call in calls] == list(range(8))
    assert [call[3] for call in calls] == [0, 1] * 4
    assert rows[0]["evaluation_type"] == "stress_test"
    assert "stress_opponent_type" in STRESS_TEST_EVALUATION_FIELDNAMES

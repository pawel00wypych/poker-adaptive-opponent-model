from pathlib import Path

import pytest

from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
)
from src.evaluation.runners.cross_play_evaluator import (
    ADAPTIVE_CROSS_PLAY_AGENTS,
    CROSS_PLAY_EVALUATION_FIELDNAMES,
    DEFAULT_CROSS_PLAY_AGENTS,
    DEFAULT_CROSS_PLAY_OPPONENT_AGENTS,
    POLICY_GENERAL_CROSS_PLAY_AGENTS,
    CrossPlayEvaluationConfig,
    build_cross_play_player,
    build_cross_play_seed,
    cross_play_agent_category,
    cross_play_matchup_type,
    cross_play_opponent_registration_name,
    evaluate_cross_play_bundle,
    should_evaluate_cross_play_matchup,
    validate_cross_play_agent,
)
from src.evaluation.runners.model_evaluator import ModelBundle
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


def test_default_cross_play_scope_is_adaptive_vs_adaptive():
    assert DEFAULT_CROSS_PLAY_AGENTS == ADAPTIVE_CROSS_PLAY_AGENTS
    assert DEFAULT_CROSS_PLAY_OPPONENT_AGENTS == ADAPTIVE_CROSS_PLAY_AGENTS
    assert POLICY_GENERAL_MC_AGENT in POLICY_GENERAL_CROSS_PLAY_AGENTS


def test_cross_play_agent_category_and_matchup_type():
    assert cross_play_agent_category(ADAPTIVE_MC_AGENT) == "adaptive"
    assert cross_play_agent_category(POLICY_GENERAL_MC_AGENT) == "policy_general"

    assert (
        cross_play_matchup_type(
            tested_agent_name=ADAPTIVE_MC_AGENT,
            opponent_agent_name=POLICY_GENERAL_MC_AGENT,
        )
        == "adaptive_vs_policy_general"
    )

    with pytest.raises(ValueError, match="Unsupported cross-play agent"):
        validate_cross_play_agent("rule_based")


def test_self_play_is_skipped_by_default_and_can_be_enabled():
    assert not should_evaluate_cross_play_matchup(
        tested_agent_name=ADAPTIVE_MC_AGENT,
        opponent_agent_name=ADAPTIVE_MC_AGENT,
        include_self_play=False,
    )

    assert should_evaluate_cross_play_matchup(
        tested_agent_name=ADAPTIVE_MC_AGENT,
        opponent_agent_name=ADAPTIVE_MC_AGENT,
        include_self_play=True,
    )


def test_duplicate_cross_play_names_get_distinct_registration_name():
    assert (
        cross_play_opponent_registration_name(
            tested_agent_name=ADAPTIVE_MC_AGENT,
            opponent_agent_name=ADAPTIVE_MC_AGENT,
        )
        == "adaptive_mc_opponent"
    )

    assert (
        cross_play_opponent_registration_name(
            tested_agent_name=ADAPTIVE_MC_AGENT,
            opponent_agent_name=ADAPTIVE_Q_LEARNING_AGENT,
        )
        == ADAPTIVE_Q_LEARNING_AGENT
    )


def test_build_cross_play_q_learning_general_policy(tmp_path, monkeypatch):
    loaded_paths = []

    def fake_load_q_learning_eval_agent(path):
        loaded_paths.append(path)
        return DummyAgent()

    monkeypatch.setattr(
        "src.evaluation.runners.cross_play_evaluator.load_q_learning_eval_agent",
        fake_load_q_learning_eval_agent,
    )

    player = build_cross_play_player(
        agent_name=POLICY_GENERAL_Q_LEARNING_AGENT,
        bundle=make_bundle_with_q_paths(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_UNKNOWN
    assert player.player_name == POLICY_GENERAL_Q_LEARNING_AGENT
    assert loaded_paths == [Path("q_unknown.pkl")]


def test_build_cross_play_q_learning_adaptive_player(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.runners.cross_play_evaluator.load_q_learning_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_cross_play_player(
        agent_name=ADAPTIVE_Q_LEARNING_AGENT,
        bundle=make_bundle_with_q_paths(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.player_name == ADAPTIVE_Q_LEARNING_AGENT
    assert player.expected_opponent_type is None


def test_build_cross_play_seed_is_deterministic_and_distinct():
    first = build_cross_play_seed(
        eval_seed_base=700_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=0,
    )
    repeated = build_cross_play_seed(
        eval_seed_base=700_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=0,
    )
    second = build_cross_play_seed(
        eval_seed_base=700_000,
        model_seed=42,
        model_episode=1000,
        matchup_game_index=1,
    )

    assert first == repeated
    assert second != first


def test_evaluate_cross_play_bundle_runs_directed_matchups(tmp_path, monkeypatch):
    calls = []

    def fake_evaluate_single_cross_play_game(
        *,
        bundle,
        tested_agent_name,
        opponent_agent_name,
        game_id,
        matchup_game_index,
        game_config,
        eval_seed_base,
    ):
        calls.append(
            (
                tested_agent_name,
                opponent_agent_name,
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
            "experiment_name": f"{tested_agent_name}_vs_{opponent_agent_name}",
            "game_id": game_id,
            "matchup_game_index": matchup_game_index,
            "evaluation_seed": 123_456,
            "agent_name": tested_agent_name,
            "opponent_name": opponent_agent_name,
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
            "other_classifications": 0,
            "classifier_accuracy": 0.0,
            "classifier_coverage": 0.0,
            "policy_switches": 0,
            "first_classification_hand": None,
            "first_correct_classification_hand": None,
            "first_classification_action_count": None,
            "first_correct_classification_action_count": None,
            "final_predicted_type": "",
            "evaluation_type": "cross_play",
            "agent_category": "adaptive",
            "opponent_agent_category": "adaptive",
            "cross_play_matchup_type": "adaptive_vs_adaptive",
        }

    monkeypatch.setattr(
        "src.evaluation.runners.cross_play_evaluator.evaluate_single_cross_play_game",
        fake_evaluate_single_cross_play_game,
    )

    bundle = make_bundle_with_q_paths(tmp_path)
    config = CrossPlayEvaluationConfig(
        games_per_matchup=2,
        tested_agents=(ADAPTIVE_MC_AGENT, ADAPTIVE_Q_LEARNING_AGENT),
        opponent_agents=(ADAPTIVE_MC_AGENT, ADAPTIVE_Q_LEARNING_AGENT),
        eval_seed_base=700_000,
        output_path=tmp_path / "cross_play.csv",
        include_self_play=False,
    )

    rows = evaluate_cross_play_bundle(
        bundle=bundle,
        config=config,
    )

    assert len(rows) == 4
    assert len(calls) == 4
    assert all(
        tested_agent != opponent_agent for tested_agent, opponent_agent, _, _ in calls
    )
    assert calls[0] == (
        ADAPTIVE_MC_AGENT,
        ADAPTIVE_Q_LEARNING_AGENT,
        0,
        0,
    )
    assert [call[2] for call in calls] == list(range(4))
    assert [call[3] for call in calls] == [0, 1, 0, 1]
    assert rows[0]["evaluation_type"] == "cross_play"
    assert "cross_play_matchup_type" in CROSS_PLAY_EVALUATION_FIELDNAMES

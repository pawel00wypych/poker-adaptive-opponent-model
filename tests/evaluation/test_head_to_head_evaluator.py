from pathlib import Path

import pytest

from src.evaluation.runners.checkpoint_evaluator import ModelBundle
from src.evaluation.runners.head_to_head_evaluator import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    DEFAULT_HEAD_TO_HEAD_AGENTS,
    DEFAULT_HEAD_TO_HEAD_OPPONENTS,
    POLICY_GENERAL_MC_AGENT,
    RULE_BASED_AGENT,
    HeadToHeadEvaluationConfig,
    build_head_to_head_opponent,
    build_head_to_head_tested_player,
    evaluate_baseline_replicate,
    evaluate_baseline_replicates,
    evaluate_head_to_head_bundle,
    validate_head_to_head_agent,
    validate_head_to_head_opponent,
    write_head_to_head_rows,
)
from src.experiments.evaluation.run_head_to_head_evaluation import parse_args
from src.players.baselines.always_call_player import AlwaysCallPlayer
from src.players.baselines.always_raise_player import AlwaysRaisePlayer
from src.players.baselines.rule_based_player import RuleBasedPlayer
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

    def act(self, state, valid_actions):
        return 1

    def remember(self, state, action_id):
        pass

    def learn_from_episode(self, reward):
        pass


def sample_bundle(tmp_path: Path) -> ModelBundle:
    return ModelBundle(
        training_run_directory=tmp_path / "run",
        seed=42,
        checkpoint_episode=2000,
        unknown_model_path=Path("unknown.pkl"),
        tight_model_path=Path("tight.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )


def dummy_agents() -> dict[str, DummyAgent]:
    return {
        OPPONENT_TYPE_UNKNOWN: DummyAgent(),
        OPPONENT_TYPE_TIGHT: DummyAgent(),
        OPPONENT_TYPE_AGGRESSIVE: DummyAgent(),
        OPPONENT_TYPE_CALLING: DummyAgent(),
    }


def test_build_head_to_head_opponent_builds_rule_based():
    player = build_head_to_head_opponent(
        RULE_BASED_AGENT
    )

    assert isinstance(player, RuleBasedPlayer)
    assert player.player_name == RULE_BASED_AGENT


def test_build_head_to_head_opponent_builds_always_raise():
    player = build_head_to_head_opponent(
        ALWAYS_RAISE_AGENT
    )

    assert isinstance(player, AlwaysRaisePlayer)
    assert player.player_name == ALWAYS_RAISE_AGENT



def test_build_head_to_head_opponent_builds_always_call():
    player = build_head_to_head_opponent(
        ALWAYS_CALL_AGENT
    )

    assert isinstance(player, AlwaysCallPlayer)
    assert player.player_name == ALWAYS_CALL_AGENT

def test_validate_head_to_head_opponent_rejects_training_opponent():
    with pytest.raises(ValueError):
        validate_head_to_head_opponent("tight")


def test_validate_head_to_head_agent_rejects_oracle_for_ood_baselines():
    with pytest.raises(ValueError):
        validate_head_to_head_agent("oracle_mc")


def test_build_head_to_head_adaptive_uses_no_expected_type(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.runners.head_to_head_evaluator.load_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_head_to_head_tested_player(
        tested_agent_name=ADAPTIVE_MC_AGENT,
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.expected_opponent_type is None


def test_build_head_to_head_fixed_general_policy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.runners.head_to_head_evaluator.load_eval_agent",
        lambda path: DummyAgent(),
    )

    player = build_head_to_head_tested_player(
        tested_agent_name=POLICY_GENERAL_MC_AGENT,
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_UNKNOWN
    assert player.player_name == POLICY_GENERAL_MC_AGENT



def test_build_head_to_head_always_call_tested_agent_needs_no_model():
    player = build_head_to_head_tested_player(
        tested_agent_name=ALWAYS_CALL_AGENT,
        bundle=None,
    )

    assert isinstance(player, AlwaysCallPlayer)
    assert player.player_name == ALWAYS_CALL_AGENT


def test_build_head_to_head_always_raise_tested_agent_needs_no_model():
    player = build_head_to_head_tested_player(
        tested_agent_name=ALWAYS_RAISE_AGENT,
        bundle=None,
    )

    assert isinstance(player, AlwaysRaisePlayer)
    assert player.player_name == ALWAYS_RAISE_AGENT

def test_evaluate_head_to_head_bundle_runs_only_learned_matchups(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_evaluate_single_head_to_head_game(**kwargs):
        calls.append(
            (
                kwargs["tested_agent_name"],
                kwargs["opponent_name"],
                kwargs["game_id"],
                kwargs["matchup_game_index"],
            )
        )
        return {
            "training_run": "run",
            "model_seed": 42,
            "checkpoint_episode": 2000,
            "experiment_id": "seed_42_episodes_2000",
            "experiment_name": (
                f"{kwargs['tested_agent_name']}_vs_{kwargs['opponent_name']}"
            ),
            "game_id": kwargs["game_id"],
            "matchup_game_index": kwargs["matchup_game_index"],
            "evaluation_seed": 123_456,
            "agent_name": kwargs["tested_agent_name"],
            "opponent_name": kwargs["opponent_name"],
            "final_stack": 120,
            "initial_stack": 100,
            "profit": 20,
            "profit_bb": 2,
            "hands_played": 10,
            "won_game": 1,
            "busted": 0,
            "ended_by_bust": 1,
            "ended_by_round_limit": 0,
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
        }

    monkeypatch.setattr(
        "src.evaluation.runners.head_to_head_evaluator.evaluate_single_head_to_head_game",
        fake_evaluate_single_head_to_head_game,
    )

    config = HeadToHeadEvaluationConfig(
        games_per_matchup=2,
        opponents=(ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT),
        tested_agents=(
            POLICY_GENERAL_MC_AGENT,
            ALWAYS_CALL_AGENT,
            ALWAYS_RAISE_AGENT,
        ),
        eval_seed_base=300_000,
        output_path=tmp_path / "h2h.csv",
    )

    rows = evaluate_head_to_head_bundle(
        bundle=sample_bundle(tmp_path),
        config=config,
    )

    assert len(rows) == 4
    assert len(calls) == 4
    assert all(call[0] == POLICY_GENERAL_MC_AGENT for call in calls)
    assert rows[0]["experiment_name"] == "policy_general_mc_vs_always_call"
    assert rows[-1]["experiment_name"] == "policy_general_mc_vs_always_raise"
    assert [row["game_id"] for row in rows] == list(range(4))
    assert [call[3] for call in calls] == [0, 1] * 2


def test_evaluate_baseline_replicate_runs_without_model_bundle(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_evaluate_single_baseline_game(**kwargs):
        calls.append(kwargs)
        return {
            "agent_name": kwargs["tested_agent_name"],
            "opponent_name": kwargs["opponent_name"],
            "evaluation_replicate_id": kwargs["evaluation_replicate_id"],
            "model_seed": None,
            "checkpoint_episode": None,
            "game_id": kwargs["game_id"],
        }

    monkeypatch.setattr(
        "src.evaluation.runners.head_to_head_evaluator."
        "evaluate_single_baseline_game",
        fake_evaluate_single_baseline_game,
    )
    config = HeadToHeadEvaluationConfig(
        games_per_matchup=2,
        opponents=(ALWAYS_CALL_AGENT, ALWAYS_RAISE_AGENT),
        tested_agents=(
            POLICY_GENERAL_MC_AGENT,
            ALWAYS_CALL_AGENT,
            RULE_BASED_AGENT,
        ),
        eval_seed_base=300_000,
        output_path=tmp_path / "baseline.csv",
        evaluation_replicates=4,
    )

    rows = evaluate_baseline_replicate(
        evaluation_replicate_id=3,
        config=config,
    )

    assert len(rows) == 8
    assert {row["agent_name"] for row in rows} == {
        ALWAYS_CALL_AGENT,
        RULE_BASED_AGENT,
    }
    assert all(row["evaluation_replicate_id"] == 3 for row in rows)
    assert all(row["model_seed"] is None for row in rows)
    assert [call["matchup_game_index"] for call in calls] == [0, 1] * 4


def test_evaluate_baseline_replicates_uses_configured_replicate_count(
    tmp_path,
    monkeypatch,
):
    replicate_ids = []

    def fake_evaluate_baseline_replicate(**kwargs):
        replicate_ids.append(kwargs["evaluation_replicate_id"])
        return [{"evaluation_replicate_id": kwargs["evaluation_replicate_id"]}]

    monkeypatch.setattr(
        "src.evaluation.runners.head_to_head_evaluator."
        "evaluate_baseline_replicate",
        fake_evaluate_baseline_replicate,
    )
    config = HeadToHeadEvaluationConfig(
        games_per_matchup=1,
        opponents=(ALWAYS_CALL_AGENT,),
        tested_agents=(ALWAYS_CALL_AGENT,),
        eval_seed_base=300_000,
        output_path=tmp_path / "baseline.csv",
        evaluation_replicates=3,
    )

    rows = evaluate_baseline_replicates(config=config)

    assert replicate_ids == [0, 1, 2]
    assert [row["evaluation_replicate_id"] for row in rows] == [0, 1, 2]


def test_write_head_to_head_rows_creates_csv(tmp_path):
    output_path = tmp_path / "head_to_head.csv"

    write_head_to_head_rows(
        output_path=output_path,
        rows=[
            {
                "training_run": "run",
                "model_seed": 42,
                "checkpoint_episode": 2000,
                "experiment_id": "seed_42_episodes_2000",
                "experiment_name": "policy_general_mc_vs_rule_based",
                "game_id": 0,
                "agent_name": POLICY_GENERAL_MC_AGENT,
                "opponent_name": RULE_BASED_AGENT,
                "final_stack": 120,
                "initial_stack": 100,
                "profit": 20,
                "profit_bb": 2,
                "hands_played": 10,
                "won_game": 1,
                "busted": 0,
                "ended_by_bust": 1,
                "ended_by_round_limit": 0,
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
            }
        ],
    )

    text = output_path.read_text(
        encoding="utf-8"
    )

    assert "policy_general_mc_vs_rule_based" in text
    assert "opponent_name" in text


def test_parse_args_uses_expected_defaults():
    args = parse_args(
        [
            "--training-run-dir",
            "results/training_runs/run",
            "--checkpoint-episodes",
            "2000",
        ]
    )

    assert args.training_run_dir == "results/training_runs/run"
    assert args.checkpoint_episodes == [2000]
    assert args.agents == list(DEFAULT_HEAD_TO_HEAD_AGENTS)
    assert args.opponents == list(DEFAULT_HEAD_TO_HEAD_OPPONENTS)


def test_parse_args_accepts_custom_head_to_head_matchups():
    args = parse_args(
        [
            "--agents",
            "always_call",
            "rule_based",
            "--opponents",
            "always_raise",
            "always_call",
            "--games",
            "50",
            "--evaluation-replicates",
            "7",
        ]
    )

    assert args.agents == ["always_call", "rule_based"]
    assert args.opponents == ["always_raise", "always_call"]
    assert args.games == 50
    assert args.evaluation_replicates == 7
    assert args.training_run_dir is None
    assert args.checkpoint_episodes is None


def test_parse_args_rejects_model_seed_for_baseline_only_run():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--agents",
                "always_call",
                "rule_based",
                "--seeds",
                "42",
            ]
        )

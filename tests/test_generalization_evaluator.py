from pathlib import Path

import pytest

from src.evaluation.checkpoint_evaluator import ModelBundle
from src.evaluation.generalization_evaluator import (
    ADAPTIVE_MC_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    DEFAULT_GENERALIZATION_AGENTS,
    DEFAULT_GENERALIZATION_OPPONENTS,
    ORACLE_ADAPTIVE_AGENT,
    POLICY_CALLING_AGENT,
    POLICY_UNKNOWN_AGENT,
    RULE_BASED_AGENT,
    GeneralizationEvaluationConfig,
    add_generalization_metadata,
    build_generalization_opponent,
    build_generalization_tested_player,
    evaluate_generalization_bundle,
    validate_generalization_agent,
    validate_generalization_opponent,
    write_generalization_rows,
)
from src.experiments.run_generalization_evaluation import parse_args
from src.players.adaptive_player import AdaptivePlayer
from src.players.always_call_player import AlwaysCallPlayer
from src.players.always_raise_player import AlwaysRaisePlayer
from src.players.fixed_policy_player import FixedPolicyPlayer
from src.players.aggressive_variant_player import AggressiveExtremePlayer
from src.players.calling_player import CallingPlayer
from src.players.strong_calling_player import StrongCallingPlayer
from src.players.oracle_adaptive_player import OracleAdaptivePlayer
from src.players.rule_based_player import RuleBasedPlayer
from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
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
        fish_model_path=Path("fish.pkl"),
        aggressive_model_path=Path("aggressive.pkl"),
        calling_model_path=Path("calling.pkl"),
    )


def dummy_agents() -> dict[str, DummyAgent]:
    return {
        OPPONENT_TYPE_UNKNOWN: DummyAgent(),
        OPPONENT_TYPE_FISH: DummyAgent(),
        OPPONENT_TYPE_AGGRESSIVE: DummyAgent(),
        OPPONENT_TYPE_CALLING: DummyAgent(),
    }


def sample_raw_row() -> dict:
    return {
        "training_run": "run",
        "model_seed": 42,
        "checkpoint_episode": 2000,
        "experiment_id": "seed_42_episodes_2000",
        "experiment_name": "adaptive_mc_vs_strong_calling",
        "game_id": 0,
        "agent_name": ADAPTIVE_MC_AGENT,
        "opponent_name": "strong_calling",
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


def test_build_generalization_opponent_builds_strong_calling():
    player = build_generalization_opponent(
        "strong_calling"
    )

    assert isinstance(player, StrongCallingPlayer)
    assert player.player_name == "strong_calling"


def test_build_generalization_opponent_builds_aggressive_extreme():
    player = build_generalization_opponent(
        "aggressive_extreme"
    )

    assert isinstance(player, AggressiveExtremePlayer)
    assert player.player_name == "aggressive_extreme"


def test_validate_generalization_opponent_accepts_base_calling_reference():
    validate_generalization_opponent("calling")


def test_validate_generalization_agent_rejects_single_policy_alias():
    with pytest.raises(ValueError):
        validate_generalization_agent("single_policy_mc")


def test_build_generalization_adaptive_uses_base_variant_family(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.generalization_evaluator.load_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_generalization_tested_player(
        tested_agent_name=ADAPTIVE_MC_AGENT,
        opponent_name="strong_calling",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.expected_opponent_type == OPPONENT_TYPE_CALLING


def test_build_generalization_oracle_uses_base_variant_family(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.generalization_evaluator.load_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_generalization_tested_player(
        tested_agent_name=ORACLE_ADAPTIVE_AGENT,
        opponent_name="aggressive_light",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, OracleAdaptivePlayer)
    assert player.oracle_opponent_type == OPPONENT_TYPE_AGGRESSIVE


def test_build_generalization_fixed_calling_specialist_does_not_use_variant_policy(
    tmp_path,
    monkeypatch,
):
    loaded_paths = []

    def fake_load_eval_agent(path):
        loaded_paths.append(path)
        return DummyAgent()

    monkeypatch.setattr(
        "src.evaluation.generalization_evaluator.load_eval_agent",
        fake_load_eval_agent,
    )

    player = build_generalization_tested_player(
        tested_agent_name=POLICY_CALLING_AGENT,
        opponent_name="strong_calling",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_CALLING
    assert loaded_paths == [Path("calling.pkl")]


def test_build_generalization_rule_based_baseline(tmp_path):
    player = build_generalization_tested_player(
        tested_agent_name=RULE_BASED_AGENT,
        opponent_name="strong_calling",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, RuleBasedPlayer)
    assert player.player_name == RULE_BASED_AGENT


def test_build_generalization_always_raise_baseline(tmp_path):
    player = build_generalization_tested_player(
        tested_agent_name=ALWAYS_RAISE_AGENT,
        opponent_name="aggressive_extreme",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, AlwaysRaisePlayer)
    assert player.player_name == ALWAYS_RAISE_AGENT




def test_build_generalization_always_call_baseline(tmp_path):
    player = build_generalization_tested_player(
        tested_agent_name=ALWAYS_CALL_AGENT,
        opponent_name="strong_calling",
        bundle=sample_bundle(tmp_path),
    )

    assert isinstance(player, AlwaysCallPlayer)
    assert player.player_name == ALWAYS_CALL_AGENT


def test_add_generalization_metadata_marks_variant_as_unseen():
    row = add_generalization_metadata(
        sample_raw_row(),
        opponent_name="strong_calling",
    )

    assert row["evaluation_type"] == "generalization"
    assert row["trained_on"] == "base_opponents"
    assert row["seen_during_training"] == 0
    assert row["opponent_family"] == OPPONENT_TYPE_CALLING
    assert row["opponent_variant"] == "strong_calling"


def test_evaluate_generalization_bundle_runs_all_matchups(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_evaluate_single_generalization_game(**kwargs):
        calls.append(
            (
                kwargs["tested_agent_name"],
                kwargs["opponent_name"],
                kwargs["game_id"],
            )
        )

        row = sample_raw_row()
        row.update(
            {
                "experiment_name": (
                    f"{kwargs['tested_agent_name']}_vs_{kwargs['opponent_name']}"
                ),
                "agent_name": kwargs["tested_agent_name"],
                "opponent_name": kwargs["opponent_name"],
                "game_id": kwargs["game_id"],
            }
        )

        return add_generalization_metadata(
            row,
            opponent_name=kwargs["opponent_name"],
        )

    monkeypatch.setattr(
        "src.evaluation.generalization_evaluator.evaluate_single_generalization_game",
        fake_evaluate_single_generalization_game,
    )

    config = GeneralizationEvaluationConfig(
        games_per_matchup=2,
        opponents=("strong_calling", "aggressive_extreme"),
        tested_agents=(POLICY_UNKNOWN_AGENT, ADAPTIVE_MC_AGENT),
        eval_seed_base=400_000,
        output_path=tmp_path / "generalization.csv",
    )

    rows = evaluate_generalization_bundle(
        bundle=sample_bundle(tmp_path),
        config=config,
    )

    assert len(rows) == 8
    assert len(calls) == 8
    assert rows[0]["experiment_name"] == "policy_unknown_vs_strong_calling"
    assert rows[-1]["experiment_name"] == "adaptive_mc_vs_aggressive_extreme"
    assert [row["game_id"] for row in rows] == list(range(8))
    assert rows[0]["evaluation_type"] == "generalization"


def test_write_generalization_rows_creates_csv_with_metadata(tmp_path):
    output_path = tmp_path / "generalization.csv"
    row = add_generalization_metadata(
        sample_raw_row(),
        opponent_name="strong_calling",
    )

    write_generalization_rows(
        output_path=output_path,
        rows=[row],
    )

    text = output_path.read_text(
        encoding="utf-8"
    )

    assert "adaptive_mc_vs_strong_calling" in text
    assert "evaluation_type" in text
    assert "opponent_family" in text
    assert "base_opponents" in text


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
    assert args.agents == list(DEFAULT_GENERALIZATION_AGENTS)
    assert args.opponents == list(DEFAULT_GENERALIZATION_OPPONENTS)
    assert ORACLE_ADAPTIVE_AGENT in args.agents


def test_parse_args_accepts_custom_generalization_matchups():
    args = parse_args(
        [
            "--training-run-dir",
            "results/training_runs/run",
            "--checkpoint-episodes",
            "2000",
            "--agents",
            "policy_unknown",
            "adaptive_mc",
            "oracle_adaptive",
            "--opponents",
            "strong_calling",
            "aggressive_extreme",
            "--games",
            "50",
        ]
    )

    assert args.agents == [
        "policy_unknown",
        "adaptive_mc",
        "oracle_adaptive",
    ]
    assert args.opponents == [
        "strong_calling",
        "aggressive_extreme",
    ]
    assert args.games == 50

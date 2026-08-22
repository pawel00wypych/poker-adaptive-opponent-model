from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    SUPPORTED_TESTED_AGENTS,
)
from src.evaluation.reporting.experiment_summary import build_experiment_summary
from src.evaluation.reporting.training_opponent_report import display_agent_name
from src.evaluation.runners.generalization_evaluator import (
    SUPPORTED_GENERALIZATION_AGENTS,
    build_generalization_tested_player,
)
from src.evaluation.runners.learning_curve_evaluator import (
    build_checkpoint_model_bundle,
)
from src.evaluation.runners.model_evaluator import (
    ModelBundle,
    build_tested_player,
)
from src.experiments.evaluation.run_generalization_evaluation import (
    parse_args as parse_generalization_args,
)
from src.experiments.evaluation.run_training_opponent_evaluation import (
    parse_args as parse_final_args,
)
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


def create_checkpoint_bundle_files(
    root: Path,
    seed: int,
    episode: int,
) -> None:
    files = [
        ("general_policy", "general_policy"),
        ("specialist_tight", "specialist_tight"),
        ("specialist_aggressive", "specialist_aggressive"),
        ("specialist_calling", "specialist_calling"),
    ]

    for directory_name, prefix in files:
        checkpoint_directory = root / f"seed_{seed}" / directory_name / "checkpoints"
        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        checkpoint_path = (
            checkpoint_directory / f"{prefix}_episodes_{episode}_seed_{seed}.pkl"
        )
        checkpoint_path.write_bytes(b"dummy-model")


def make_bundle_with_double_q_learning_paths(tmp_path: Path) -> ModelBundle:
    return ModelBundle(
        training_run_directory=tmp_path / "mc_run",
        seed=42,
        episode=2000,
        model_source="final",
        unknown_model_path=Path("mc_unknown.pkl"),
        tight_model_path=Path("mc_tight.pkl"),
        aggressive_model_path=Path("mc_aggressive.pkl"),
        calling_model_path=Path("mc_calling.pkl"),
        double_q_learning_training_run_directory=tmp_path / "double_q_learning_run",
        double_q_learning_unknown_model_path=Path("double_q_unknown.pkl"),
        double_q_learning_tight_model_path=Path("double_q_tight.pkl"),
        double_q_learning_aggressive_model_path=Path("double_q_aggressive.pkl"),
        double_q_learning_calling_model_path=Path("double_q_calling.pkl"),
    )


def dummy_agents() -> dict[str, DummyAgent]:
    return {
        OPPONENT_TYPE_UNKNOWN: DummyAgent(),
        OPPONENT_TYPE_TIGHT: DummyAgent(),
        OPPONENT_TYPE_AGGRESSIVE: DummyAgent(),
        OPPONENT_TYPE_CALLING: DummyAgent(),
    }


def test_build_checkpoint_model_bundle_includes_double_q_learning_paths(tmp_path):
    create_checkpoint_bundle_files(
        root=tmp_path / "mc_run",
        seed=42,
        episode=2000,
    )
    create_checkpoint_bundle_files(
        root=tmp_path / "double_q_learning_run",
        seed=42,
        episode=2000,
    )

    bundle = build_checkpoint_model_bundle(
        training_run_directory=tmp_path / "mc_run",
        double_q_learning_run_directory=tmp_path / "double_q_learning_run",
        seed=42,
        checkpoint_episode=2000,
    )

    assert bundle.has_double_q_learning_models()
    assert (
        bundle.double_q_learning_training_run_directory
        == tmp_path / "double_q_learning_run"
    )
    assert bundle.double_q_learning_agent_paths()[OPPONENT_TYPE_UNKNOWN].name == (
        "general_policy_episodes_2000_seed_42.pkl"
    )


def test_double_q_learning_agent_names_are_supported():
    assert ADAPTIVE_DOUBLE_Q_LEARNING_AGENT in SUPPORTED_TESTED_AGENTS
    assert POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT in SUPPORTED_TESTED_AGENTS
    assert ADAPTIVE_DOUBLE_Q_LEARNING_AGENT in SUPPORTED_GENERALIZATION_AGENTS
    assert POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT in SUPPORTED_GENERALIZATION_AGENTS


def test_build_double_q_learning_general_policy_player(tmp_path, monkeypatch):
    loaded_paths = []

    def fake_load_double_q_learning_eval_agent(path):
        loaded_paths.append(path)
        return DummyAgent()

    monkeypatch.setattr(
        "src.evaluation.runners.model_evaluator.load_double_q_learning_eval_agent",
        fake_load_double_q_learning_eval_agent,
    )

    player = build_tested_player(
        tested_agent_name=POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
        opponent_name="calling",
        bundle=make_bundle_with_double_q_learning_paths(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_UNKNOWN
    assert player.player_name == POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT
    assert loaded_paths == [Path("double_q_unknown.pkl")]


def test_build_double_q_learning_adaptive_player(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.runners.model_evaluator.load_double_q_learning_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_tested_player(
        tested_agent_name=ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
        opponent_name="calling",
        bundle=make_bundle_with_double_q_learning_paths(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.player_name == ADAPTIVE_DOUBLE_Q_LEARNING_AGENT
    assert player.expected_opponent_type == OPPONENT_TYPE_CALLING


def test_double_q_learning_agents_require_double_q_learning_run_dir(tmp_path):
    bundle = ModelBundle(
        training_run_directory=tmp_path / "mc_run",
        seed=42,
        episode=2000,
        model_source="final",
        unknown_model_path=Path("mc_unknown.pkl"),
        tight_model_path=Path("mc_tight.pkl"),
        aggressive_model_path=Path("mc_aggressive.pkl"),
        calling_model_path=Path("mc_calling.pkl"),
    )

    with pytest.raises(
        ValueError,
        match="Pass --double-q-learning-run-dir",
    ):
        build_tested_player(
            tested_agent_name=POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
            opponent_name="calling",
            bundle=bundle,
        )


def test_build_generalization_double_q_learning_adaptive_uses_base_family(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.evaluation.runners.generalization_evaluator.load_double_q_learning_adaptive_agents",
        lambda bundle: dummy_agents(),
    )

    player = build_generalization_tested_player(
        tested_agent_name=ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
        opponent_name="calling_extreme",
        bundle=make_bundle_with_double_q_learning_paths(tmp_path),
    )

    assert isinstance(player, AdaptivePlayer)
    assert player.expected_opponent_type == OPPONENT_TYPE_CALLING


def test_build_generalization_double_q_learning_general_policy(tmp_path, monkeypatch):
    loaded_paths = []

    def fake_load_double_q_learning_eval_agent(path):
        loaded_paths.append(path)
        return DummyAgent()

    monkeypatch.setattr(
        "src.evaluation.runners.generalization_evaluator.load_double_q_learning_eval_agent",
        fake_load_double_q_learning_eval_agent,
    )

    player = build_generalization_tested_player(
        tested_agent_name=POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
        opponent_name="calling_extreme",
        bundle=make_bundle_with_double_q_learning_paths(tmp_path),
    )

    assert isinstance(player, FixedPolicyPlayer)
    assert player.policy_type == OPPONENT_TYPE_UNKNOWN
    assert loaded_paths == [Path("double_q_unknown.pkl")]


def test_final_model_cli_accepts_double_q_learning_run_dir():
    args = parse_final_args(
        [
            "--training-run-dir",
            "results/training_runs/mc_run",
            "--double-q-learning-run-dir",
            "results/training_runs/double_q_learning_2000",
            "--agents",
            "adaptive_mc",
            "adaptive_double_q_learning",
            "policy_general_double_q_learning",
        ]
    )

    assert args.double_q_learning_run_dir == (
        "results/training_runs/double_q_learning_2000"
    )
    assert ADAPTIVE_DOUBLE_Q_LEARNING_AGENT in args.agents
    assert POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT in args.agents


def test_generalization_cli_accepts_double_q_learning_run_dir():
    args = parse_generalization_args(
        [
            "--training-run-dir",
            "results/training_runs/mc_run",
            "--double-q-learning-run-dir",
            "results/training_runs/double_q_learning_2000",
            "--agents",
            "adaptive_mc",
            "adaptive_double_q_learning",
            "policy_general_double_q_learning",
        ]
    )

    assert args.double_q_learning_run_dir == (
        "results/training_runs/double_q_learning_2000"
    )
    assert ADAPTIVE_DOUBLE_Q_LEARNING_AGENT in args.agents
    assert POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT in args.agents


def test_report_labels_include_double_q_learning_agents():
    assert display_agent_name(ADAPTIVE_DOUBLE_Q_LEARNING_AGENT) == (
        "Adaptive Double Q-learning"
    )
    assert display_agent_name(POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT) == (
        "Fixed general Double Q-learning policy"
    )


def make_result_row(agent_name: str, profit_bb: float, seed: int) -> dict:
    return {
        "training_run": "sample_run",
        "model_seed": seed,
        "model_source": "final",
        "training_episode": 2000,
        "checkpoint_episode": None,
        "experiment_id": f"seed_{seed}_episodes_2000",
        "experiment_name": f"{agent_name}_vs_calling",
        "game_id": seed,
        "agent_name": agent_name,
        "opponent_name": "calling",
        "final_stack": 100 + int(profit_bb * 10),
        "initial_stack": 100,
        "profit": int(profit_bb * 10),
        "profit_bb": profit_bb,
        "hands_played": 20,
        "won_game": int(profit_bb > 0),
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
        "first_classification_hand": 0,
        "first_correct_classification_hand": 0,
        "first_classification_action_count": 0,
        "first_correct_classification_action_count": 0,
        "final_predicted_type": "",
        "policy_decisions": 0,
        "unseen_state_decisions": 0,
        "untried_action_selections": 0,
        "unseen_state_decision_rate": 0.0,
        "untried_action_selection_rate": 0.0,
    }


def test_experiment_summary_accepts_double_q_learning_rows(tmp_path):
    rows = []
    for seed in (42, 123):
        rows.append(make_result_row("rule_based", -1.0, seed))
        rows.append(make_result_row("oracle_mc", 10.0, seed))
        rows.append(make_result_row("adaptive_mc", 9.0, seed))
        rows.append(make_result_row(ADAPTIVE_DOUBLE_Q_LEARNING_AGENT, 8.0, seed))
        rows.append(make_result_row(POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT, 4.0, seed))

    input_path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    summary, _ranking, _deltas, _quality_flags = build_experiment_summary(input_path)

    assert ADAPTIVE_DOUBLE_Q_LEARNING_AGENT in summary.overview["agents"]
    assert any(
        row["agent_name"] == ADAPTIVE_DOUBLE_Q_LEARNING_AGENT for row in summary.ranking
    )

from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
)
from src.evaluation.runners.cross_play_evaluator import (
    DEFAULT_CROSS_PLAY_AGENTS,
    DEFAULT_CROSS_PLAY_OPPONENT_AGENTS,
    POLICY_GENERAL_CROSS_PLAY_AGENTS,
    SUPPORTED_CROSS_PLAY_AGENTS,
)
from src.experiments.evaluation.run_cross_play_evaluation import parse_args


def test_cross_play_default_agent_names_are_main_adaptive_agents():
    assert set(DEFAULT_CROSS_PLAY_AGENTS) == {
        ADAPTIVE_MC_AGENT,
        ADAPTIVE_Q_LEARNING_AGENT,
        ADAPTIVE_SARSA_AGENT,
        ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    }
    assert DEFAULT_CROSS_PLAY_OPPONENT_AGENTS == DEFAULT_CROSS_PLAY_AGENTS


def test_cross_play_supports_policy_general_agents_for_optional_comparisons():
    assert set(POLICY_GENERAL_CROSS_PLAY_AGENTS) == {
        POLICY_GENERAL_MC_AGENT,
        POLICY_GENERAL_Q_LEARNING_AGENT,
        POLICY_GENERAL_SARSA_AGENT,
        POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    }

    assert set(POLICY_GENERAL_CROSS_PLAY_AGENTS).issubset(
        SUPPORTED_CROSS_PLAY_AGENTS
    )


def test_parse_cross_play_args_accepts_td_run_directories_and_learned_agents():
    args = parse_args(
        [
            "--training-run-dir",
            "results/training_runs/mc",
            "--q-learning-run-dir",
            "results/training_runs/q_learning",
            "--sarsa-run-dir",
            "results/training_runs/sarsa",
            "--double-q-learning-run-dir",
            "results/training_runs/double_q_learning",
            "--checkpoint-episodes",
            "1000",
            "--seeds",
            "42",
            "123",
            "--games",
            "10",
            "--agents",
            ADAPTIVE_MC_AGENT,
            POLICY_GENERAL_Q_LEARNING_AGENT,
            "--opponent-agents",
            ADAPTIVE_Q_LEARNING_AGENT,
            POLICY_GENERAL_MC_AGENT,
            "--include-self-play",
            "--output-path",
            "results/evaluation/learned_agent_cross_play_1000.csv",
        ]
    )

    assert args.training_run_dir == "results/training_runs/mc"
    assert args.q_learning_run_dir == "results/training_runs/q_learning"
    assert args.sarsa_run_dir == "results/training_runs/sarsa"
    assert args.double_q_learning_run_dir == "results/training_runs/double_q_learning"
    assert args.checkpoint_episodes == [1000]
    assert args.seeds == [42, 123]
    assert args.games == 10
    assert args.agents == [
        ADAPTIVE_MC_AGENT,
        POLICY_GENERAL_Q_LEARNING_AGENT,
    ]
    assert args.opponent_agents == [
        ADAPTIVE_Q_LEARNING_AGENT,
        POLICY_GENERAL_MC_AGENT,
    ]
    assert args.include_self_play

from src.evaluation.constants import (
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ALWAYS_CALL_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    RULE_BASED_AGENT,
)
from src.evaluation.runners.stress_test_evaluator import (
    DEFAULT_STRESS_TEST_AGENTS,
    DEFAULT_STRESS_TEST_OPPONENTS,
    SUPPORTED_STRESS_TEST_AGENTS,
    SUPPORTED_STRESS_TEST_OPPONENTS,
)
from src.experiments.evaluation.run_stress_test_evaluation import parse_args


def test_stress_test_agent_names_cover_final_learned_models():
    assert set(DEFAULT_STRESS_TEST_AGENTS) == {
        ADAPTIVE_MC_AGENT,
        ADAPTIVE_Q_LEARNING_AGENT,
        ADAPTIVE_SARSA_AGENT,
        ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
        POLICY_GENERAL_MC_AGENT,
        POLICY_GENERAL_Q_LEARNING_AGENT,
        POLICY_GENERAL_SARSA_AGENT,
        POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    }

    assert RULE_BASED_AGENT in SUPPORTED_STRESS_TEST_AGENTS


def test_stress_test_opponent_names_are_scripted_sanity_opponents():
    assert DEFAULT_STRESS_TEST_OPPONENTS == (
        ALWAYS_CALL_AGENT,
        "always_raise",
        RULE_BASED_AGENT,
    )
    assert set(DEFAULT_STRESS_TEST_OPPONENTS) == SUPPORTED_STRESS_TEST_OPPONENTS


def test_parse_stress_test_args_accepts_td_run_directories():
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
            ADAPTIVE_Q_LEARNING_AGENT,
            POLICY_GENERAL_Q_LEARNING_AGENT,
            "--opponents",
            ALWAYS_CALL_AGENT,
            RULE_BASED_AGENT,
            "--output-path",
            "results/evaluation/stress_test_learned_models_1000.csv",
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
        ADAPTIVE_Q_LEARNING_AGENT,
        POLICY_GENERAL_Q_LEARNING_AGENT,
    ]
    assert args.opponents == [
        ALWAYS_CALL_AGENT,
        RULE_BASED_AGENT,
    ]

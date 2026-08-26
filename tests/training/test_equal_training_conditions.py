"""Equal training conditions across the four compared algorithms.

The headline claim of the project is a comparison between Monte Carlo,
Q-learning, SARSA and Double Q-learning. It only holds if the algorithms differ
in their update rule and in nothing else: not in how far a reward travels per
hand, not in the learning-rate schedule, and not in how many episodes they get.

Double Q-learning is treated separately where its two-table structure makes a
different outcome correct rather than wrong: each transition updates one table
chosen at random while the other evaluates the target, so propagation is
stochastic and the combined view holds the average of the two tables.
"""

import random
import sys

import pytest

from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.config import TrainingConfig
from src.experiments.training import run_monte_carlo_suite, td_cli
from src.poker.action_mapper import ActionMapper
from src.training.constants import SUPPORTED_ALPHA_MODES
from src.training.q_learning_trainer import Q_LEARNING_TRAINING_SPEC
from src.training.td_trainer import run_td_model_training

SINGLE_TABLE_TD_AGENTS = (QLearningAgent, SarsaAgent)
ALL_AGENT_CLASSES = (
    MonteCarloAgent,
    QLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
)

VALID_ACTIONS = [
    {"action": "fold", "amount": 0},
    {"action": "call", "amount": 10},
    {"action": "raise", "amount": {"min": 20, "max": 200}},
]

TRAJECTORY = [
    (0, 1, 1, 0, 0, 0, 0),
    (1, 2, 2, 1, 1, 1, 0),
    (2, 3, 3, 2, 2, 2, 0),
    (3, 4, 4, 3, 3, 3, 0),
]


def _play_one_hand(agent, reward=10.0):
    agent.train()

    for state in TRAJECTORY:
        agent.remember(state, ActionMapper.CALL, valid_actions=VALID_ACTIONS)

    agent.learn_from_episode(reward=reward)


def _trajectory_values(agent):
    return [
        agent.policy.get_q_value(state, ActionMapper.CALL)
        for state in TRAJECTORY
    ]


def _td_cli_args():
    return td_cli.parse_td_training_args(
        td_cli.TDTrainingCliSpec(
            algorithm_name=Q_LEARNING_TRAINING_SPEC.algorithm_name,
            display_name=Q_LEARNING_TRAINING_SPEC.display_name,
            default_output_dir="results/models/q_learning",
            trainer_function=run_td_model_training,
            model_run_name_function=lambda model_type: model_type,
        )
    )


@pytest.mark.parametrize("agent_class", SINGLE_TABLE_TD_AGENTS)
def test_terminal_reward_propagates_through_the_whole_hand(agent_class):
    """A forward pass moved the reward one step per hand, starving early states."""
    agent = agent_class(alpha=0.5, gamma=1.0, epsilon=0.0)

    _play_one_hand(agent)

    values = _trajectory_values(agent)

    assert all(value > 0.0 for value in values), values


@pytest.mark.parametrize("agent_class", SINGLE_TABLE_TD_AGENTS)
def test_reverse_backup_matches_a_manual_bellman_calculation(agent_class):
    """Exact values, not merely 'non-zero'.

    With alpha = 0.5, gamma = 1.0, zero intermediate rewards and a terminal
    reward of 10, backups applied from the end of the hand give:

        Q(s4) = 0 + 0.5 * (10 - 0)   = 5.0
        Q(s3) = 0 + 0.5 * (5.0 - 0)  = 2.5
        Q(s2) = 0 + 0.5 * (2.5 - 0)  = 1.25
        Q(s1) = 0 + 0.5 * (1.25 - 0) = 0.625

    A forward pass leaves s1, s2 and s3 at zero.
    """
    agent = agent_class(alpha=0.5, gamma=1.0, epsilon=0.0)

    _play_one_hand(agent)

    assert _trajectory_values(agent) == pytest.approx([0.625, 1.25, 2.5, 5.0])


def test_double_q_learning_also_propagates_beyond_the_terminal_state():
    """Propagation is stochastic here, so it is measured over repeated hands.

    Each transition updates one table while the other evaluates the target, so
    a single hand may leave earlier states untouched. That is the algorithm,
    not a defect; what matters is that the reward is no longer confined to the
    final decision.
    """
    random.seed(7)
    agent = DoubleQLearningAgent(alpha=0.5, gamma=1.0, epsilon=0.0)

    for _ in range(10):
        _play_one_hand(agent)

    values = _trajectory_values(agent)

    assert all(value > 0.0 for value in values), values


def test_monte_carlo_assigns_the_full_return_to_every_first_visit():
    """Monte Carlo needs no propagation: the return is known at the end."""
    agent = MonteCarloAgent(alpha=0.5, gamma=1.0, epsilon=0.0)

    _play_one_hand(agent)

    assert _trajectory_values(agent) == pytest.approx([5.0, 5.0, 5.0, 5.0])


def test_monte_carlo_discounts_by_distance_from_the_terminal_reward():
    """gamma must mean the same thing for Monte Carlo as for the TD agents."""
    agent = MonteCarloAgent(alpha=1.0, gamma=0.5, epsilon=0.0)

    _play_one_hand(agent)

    assert _trajectory_values(agent) == pytest.approx([1.25, 2.5, 5.0, 10.0])


@pytest.mark.parametrize("agent_class", ALL_AGENT_CLASSES)
def test_every_algorithm_accepts_gamma(agent_class):
    assert agent_class(gamma=0.9).gamma == 0.9


@pytest.mark.parametrize("agent_class", ALL_AGENT_CLASSES)
@pytest.mark.parametrize("alpha_mode", SUPPORTED_ALPHA_MODES)
def test_every_algorithm_supports_every_learning_rate_schedule(
    agent_class,
    alpha_mode,
):
    assert agent_class(alpha=0.5, alpha_mode=alpha_mode).alpha_mode == alpha_mode


@pytest.mark.parametrize("agent_class", ALL_AGENT_CLASSES)
def test_every_algorithm_rejects_an_unknown_learning_rate_schedule(agent_class):
    with pytest.raises(ValueError, match="Unsupported alpha_mode"):
        agent_class(alpha_mode="cosine_decay")


@pytest.mark.parametrize("agent_class", SINGLE_TABLE_TD_AGENTS + (MonteCarloAgent,))
def test_visit_count_schedule_gives_the_first_update_full_weight(agent_class):
    """1/N(s,a) means the first visit to a pair moves all the way to its target."""
    agent = agent_class(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
        alpha_mode="visit_count",
    )
    agent.train()
    agent.remember(TRAJECTORY[0], ActionMapper.CALL, valid_actions=VALID_ACTIONS)
    agent.learn_from_episode(reward=8.0)

    assert agent.policy.get_q_value(
        TRAJECTORY[0], ActionMapper.CALL
    ) == pytest.approx(8.0)


def test_double_q_visit_count_moves_one_table_fully():
    """The combined view averages the two tables, so it reads half the target."""
    agent = DoubleQLearningAgent(
        alpha=0.5,
        gamma=1.0,
        epsilon=0.0,
        alpha_mode="visit_count",
    )
    agent.train()
    agent.remember(TRAJECTORY[0], ActionMapper.CALL, valid_actions=VALID_ACTIONS)
    agent.learn_from_episode(reward=8.0)

    updated = max(
        agent.q1_table[TRAJECTORY[0]][ActionMapper.CALL],
        agent.q2_table[TRAJECTORY[0]][ActionMapper.CALL],
    )

    assert updated == pytest.approx(8.0)
    assert agent.policy.get_q_value(
        TRAJECTORY[0], ActionMapper.CALL
    ) == pytest.approx(4.0)


@pytest.mark.parametrize("agent_class", ALL_AGENT_CLASSES)
def test_learning_rate_schedule_survives_a_save_and_load(agent_class, tmp_path):
    path = tmp_path / "model.pkl"
    agent_class(alpha=0.25, alpha_mode="sqrt_visit").save(str(path))

    loaded = agent_class.load(str(path))

    assert loaded.alpha_mode == "sqrt_visit"
    assert loaded.alpha == 0.25


def test_all_algorithms_share_the_default_episode_budget(monkeypatch):
    """Defaults used to be 10,000 for Monte Carlo and 2,000 for the TD agents."""
    monkeypatch.setattr(sys, "argv", ["run_monte_carlo_suite"])
    monte_carlo_args = run_monte_carlo_suite.parse_args()

    monkeypatch.setattr(sys, "argv", ["run_q_learning_training"])
    td_args = _td_cli_args()

    assert monte_carlo_args.episodes == TrainingConfig.episodes
    assert td_args.episodes == TrainingConfig.episodes


def test_default_checkpoints_fit_inside_the_default_budget(monkeypatch):
    """A checkpoint past the final episode makes the suite refuse to start."""
    monkeypatch.setattr(sys, "argv", ["run_monte_carlo_suite"])
    args = run_monte_carlo_suite.parse_args()

    assert max(args.checkpoint_episodes) <= args.episodes


def test_td_cli_allows_short_runs_when_checkpoints_are_disabled(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_q_learning_training",
            "--episodes",
            "2",
            "--no-checkpoints",
        ],
    )

    args = _td_cli_args()

    assert args.episodes == 2
    assert args.checkpoints is False


def test_both_training_clis_expose_the_learning_rate_schedule(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_monte_carlo_suite"])
    monte_carlo_args = run_monte_carlo_suite.parse_args()

    monkeypatch.setattr(sys, "argv", ["run_q_learning_training"])
    td_args = _td_cli_args()

    assert monte_carlo_args.alpha_mode == td_args.alpha_mode
    assert monte_carlo_args.alpha_mode == "sqrt_visit"

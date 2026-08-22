import pytest

from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent

AGENT_CLASSES = (
    MonteCarloAgent,
    QLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
)

MATCHING_PAIRS = [
    (agent_class, agent_class) for agent_class in AGENT_CLASSES
]

MISMATCHED_PAIRS = [
    (saving_class, loading_class)
    for saving_class in AGENT_CLASSES
    for loading_class in AGENT_CLASSES
    if saving_class is not loading_class
]


def _agent_ids(pair):
    saving_class, loading_class = pair
    return f"{saving_class.__name__}->{loading_class.__name__}"


def test_every_agent_declares_a_unique_algorithm_id():
    algorithm_ids = [
        agent_class.ALGORITHM_ID for agent_class in AGENT_CLASSES
    ]

    assert len(set(algorithm_ids)) == len(AGENT_CLASSES)


@pytest.mark.parametrize(
    ("saving_class", "loading_class"),
    MATCHING_PAIRS,
    ids=[_agent_ids(pair) for pair in MATCHING_PAIRS],
)
def test_agent_loads_its_own_model(saving_class, loading_class, tmp_path):
    path = tmp_path / "model.pkl"
    saving_class().save(str(path))

    agent = loading_class.load(str(path))

    assert agent.ALGORITHM_ID == saving_class.ALGORITHM_ID


@pytest.mark.parametrize(
    ("saving_class", "loading_class"),
    MISMATCHED_PAIRS,
    ids=[_agent_ids(pair) for pair in MISMATCHED_PAIRS],
)
def test_load_rejects_model_saved_by_another_algorithm(
    saving_class,
    loading_class,
    tmp_path,
):
    """A wrong --*-run-dir must fail loudly instead of mislabelling results."""
    path = tmp_path / "model.pkl"
    saving_class().save(str(path))

    with pytest.raises(ValueError) as error:
        loading_class.load(str(path))

    message = str(error.value)
    assert repr(loading_class.ALGORITHM_ID) in message
    assert repr(saving_class.ALGORITHM_ID) in message

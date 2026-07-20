import pytest

from src.agents.monte_carlo_agent import MonteCarloAgent


@pytest.fixture
def valid_actions() -> list[dict]:
    return [
        {
            "action": "fold",
            "amount": 0,
        },
        {
            "action": "call",
            "amount": 10,
        },
        {
            "action": "raise",
            "amount": {
                "min": 20,
                "max": 200,
            },
        },
    ]


@pytest.fixture
def round_state_factory():
    def build_round_state(
        player_stack: int = 200,
        opponent_stack: int = 200,
        round_count: int = 1,
        community_cards: list[str] | None = None,
    ) -> dict:
        return {
            "round_count": round_count,
            "community_card": community_cards or [],
            "seats": [
                {
                    "name": "tested_player",
                    "uuid": "uuid-tested",
                    "stack": player_stack,
                    "state": "participating",
                },
                {
                    "name": "opponent",
                    "uuid": "uuid-opponent",
                    "stack": opponent_stack,
                    "state": "participating",
                },
            ],
            "pot": {
                "main": {
                    "amount": 15,
                }
            },
        }

    return build_round_state


@pytest.fixture
def eval_agent() -> MonteCarloAgent:
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        epsilon_min=0.0,
    )
    agent.eval()

    return agent


@pytest.fixture
def training_agent() -> MonteCarloAgent:
    agent = MonteCarloAgent(
        alpha=0.1,
        epsilon=0.0,
        epsilon_min=0.0,
    )
    agent.train()

    return agent


@pytest.fixture
def adaptive_agents() -> dict[str, MonteCarloAgent]:
    agents = {}

    for opponent_type in [
        "unknown",
        "fish",
        "aggressive",
        "calling",
    ]:
        agent = MonteCarloAgent(
            alpha=0.1,
            epsilon=0.0,
            epsilon_min=0.0,
        )
        agent.eval()
        agents[opponent_type] = agent

    return agents
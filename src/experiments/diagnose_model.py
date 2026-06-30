from collections import Counter

import numpy as np

from src.agents.q_learning_agent import QLearningAgent
from src.config import TrainingConfig


def diagnose_model() -> None:
    config = TrainingConfig()

    agent = QLearningAgent.load(config.model_path)

    print(f"Loaded model from: {config.model_path}")
    print(f"Number of states: {len(agent.q_table)}")
    print(f"Epsilon: {agent.epsilon:.4f}")

    if not agent.q_table:
        print("Q-table is empty. Agent did not learn any states.")
        return

    non_zero_states = 0
    action_counter = Counter()

    for state, q_values in agent.q_table.items():
        if np.any(q_values != 0):
            non_zero_states += 1

        best_action = int(np.argmax(q_values))
        action_counter[best_action] += 1

    print(f"States with non-zero Q-values: {non_zero_states}")
    print(f"Best action distribution: {dict(action_counter)}")

    print("\nSample states:")
    for idx, (state, q_values) in enumerate(list(agent.q_table.items())[:20]):
        print(f"{idx + 1}. state={state}, q_values={q_values}")

    if non_zero_states == 0:
        print(
            "\nWARNING: All Q-values are zero. "
            "This usually means rewards are always zero or learn_from_episode is not receiving useful data."
        )


if __name__ == "__main__":
    diagnose_model()
from collections import Counter, defaultdict

import numpy as np

from src.agents.monte_carlo_agent import MonteCarloAgent
from src.config import TrainingConfig


ACTIONS = ("fold", "call", "raise")


def main() -> None:
    config = TrainingConfig()

    agent = MonteCarloAgent.load(config.calling_model_path)

    best_actions: Counter[str] = Counter()
    q_values: defaultdict[str, list[float]] = defaultdict(list)

    tied_best_action_states = 0
    fully_uninitialized_states = 0
    best_action_advantages: list[float] = []

    for action_values in agent.q_table.values():
        action_values = np.asarray(
            action_values,
            dtype=float,
        )

        if action_values.size == 0:
            continue

        if action_values.size != len(ACTIONS):
            raise ValueError(
                f"Expected {len(ACTIONS)} action values, "
                f"but received {action_values.size}: {action_values}"
            )

        if np.all(np.isclose(action_values, 0.0)):
            fully_uninitialized_states += 1

        max_value = np.max(action_values)

        best_action_indices = np.flatnonzero(
            np.isclose(
                action_values,
                max_value,
            )
        )

        if len(best_action_indices) > 1:
            tied_best_action_states += 1

        best_action_index = int(np.argmax(action_values))
        best_action = ACTIONS[best_action_index]

        best_actions[best_action] += 1

        sorted_values = np.sort(action_values)
        best_action_advantage = (
            sorted_values[-1] - sorted_values[-2]
        )
        best_action_advantages.append(
            float(best_action_advantage)
        )

        for action, value in zip(
            ACTIONS,
            action_values,
            strict=True,
        ):
            q_values[action].append(float(value))

    total_states = len(agent.q_table)

    print(f"States: {total_states}")
    print()

    print("Best actions:")

    for action in ACTIONS:
        count = best_actions[action]

        percentage = (
            count / total_states * 100
            if total_states > 0
            else 0.0
        )

        print(
            f"{action}: "
            f"count={count}, "
            f"percentage={percentage:.2f}%"
        )

    print()
    print("Q-value statistics:")

    for action in ACTIONS:
        values = np.asarray(
            q_values[action],
            dtype=float,
        )

        if values.size == 0:
            print(f"{action}: no values")
            continue

        zero_count = int(
            np.sum(
                np.isclose(
                    values,
                    0.0,
                )
            )
        )

        zero_rate = zero_count / values.size

        print(
            f"{action}: "
            f"count={values.size}, "
            f"mean={np.mean(values):.4f}, "
            f"median={np.median(values):.4f}, "
            f"std={np.std(values):.4f}, "
            f"min={np.min(values):.4f}, "
            f"max={np.max(values):.4f}, "
            f"zeros={zero_count}, "
            f"zero_rate={zero_rate:.2%}"
        )

    print()
    print("Policy diagnostics:")

    tied_states_rate = (
        tied_best_action_states / total_states
        if total_states > 0
        else 0.0
    )

    uninitialized_states_rate = (
        fully_uninitialized_states / total_states
        if total_states > 0
        else 0.0
    )

    print(
        "States with tied best actions: "
        f"{tied_best_action_states} "
        f"({tied_states_rate:.2%})"
    )

    print(
        "Fully uninitialized states: "
        f"{fully_uninitialized_states} "
        f"({uninitialized_states_rate:.2%})"
    )

    if best_action_advantages:
        advantages = np.asarray(
            best_action_advantages,
            dtype=float,
        )

        print(
            "Best-action advantage: "
            f"mean={np.mean(advantages):.4f}, "
            f"median={np.median(advantages):.4f}, "
            f"std={np.std(advantages):.4f}, "
            f"min={np.min(advantages):.4f}, "
            f"max={np.max(advantages):.4f}"
        )

        small_advantage_count = int(
            np.sum(advantages < 0.1)
        )

        small_advantage_rate = (
            small_advantage_count / advantages.size
        )

        print(
            "States with advantage below 0.1: "
            f"{small_advantage_count} "
            f"({small_advantage_rate:.2%})"
        )


if __name__ == "__main__":
    main()
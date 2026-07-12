from src.evaluation.metrics import calculate_bb_per_100


def show_agent_comparison() -> None:
    summary = calculate_bb_per_100("results/raw/agent_comparison_results.csv")

    summary = summary.sort_values(
        by=["experiment_name", "bb_per_100"],
        ascending=[True, False],
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    show_agent_comparison()
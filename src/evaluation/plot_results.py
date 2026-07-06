from pathlib import Path

import matplotlib.pyplot as plt

from src.evaluation.metrics import calculate_bb_per_100


def plot_bb_per_100(results_csv_path: str, output_path: str) -> None:
    summary = calculate_bb_per_100(results_csv_path)

    summary["label"] = summary["experiment_name"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(summary["label"], summary["bb_per_100"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("bb/100")
    plt.title("Agent performance comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    plot_bb_per_100(
        results_csv_path="results/raw/agent_comparison_results.csv",
        output_path="results/plots/agent_comparison_bb_per_100.png",
    )
from src.config import EvaluationConfig
from src.evaluation.metrics import calculate_bb_per_100


def show_results() -> None:
    eval_config = EvaluationConfig()
    summary = calculate_bb_per_100(eval_config.output_path)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    show_results()
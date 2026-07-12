from src.config import EvaluationConfig
from src.evaluation.classifier_metrics import (
    calculate_classifier_summary,
)


def main() -> None:
    config = EvaluationConfig()

    summary = calculate_classifier_summary(
        config.output_path
    )

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
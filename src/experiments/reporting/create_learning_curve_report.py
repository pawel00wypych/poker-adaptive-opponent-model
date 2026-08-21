import argparse

from src.evaluation.constants import SUPPORTED_TESTED_AGENTS
from src.evaluation.reporting.learning_curve_report import (
    write_learning_curve_html_report,
    write_learning_curve_markdown_report,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create diagnostic learning-curve reports from checkpoints."
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--opponent",
        default=None,
        choices=TRAINING_OPPONENT_TYPES,
    )
    parser.add_argument(
        "--agent",
        default=None,
        choices=sorted(SUPPORTED_TESTED_AGENTS),
    )
    parser.add_argument(
        "--format",
        default="both",
        choices=["html", "markdown", "both"],
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    created_paths = []

    if args.format in {"html", "both"}:
        created_paths.append(
            write_learning_curve_html_report(
                input_path=args.input_path,
                output_dir=args.output_dir,
                opponent=args.opponent,
                agent=args.agent,
            )
        )

    if args.format in {"markdown", "both"}:
        created_paths.append(
            write_learning_curve_markdown_report(
                input_path=args.input_path,
                output_dir=args.output_dir,
                opponent=args.opponent,
                agent=args.agent,
            )
        )

    print("Created report files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

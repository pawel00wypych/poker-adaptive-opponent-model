import argparse

from src.evaluation.reporting.q_table_report import (
    write_q_table_html_report,
    write_q_table_markdown_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create readable Q-table comparison reports with plots."
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--format",
        default="both",
        choices=["html", "markdown", "both"],
    )
    parser.add_argument(
        "--disagreement-limit",
        type=int,
        default=5,
        help="Number of largest disagreements per pair to include.",
    )

    args = parser.parse_args(argv)

    if args.disagreement_limit <= 0:
        parser.error("--disagreement-limit must be greater than zero")

    return args


def main() -> None:
    args = parse_args()
    created_paths = []

    if args.format in {"html", "both"}:
        created_paths.append(
            write_q_table_html_report(
                input_path=args.input_path,
                output_dir=args.output_dir,
                disagreement_limit=args.disagreement_limit,
            )
        )

    if args.format in {"markdown", "both"}:
        created_paths.append(
            write_q_table_markdown_report(
                input_path=args.input_path,
                output_dir=args.output_dir,
                disagreement_limit=args.disagreement_limit,
            )
        )

    print("Created report files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

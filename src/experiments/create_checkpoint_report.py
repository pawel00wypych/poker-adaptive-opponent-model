import argparse

from src.evaluation.checkpoint_report import (
    write_checkpoint_html_report,
    write_checkpoint_markdown_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create readable checkpoint evaluation reports with plots."
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--opponent",
        default=None,
        choices=["fish", "aggressive", "calling"],
    )
    parser.add_argument(
        "--agent",
        default=None,
        choices=[
            "rule_based",
            "single_policy_mc",
            "adaptive_mc",
            "oracle_adaptive",
            "policy_unknown",
            "policy_fish",
            "policy_aggressive",
            "policy_calling",
        ],
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
            write_checkpoint_html_report(
                input_path=args.input_path,
                output_dir=args.output_dir,
                opponent=args.opponent,
                agent=args.agent,
            )
        )

    if args.format in {"markdown", "both"}:
        created_paths.append(
            write_checkpoint_markdown_report(
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

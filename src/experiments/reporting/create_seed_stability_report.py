from __future__ import annotations

import argparse

from src.evaluation.reporting.seed_stability import (
    SeedStabilityConfig,
    write_seed_stability_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated report describing performance and ranking "
            "stability across independent model seeds."
        )
    )
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--min-complete-seeds-for-ranking",
        type=int,
        default=2,
        help=(
            "Minimum number of complete per-seed rankings required to "
            "calculate Kendall's W."
        ),
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["markdown", "json", "both", "all"],
        help=(
            "Report format. CSV exports are always written; "
            "'all' writes Markdown, JSON, CSV, and LaTeX."
        ),
    )
    parser.add_argument(
        "--export-latex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LaTeX table export.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SeedStabilityConfig:
    return SeedStabilityConfig(
        min_complete_seeds_for_ranking=(args.min_complete_seeds_for_ranking),
    )


def main() -> None:
    args = parse_args()
    created_paths = write_seed_stability_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config=build_config(args),
        report_format=args.format,
        export_latex=args.export_latex,
    )

    print("Created seed stability files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

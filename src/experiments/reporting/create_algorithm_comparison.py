from __future__ import annotations

import argparse

from src.evaluation.reporting.algorithm_comparison import (
    AlgorithmComparisonConfig,
    write_algorithm_comparison_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated comparison report for adaptive tabular RL "
            "algorithms: Monte Carlo, Q-learning, SARSA, and Double Q-learning."
        )
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
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
    parser.add_argument(
        "--include-charts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate algorithm comparison charts as PNG files.",
    )
    parser.add_argument(
        "--max-std-across-seeds-bb",
        type=float,
        default=5.0,
        help="Seed-stability warning threshold used in findings and charts.",
    )
    parser.add_argument(
        "--max-chart-label-length",
        type=int,
        default=28,
        help="Maximum label length for chart labels.",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AlgorithmComparisonConfig:
    return AlgorithmComparisonConfig(
        max_std_across_seeds_bb=args.max_std_across_seeds_bb,
        max_label_length=args.max_chart_label_length,
    )


def main() -> None:
    args = parse_args()
    created_paths = write_algorithm_comparison_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config=build_config(args),
        report_format=args.format,
        export_latex=args.export_latex,
        include_charts=args.include_charts,
    )

    print("Created algorithm comparison files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from src.evaluation.experiment_charts import ExperimentChartConfig
from src.evaluation.experiment_summary import (
    SummaryThresholds,
    write_experiment_summary_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an experiment summary with rankings, baseline "
            "deltas, traffic-light statuses, and exportable tables."
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
        "--max-std-across-seeds-bb",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--min-warning-win-rate",
        type=float,
        default=55.0,
    )
    parser.add_argument(
        "--high-always-raise-mean-profit-bb",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--high-always-raise-win-rate",
        type=float,
        default=95.0,
    )
    parser.add_argument(
        "--fish-saturation-win-rate",
        type=float,
        default=95.0,
    )
    parser.add_argument(
        "--fish-saturation-mean-profit-bb",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--include-charts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate experiment summary charts as PNG files.",
    )
    parser.add_argument(
        "--chart-ci-multiplier",
        type=float,
        default=1.96,
        help="Multiplier used for approximate cross-seed confidence intervals.",
    )

    return parser.parse_args(argv)


def build_thresholds(args: argparse.Namespace) -> SummaryThresholds:
    return SummaryThresholds(
        max_std_across_seeds_bb=args.max_std_across_seeds_bb,
        min_warning_win_rate=args.min_warning_win_rate,
        high_always_raise_mean_profit_bb=(
            args.high_always_raise_mean_profit_bb
        ),
        high_always_raise_win_rate=args.high_always_raise_win_rate,
        fish_saturation_win_rate=args.fish_saturation_win_rate,
        fish_saturation_mean_profit_bb=(
            args.fish_saturation_mean_profit_bb
        ),
    )


def build_chart_config(args: argparse.Namespace) -> ExperimentChartConfig:
    return ExperimentChartConfig(
        ci_multiplier=args.chart_ci_multiplier,
        max_std_across_seeds_bb=args.max_std_across_seeds_bb,
    )


def main() -> None:
    args = parse_args()
    created_paths = write_experiment_summary_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        thresholds=build_thresholds(args),
        report_format=args.format,
        export_latex=args.export_latex,
        include_charts=args.include_charts,
        chart_config=build_chart_config(args),
    )

    print("Created experiment summary files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

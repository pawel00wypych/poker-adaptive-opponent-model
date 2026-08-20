from __future__ import annotations

import argparse

from src.evaluation.reporting.classifier_quality import (
    ClassifierQualityConfig,
    write_classifier_quality_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an algorithm-aware classifier quality report with "
            "decision-level unknown rates and final-prediction confusion "
            "matrices."
        )
    )
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--checkpoint-episode",
        type=int,
        default=None,
        help=(
            "Checkpoint to analyse. By default, the latest checkpoint from "
            "each training run is used."
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


def build_config(args: argparse.Namespace) -> ClassifierQualityConfig:
    return ClassifierQualityConfig(
        checkpoint_episode=args.checkpoint_episode,
    )


def main() -> None:
    args = parse_args()
    created_paths = write_classifier_quality_outputs(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config=build_config(args),
        report_format=args.format,
        export_latex=args.export_latex,
    )

    print("Created classifier quality files:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()

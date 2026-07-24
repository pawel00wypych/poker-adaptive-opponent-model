from __future__ import annotations

import argparse

from src.evaluation.experiment_validation import (
    ValidationThresholds,
    validate_checkpoint_results,
    write_validation_json_report,
    write_validation_markdown_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run automated sanity checks on checkpoint "
            "evaluation results."
        )
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--format",
        default="both",
        choices=["markdown", "json", "both"],
    )
    parser.add_argument(
        "--min-adaptive-delta-vs-rule-based-bb",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-oracle-underperformance-bb",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-fish-win-rate",
        type=float,
        default=95.0,
    )
    parser.add_argument(
        "--min-fish-mean-profit-bb",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--min-classifier-accuracy",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--min-classifier-coverage",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--max-std-across-seeds-bb",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--extreme-bb-per-100-threshold",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--low-mean-hands-played-threshold",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--always-raise-adaptive-warning-gap-bb",
        type=float,
        default=3.0,
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

    return parser.parse_args(argv)


def build_thresholds(args: argparse.Namespace) -> ValidationThresholds:
    return ValidationThresholds(
        min_adaptive_delta_vs_rule_based_bb=(
            args.min_adaptive_delta_vs_rule_based_bb
        ),
        max_oracle_underperformance_bb=(
            args.max_oracle_underperformance_bb
        ),
        min_fish_win_rate=args.min_fish_win_rate,
        min_fish_mean_profit_bb=args.min_fish_mean_profit_bb,
        min_classifier_accuracy=args.min_classifier_accuracy,
        min_classifier_coverage=args.min_classifier_coverage,
        max_std_across_seeds_bb=args.max_std_across_seeds_bb,
        extreme_bb_per_100_threshold=(
            args.extreme_bb_per_100_threshold
        ),
        low_mean_hands_played_threshold=(
            args.low_mean_hands_played_threshold
        ),
        always_raise_adaptive_warning_gap_bb=(
            args.always_raise_adaptive_warning_gap_bb
        ),
        high_always_raise_mean_profit_bb=(
            args.high_always_raise_mean_profit_bb
        ),
        high_always_raise_win_rate=args.high_always_raise_win_rate,
    )


def main() -> None:
    args = parse_args()
    thresholds = build_thresholds(args)
    report = validate_checkpoint_results(
        input_path=args.input_path,
        thresholds=thresholds,
    )

    created_paths = []

    if args.format in {"markdown", "both"}:
        created_paths.append(
            write_validation_markdown_report(
                report,
                args.output_dir,
            )
        )

    if args.format in {"json", "both"}:
        created_paths.append(
            write_validation_json_report(
                report,
                args.output_dir,
            )
        )

    print("Created validation report files:")
    for path in created_paths:
        print(path)

    counts = report.status_counts()
    print(
        "Validation status: "
        f"{'PASS' if report.passed else 'FAIL'} "
        f"{counts}"
    )


if __name__ == "__main__":
    main()

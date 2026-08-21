from __future__ import annotations

import argparse

from src.evaluation.algorithm_metadata import (
    SUPPORTED_ALGORITHM_KEYS,
    algorithm_specs_from_keys,
)
from src.evaluation.validation import (
    VALIDATION_MODE_TRAINING_OPPONENT,
    VALIDATION_MODES,
    ValidationThresholds,
    validate_evaluation_results,
    write_validation_json_report,
    write_validation_markdown_report,
)


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run automated sanity checks on evaluation results.")
    )

    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--format",
        default="both",
        choices=["markdown", "json", "both"],
    )
    parser.add_argument(
        "--validation-mode",
        default=VALIDATION_MODE_TRAINING_OPPONENT,
        choices=VALIDATION_MODES,
        help=(
            "Use 'training-opponent' for final-model benchmark results, "
            "'head-to-head' for direct matchups against baselines, "
            "'generalization' for unseen opponent variants, "
            "'stress-test' for learned agents against scripted extremes, "
            "'baseline-sanity' for the 3x3 scripted baseline matrix, "
            "or 'cross-play' for learned-agent matchups."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=SUPPORTED_ALGORITHM_KEYS,
        default=None,
        help=(
            "Optional subset of algorithms expected in the validation "
            "input. Defaults to algorithms detected in the CSV unless "
            "--require-all-algorithms is used."
        ),
    )
    parser.add_argument(
        "--require-all-algorithms",
        action="store_true",
        help=(
            "Fail validation when an expected final RL algorithm or any "
            "required evaluation matchup is missing from the result file."
        ),
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
        "--min-tight-win-rate",
        type=float,
        default=95.0,
    )
    parser.add_argument(
        "--min-tight-mean-profit-bb",
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
        "--min-seeds-per-matchup",
        type=_positive_int,
        default=2,
        help=(
            "Minimum number of distinct model seeds required for each "
            "evaluated agent/opponent matchup."
        ),
    )
    parser.add_argument(
        "--max-std-across-seeds-bb",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--min-evaluation-replicates-per-matchup",
        type=_positive_int,
        default=2,
        help=(
            "Minimum number of distinct simulation replicates required "
            "for each baseline-only matchup."
        ),
    )
    parser.add_argument(
        "--max-std-across-evaluation-replicates-bb",
        type=float,
        default=5.0,
        help=(
            "Maximum warning-free standard deviation across baseline "
            "evaluation replicates."
        ),
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
    parser.add_argument(
        "--max-baseline-mirror-abs-profit-bb",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-baseline-pair-sum-abs-profit-bb",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-cross-play-pair-sum-abs-profit-bb",
        type=float,
        default=2.0,
        help=(
            "Maximum absolute sum of opposite-direction mean profits "
            "before a learned-agent pair produces a warning."
        ),
    )
    parser.add_argument(
        "--min-head-to-head-mean-profit-bb",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-adaptive-underperformance-vs-general-bb",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--always-raise-stress-loss-bb",
        type=float,
        default=-15.0,
    )
    parser.add_argument(
        "--always-raise-stress-bust-rate",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--min-generalization-positive-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-generalization-adaptive-beats-general-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-generalization-adaptive-beats-rule-based-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-generalization-oracle-gap-bb",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--generalization-extreme-aggressive-min-profit-bb",
        type=float,
        default=-5.0,
    )
    parser.add_argument(
        "--generalization-extreme-aggressive-max-bust-rate",
        type=float,
        default=85.0,
    )

    return parser.parse_args(argv)


def build_thresholds(args: argparse.Namespace) -> ValidationThresholds:
    return ValidationThresholds(
        min_adaptive_delta_vs_rule_based_bb=(args.min_adaptive_delta_vs_rule_based_bb),
        max_oracle_underperformance_bb=(args.max_oracle_underperformance_bb),
        min_tight_win_rate=args.min_tight_win_rate,
        min_tight_mean_profit_bb=args.min_tight_mean_profit_bb,
        min_classifier_accuracy=args.min_classifier_accuracy,
        min_classifier_coverage=args.min_classifier_coverage,
        min_seeds_per_matchup=args.min_seeds_per_matchup,
        max_std_across_seeds_bb=args.max_std_across_seeds_bb,
        min_evaluation_replicates_per_matchup=(
            args.min_evaluation_replicates_per_matchup
        ),
        max_std_across_evaluation_replicates_bb=(
            args.max_std_across_evaluation_replicates_bb
        ),
        extreme_bb_per_100_threshold=(args.extreme_bb_per_100_threshold),
        low_mean_hands_played_threshold=(args.low_mean_hands_played_threshold),
        always_raise_adaptive_warning_gap_bb=(
            args.always_raise_adaptive_warning_gap_bb
        ),
        high_always_raise_mean_profit_bb=(args.high_always_raise_mean_profit_bb),
        high_always_raise_win_rate=args.high_always_raise_win_rate,
        max_baseline_mirror_abs_profit_bb=(args.max_baseline_mirror_abs_profit_bb),
        max_baseline_pair_sum_abs_profit_bb=(args.max_baseline_pair_sum_abs_profit_bb),
        max_cross_play_pair_sum_abs_profit_bb=(
            args.max_cross_play_pair_sum_abs_profit_bb
        ),
        min_head_to_head_mean_profit_bb=(args.min_head_to_head_mean_profit_bb),
        max_adaptive_underperformance_vs_general_bb=(
            args.max_adaptive_underperformance_vs_general_bb
        ),
        always_raise_stress_loss_bb=args.always_raise_stress_loss_bb,
        always_raise_stress_bust_rate=args.always_raise_stress_bust_rate,
        min_generalization_positive_variants=(
            args.min_generalization_positive_variants
        ),
        min_generalization_adaptive_beats_general_variants=(
            args.min_generalization_adaptive_beats_general_variants
        ),
        min_generalization_adaptive_beats_rule_based_variants=(
            args.min_generalization_adaptive_beats_rule_based_variants
        ),
        max_generalization_oracle_gap_bb=(args.max_generalization_oracle_gap_bb),
        generalization_extreme_aggressive_min_profit_bb=(
            args.generalization_extreme_aggressive_min_profit_bb
        ),
        generalization_extreme_aggressive_max_bust_rate=(
            args.generalization_extreme_aggressive_max_bust_rate
        ),
    )


def main() -> int:
    args = parse_args()
    thresholds = build_thresholds(args)
    report = validate_evaluation_results(
        input_path=args.input_path,
        thresholds=thresholds,
        validation_mode=args.validation_mode,
        algorithm_specs=algorithm_specs_from_keys(args.algorithms),
        require_all_algorithms=args.require_all_algorithms,
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
    print(f"Validation status: {'PASS' if report.passed else 'FAIL'} {counts}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
from itertools import combinations
from pathlib import Path

from src.evaluation.constants import SUPPORTED_POLICY_TYPES
from src.evaluation.diagnostics.q_table_comparator import (
    build_selected_targets,
    compare_q_tables,
    find_largest_disagreements,
    load_q_table,
    save_json,
    summarize_q_table,
    validate_targets_exist,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Q-tables for selected policies, seeds and checkpoint. "
            "The script loads trained Monte Carlo Q-tables, summarizes action "
            "preferences and compares learned policies on common abstract states."
        )
    )

    parser.add_argument(
        "--training-run-dir",
        required=True,
        type=str,
        help=(
            "Directory with training run checkpoints, for example: "
            "results/training_runs/state_v2_linear_2000_sqrt_visit"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=int,
        help=(
            "Checkpoint episode to compare, for example 1500, 2000 or 4000."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Training seeds to compare, for example: 42 123 456.",
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=list(SUPPORTED_POLICY_TYPES),
        default=list(SUPPORTED_POLICY_TYPES),
        help=(
            "Policies to compare. Available values: "
            f"{list(SUPPORTED_POLICY_TYPES)}."
        ),
    )

    parser.add_argument(
        "--output-path",
        default="results/evaluation/q_table_comparison.json",
        type=str,
    )

    parser.add_argument(
        "--top-n",
        default=20,
        type=int,
    )

    args = parser.parse_args(argv)

    if args.checkpoint <= 0:
        parser.error("--checkpoint must be greater than zero")

    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")

    if args.top_n <= 0:
        parser.error("--top-n must be greater than zero")

    return args


def print_summary(summary) -> None:
    print("\n" + "=" * 100)
    print(summary.name)
    print("=" * 100)
    print(f"path={summary.path}")
    print(f"policy_type={summary.policy_type}")
    print(f"seed={summary.seed}")
    print(f"checkpoint_episode={summary.checkpoint_episode}")
    print(f"states={summary.states}")
    print(
        "fully_zero_states="
        f"{summary.fully_zero_states} "
        f"({summary.fully_zero_rate:.2f}%)"
    )
    print(
        "tied_best_states="
        f"{summary.tied_best_states} "
        f"({summary.tied_best_rate:.2f}%)"
    )

    print("\nbest_action_counts:")
    for action, count in summary.best_action_counts.items():
        rate = summary.best_action_rates[action]
        print(f"  {action:>5}: {count:>5} ({rate:>6.2f}%)")

    print("\naction_stats:")
    for stats in summary.action_stats:
        print(
            f"  {stats.action:>5}: "
            f"mean={stats.mean_q:>8.4f}, "
            f"median={stats.median_q:>8.4f}, "
            f"std={stats.std_q:>8.4f}, "
            f"min={stats.min_q:>8.4f}, "
            f"max={stats.max_q:>8.4f}, "
            f"zero_rate={stats.zero_rate:>6.2f}%"
        )


def print_pairwise(comparison) -> None:
    print("\n" + "-" * 100)
    print(f"{comparison.left_name}  VS  {comparison.right_name}")
    print("-" * 100)
    print(
        "left="
        f"{comparison.left_policy_type}, "
        f"seed={comparison.left_seed}, "
        f"checkpoint={comparison.left_checkpoint_episode}"
    )
    print(
        "right="
        f"{comparison.right_policy_type}, "
        f"seed={comparison.right_seed}, "
        f"checkpoint={comparison.right_checkpoint_episode}"
    )
    print(f"left_states={comparison.left_states}")
    print(f"right_states={comparison.right_states}")
    print(f"common_states={comparison.common_states}")
    print(f"left_only_states={comparison.left_only_states}")
    print(f"right_only_states={comparison.right_only_states}")
    print(
        "best_action_agreement="
        f"{comparison.best_action_agreement} "
        f"({comparison.best_action_agreement_rate:.2f}%)"
    )
    print(
        "mean_max_abs_q_delta="
        f"{comparison.mean_max_abs_q_delta:.4f}"
    )

    print("\nmean_abs_q_delta_by_action:")
    for action, value in comparison.mean_abs_q_delta_by_action.items():
        print(f"  {action:>5}: {value:.4f}")

    print("\ntransition_counts: left_best_action -> right_best_action")
    for left_action, row in comparison.transition_counts.items():
        formatted = ", ".join(
            f"{right_action}={count}"
            for right_action, count in row.items()
        )
        print(f"  {left_action:>5}: {formatted}")


def should_compare_pair(left_target, right_target) -> bool:
    """
    Keep the report useful instead of producing all possible noisy pairs.

    We compare:
    1. Same seed, different policies -> specialization differences.
    2. Same policy, different seeds -> training stability.
    """
    same_seed = left_target.seed == right_target.seed
    same_policy = left_target.policy_type == right_target.policy_type

    return same_seed != same_policy


def main() -> None:
    args = parse_args()

    targets = build_selected_targets(
        training_run_directory=args.training_run_dir,
        checkpoint_episode=args.checkpoint,
        seeds=args.seeds,
        policies=args.policies,
    )

    validate_targets_exist(targets)

    q_tables = {
        target.name: load_q_table(target.path)
        for target in targets
    }

    abstract_q_tables = q_tables

    summaries = []

    for target in targets:
        summary = summarize_q_table(
            target=target,
            q_table=q_tables[target.name],
        )
        summaries.append(summary)
        print_summary(summary)

    comparisons = []
    disagreements = {}

    for left_target, right_target in combinations(targets, 2):
        if not should_compare_pair(left_target, right_target):
            continue

        comparison = compare_q_tables(
            left_target=left_target,
            left_q_table=abstract_q_tables[left_target.name],
            right_target=right_target,
            right_q_table=abstract_q_tables[right_target.name],
        )

        comparisons.append(comparison)
        print_pairwise(comparison)

        key = f"{left_target.name}__vs__{right_target.name}"

        disagreements[key] = find_largest_disagreements(
            left_target=left_target,
            left_q_table=abstract_q_tables[left_target.name],
            right_target=right_target,
            right_q_table=abstract_q_tables[right_target.name],
            top_n=args.top_n,
        )

    output = {
        "training_run_dir": args.training_run_dir,
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "policies": args.policies,
        "comparison_strategy": (
            "same_seed_different_policies_and_"
            "same_policy_different_seeds"
        ),
        "targets": [
            {
                "name": target.name,
                "path": str(target.path),
                "policy_type": target.policy_type,
                "seed": target.seed,
                "checkpoint_episode": target.checkpoint_episode,
            }
            for target in targets
        ],
        "summaries": summaries,
        "comparisons": comparisons,
        "largest_disagreements": disagreements,
    }

    save_json(
        path=args.output_path,
        data=output,
    )

    print("\nSaved JSON report:")
    print(Path(args.output_path))


if __name__ == "__main__":
    main()

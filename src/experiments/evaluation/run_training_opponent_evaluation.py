import argparse
import json
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from time import perf_counter

from src.evaluation.constants import (
    ADAPTIVE_MC_AGENT,
    POLICY_GENERAL_MC_AGENT,
    RULE_BASED_AGENT,
    SUPPORTED_TESTED_AGENTS,
)
from src.evaluation.runners.model_evaluator import (
    TrainingOpponentEvaluationConfig,
    discover_final_model_bundles,
    evaluate_training_opponent_bundle,
    write_rows,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final trained model from every selected training "
            "seed against the base training opponents."
        )
    )

    parser.add_argument(
        "--training-run-dir",
        required=True,
        type=str,
        help=(
            "Directory created by run_training_suite, "
            "for example "
            "results/training_runs/state_v2_linear_10000."
        ),
    )

    parser.add_argument(
        "--q-learning-run-dir",
        type=str,
        default=None,
        help=(
            "Optional Q-learning training run directory. When provided, "
            "Q-learning evaluation agents can be selected with --agents."
        ),
    )

    parser.add_argument(
        "--sarsa-run-dir",
        type=str,
        default=None,
        help=(
            "Optional SARSA training run directory. When provided, "
            "SARSA evaluation agents can be selected with --agents."
        ),
    )

    parser.add_argument(
        "--double-q-learning-run-dir",
        type=str,
        default=None,
        help=(
            "Optional Double Q-learning training run directory. When provided, "
            "Double Q-learning evaluation agents can be selected with --agents."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds to evaluate. When omitted, all seed_* directories are discovered."
        ),
    )

    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Games per matchup.",
    )

    parser.add_argument(
        "--agents",
        choices=sorted(SUPPORTED_TESTED_AGENTS),
        nargs="+",
        default=[
            POLICY_GENERAL_MC_AGENT,
            ADAPTIVE_MC_AGENT,
            RULE_BASED_AGENT,
        ],
    )

    parser.add_argument(
        "--opponents",
        choices=sorted(TRAINING_OPPONENT_TYPES),
        nargs="+",
        default=list(TRAINING_OPPONENT_TYPES),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<training-run-dir>/training_opponent_evaluation.csv"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of model bundles evaluated in parallel. "
            "Use 1 for the most predictable run."
        ),
    )

    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=100_000,
        help=("Base seed used to make evaluation reproducible."),
    )

    parser.add_argument(
        "--fail-on-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail when any requested seed/final-model bundle "
            "is incomplete. By default incomplete bundles "
            "are skipped."
        ),
    )

    args = parser.parse_args(argv)

    if args.games <= 0:
        parser.error("--games must be greater than zero")

    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    return args


def evaluate_bundle_worker(
    bundle,
    config: TrainingOpponentEvaluationConfig,
) -> list[dict]:
    return evaluate_training_opponent_bundle(
        bundle=bundle,
        config=config,
    )


def save_summary(
    *,
    output_path: Path,
    arguments: argparse.Namespace,
    bundle_count: int,
    training_episodes: list[int],
    row_count: int,
    duration_seconds: float,
) -> None:
    summary_path = output_path.with_suffix(".summary.json")

    summary = {
        "output_path": str(output_path),
        "training_run_dir": arguments.training_run_dir,
        "q_learning_run_dir": arguments.q_learning_run_dir,
        "sarsa_run_dir": arguments.sarsa_run_dir,
        "double_q_learning_run_dir": arguments.double_q_learning_run_dir,
        "model_source": "final",
        "training_episodes": training_episodes,
        "seeds": arguments.seeds,
        "games": arguments.games,
        "agents": arguments.agents,
        "opponents": arguments.opponents,
        "workers": arguments.workers,
        "eval_seed_base": arguments.eval_seed_base,
        "bundle_count": bundle_count,
        "row_count": row_count,
        "duration_seconds": duration_seconds,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()

    training_run_dir = Path(args.training_run_dir)

    output_path = Path(
        args.output_path
        if args.output_path is not None
        else training_run_dir / "training_opponent_evaluation.csv"
    )

    bundles = discover_final_model_bundles(
        training_run_directory=training_run_dir,
        seeds=args.seeds,
        skip_incomplete=not args.fail_on_incomplete,
        q_learning_run_directory=args.q_learning_run_dir,
        sarsa_run_directory=args.sarsa_run_dir,
        double_q_learning_run_directory=args.double_q_learning_run_dir,
    )

    if not bundles:
        raise SystemExit(
            "No complete final model bundles found for the requested seeds."
        )

    config = TrainingOpponentEvaluationConfig(
        games_per_matchup=args.games,
        opponents=tuple(args.opponents),
        tested_agents=tuple(args.agents),
        eval_seed_base=args.eval_seed_base,
        output_path=output_path,
    )

    print(
        "Training-opponent evaluation started\n"
        f"training_run_dir={training_run_dir}\n"
        f"q_learning_run_dir={args.q_learning_run_dir or 'not provided'}\n"
        f"sarsa_run_dir={args.sarsa_run_dir or 'not provided'}\n"
        f"double_q_learning_run_dir={args.double_q_learning_run_dir or 'not provided'}\n"
        f"bundles={len(bundles)}\n"
        f"training_episodes={sorted({bundle.episode for bundle in bundles})}\n"
        f"seeds={args.seeds or 'auto'}\n"
        f"games_per_matchup={args.games}\n"
        f"agents={args.agents}\n"
        f"opponents={args.opponents}\n"
        f"workers={args.workers}\n"
        f"output={output_path}"
    )

    started_at = perf_counter()

    all_rows: list[dict] = []

    if args.workers == 1:
        for index, bundle in enumerate(
            bundles,
            start=1,
        ):
            print(f"[{index}/{len(bundles)}] Evaluating {bundle.experiment_id}")

            rows = evaluate_training_opponent_bundle(
                bundle=bundle,
                config=config,
            )

            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_bundle = {
                executor.submit(
                    evaluate_bundle_worker,
                    bundle,
                    config,
                ): bundle
                for bundle in bundles
            }

            for completed, future in enumerate(
                as_completed(future_to_bundle),
                start=1,
            ):
                bundle = future_to_bundle[future]
                rows = future.result()
                all_rows.extend(rows)
                print(f"[{completed}/{len(bundles)}] Finished {bundle.experiment_id}")

    write_rows(
        output_path=output_path,
        rows=all_rows,
    )

    duration_seconds = perf_counter() - started_at

    save_summary(
        output_path=output_path,
        arguments=args,
        bundle_count=len(bundles),
        training_episodes=sorted({bundle.episode for bundle in bundles}),
        row_count=len(all_rows),
        duration_seconds=duration_seconds,
    )

    print(
        "Training-opponent evaluation finished\n"
        f"bundles={len(bundles)}\n"
        f"rows={len(all_rows)}\n"
        f"duration_seconds={duration_seconds:.3f}\n"
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

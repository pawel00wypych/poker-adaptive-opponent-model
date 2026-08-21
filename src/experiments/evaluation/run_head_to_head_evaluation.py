import argparse
import json
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from time import perf_counter

from src.evaluation.runners.head_to_head_evaluator import (
    DEFAULT_HEAD_TO_HEAD_AGENTS,
    DEFAULT_HEAD_TO_HEAD_OPPONENTS,
    SUPPORTED_HEAD_TO_HEAD_AGENTS,
    SUPPORTED_HEAD_TO_HEAD_OPPONENTS,
    HeadToHeadEvaluationConfig,
    baseline_tested_agents,
    evaluate_baseline_replicate,
    evaluate_baseline_replicates,
    evaluate_head_to_head_bundle,
    learned_tested_agents,
    write_head_to_head_rows,
)
from src.evaluation.runners.model_evaluator import (
    discover_final_model_bundles,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run direct head-to-head evaluation between learned policies, "
            "rule-based baselines and deterministic sanity baselines."
        )
    )

    parser.add_argument(
        "--training-run-dir",
        type=str,
        default=None,
        help=(
            "Required only for learned agents. Directory created by "
            "run_training_suite, for example "
            "results/training_runs/state_v2_linear_2000_sqrt_visit."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Training/model seeds to evaluate for learned agents. When "
            "omitted, all seed_* directories are discovered."
        ),
    )

    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Games per direct matchup.",
    )

    parser.add_argument(
        "--evaluation-replicates",
        type=int,
        default=5,
        help=(
            "Independent card/simulation replicates for baseline-only "
            "matchups. These are not training seeds."
        ),
    )

    parser.add_argument(
        "--agents",
        choices=sorted(SUPPORTED_HEAD_TO_HEAD_AGENTS),
        nargs="+",
        default=list(DEFAULT_HEAD_TO_HEAD_AGENTS),
        help=("Agents/policies to evaluate against direct baseline opponents."),
    )

    parser.add_argument(
        "--opponents",
        choices=sorted(SUPPORTED_HEAD_TO_HEAD_OPPONENTS),
        nargs="+",
        default=list(DEFAULT_HEAD_TO_HEAD_OPPONENTS),
        help="Direct baseline opponents.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Output CSV path. Default: <training-run-dir>/"
            "head_to_head_evaluation.csv when learned agents are selected, "
            "otherwise results/evaluation/head_to_head_evaluation.csv."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of model bundles or evaluation replicates run in "
            "parallel. Use 1 for the most predictable run."
        ),
    )

    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=300_000,
        help="Base seed used to make direct evaluation reproducible.",
    )

    parser.add_argument(
        "--fail-on-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail when any requested seed/final-model bundle is incomplete. "
            "By default incomplete bundles are skipped."
        ),
    )

    args = parser.parse_args(argv)

    if args.games <= 0:
        parser.error("--games must be greater than zero")

    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    if args.evaluation_replicates <= 0:
        parser.error("--evaluation-replicates must be greater than zero")

    if args.eval_seed_base < 0:
        parser.error("--eval-seed-base must be non-negative")

    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error("All model seeds must be non-negative")

    learned_agents = learned_tested_agents(args.agents)
    if learned_agents:
        if args.training_run_dir is None:
            parser.error(
                "--training-run-dir is required when learned agents are selected"
            )
    else:
        model_only_options = []
        if args.training_run_dir is not None:
            model_only_options.append("--training-run-dir")
        if args.seeds is not None:
            model_only_options.append("--seeds")
        if args.fail_on_incomplete:
            model_only_options.append("--fail-on-incomplete")
        if model_only_options:
            parser.error(
                f"{', '.join(model_only_options)} apply only to learned "
                "agents. Baseline-only runs use --evaluation-replicates."
            )

    return args


def evaluate_bundle_worker(
    bundle,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    return evaluate_head_to_head_bundle(
        bundle=bundle,
        config=config,
    )


def evaluate_baseline_replicate_worker(
    evaluation_replicate_id: int,
    config: HeadToHeadEvaluationConfig,
) -> list[dict]:
    return evaluate_baseline_replicate(
        evaluation_replicate_id=evaluation_replicate_id,
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
        "evaluation_type": "direct_head_to_head",
        "output_path": str(output_path),
        "training_run_dir": arguments.training_run_dir,
        "model_source": "final" if training_episodes else None,
        "training_episodes": training_episodes,
        "model_seeds": arguments.seeds,
        "games": arguments.games,
        "evaluation_replicates": (
            arguments.evaluation_replicates
            if baseline_tested_agents(arguments.agents)
            else None
        ),
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

    learned_agents = learned_tested_agents(args.agents)
    baseline_agents = baseline_tested_agents(args.agents)
    training_run_dir = (
        Path(args.training_run_dir) if args.training_run_dir is not None else None
    )

    output_path = Path(
        args.output_path
        if args.output_path is not None
        else (
            training_run_dir / "head_to_head_evaluation.csv"
            if training_run_dir is not None
            else Path("results/evaluation/head_to_head_evaluation.csv")
        )
    )

    bundles = []
    if learned_agents:
        assert training_run_dir is not None
        bundles = discover_final_model_bundles(
            training_run_directory=training_run_dir,
            seeds=args.seeds,
            skip_incomplete=not args.fail_on_incomplete,
        )

    if learned_agents and not bundles:
        raise SystemExit(
            "No complete final model bundles found for the requested seeds."
        )

    config = HeadToHeadEvaluationConfig(
        games_per_matchup=args.games,
        opponents=tuple(args.opponents),
        tested_agents=tuple(args.agents),
        eval_seed_base=args.eval_seed_base,
        output_path=output_path,
        evaluation_replicates=args.evaluation_replicates,
    )

    print(
        "Direct head-to-head evaluation started\n"
        f"training_run_dir={training_run_dir or 'not_applicable'}\n"
        f"bundles={len(bundles)}\n"
        f"training_episodes={sorted({bundle.episode for bundle in bundles}) or 'not_applicable'}\n"
        f"model_seeds={args.seeds or ('auto' if learned_agents else 'not_applicable')}\n"
        f"evaluation_replicates={args.evaluation_replicates if baseline_agents else 'not_applicable'}\n"
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

            rows = evaluate_head_to_head_bundle(
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

    if baseline_agents:
        if args.workers == 1:
            all_rows.extend(evaluate_baseline_replicates(config=config))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_replicate = {
                    executor.submit(
                        evaluate_baseline_replicate_worker,
                        evaluation_replicate_id,
                        config,
                    ): evaluation_replicate_id
                    for evaluation_replicate_id in range(args.evaluation_replicates)
                }
                for completed, future in enumerate(
                    as_completed(future_to_replicate),
                    start=1,
                ):
                    replicate_id = future_to_replicate[future]
                    all_rows.extend(future.result())
                    print(
                        f"[{completed}/{args.evaluation_replicates}] "
                        "Finished evaluation replicate "
                        f"{replicate_id}"
                    )

    write_head_to_head_rows(
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
        "Direct head-to-head evaluation finished\n"
        f"bundles={len(bundles)}\n"
        f"rows={len(all_rows)}\n"
        f"duration_seconds={duration_seconds:.3f}\n"
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()

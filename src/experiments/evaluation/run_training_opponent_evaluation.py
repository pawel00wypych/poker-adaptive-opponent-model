import argparse
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from time import perf_counter

from src.evaluation.algorithm_metadata import (
    ADAPTIVE_AGENTS,
    GENERAL_POLICY_AGENTS,
    ORACLE_ALGORITHM_AGENTS,
)
from src.evaluation.constants import (
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
    RULE_BASED_AGENT,
    SUPPORTED_TESTED_AGENTS,
)
from src.evaluation.runners.model_evaluator import (
    TrainingOpponentEvaluationConfig,
    discover_final_model_bundles,
    evaluate_training_opponent_bundle,
    partition_agents_by_support,
    write_rows,
)
from src.experiment_protocol import TRAINING_OPPONENT_EVALUATION
from src.experiments.evaluation.protocol_cli import (
    attach_model_provenance,
    add_evaluation_protocol_arguments,
    model_provenance_summary,
    resolve_evaluation_protocol,
    save_evaluation_summary,
)
from src.poker.constants import TRAINING_OPPONENT_TYPES

# Experiment 1 of final_experiment_guidelines.md: the four adaptive agents,
# their oracle benchmarks, the four non-adaptive general policies, and the
# three sanity baselines. Defaulting to a narrower set meant running the
# documented experiment required undocumented flags, and made an incomplete
# result set easy to produce by accident.
DEFAULT_TRAINING_OPPONENT_AGENTS = (
    *ADAPTIVE_AGENTS,
    *ORACLE_ALGORITHM_AGENTS,
    *GENERAL_POLICY_AGENTS,
    RULE_BASED_AGENT,
    ALWAYS_CALL_AGENT,
    ALWAYS_RAISE_AGENT,
)


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

    add_evaluation_protocol_arguments(parser)

    parser.add_argument(
        "--agents",
        choices=sorted(SUPPORTED_TESTED_AGENTS),
        nargs="+",
        default=list(DEFAULT_TRAINING_OPPONENT_AGENTS),
        help=(
            "Tested agents. Defaults to the full experiment-1 set from the "
            "guidelines. Agents whose algorithm was not trained are skipped "
            "with a notice, or rejected under --fail-on-incomplete."
        ),
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
        "--fail-on-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail when any requested seed/final-model bundle "
            "is incomplete. By default incomplete bundles "
            "are skipped."
        ),
    )

    args = resolve_evaluation_protocol(
        parser.parse_args(argv),
        evaluation_type=TRAINING_OPPONENT_EVALUATION,
    )

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
    evaluated_agents: tuple[str, ...],
    skipped_agents: dict[str, str],
) -> None:
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
        "requested_agents": arguments.agents,
        "evaluated_agents": list(evaluated_agents),
        "skipped_agents": skipped_agents,
        "opponents": arguments.opponents,
        "workers": arguments.workers,
        "eval_seed_base": arguments.eval_seed_base,
        "bundle_count": bundle_count,
        "row_count": row_count,
        "duration_seconds": duration_seconds,
        **model_provenance_summary(arguments),
    }
    save_evaluation_summary(
        output_path=output_path,
        summary=summary,
        provenance=arguments.protocol_provenance,
    )


RUN_DIRECTORY_FLAG_FOR_ALGORITHM = {
    "q_learning": "--q-learning-run-dir",
    "sarsa": "--sarsa-run-dir",
    "double_q_learning": "--double-q-learning-run-dir",
}


def resolve_agent_support(
    *,
    bundles,
    requested_agents: tuple[str, ...],
    fail_on_incomplete: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Report which requested agents every bundle can actually evaluate.

    An agent is only kept when *every* bundle supports it, so the result set
    stays rectangular - a seed that silently contributed fewer agents than the
    others would skew any comparison across seeds.
    """
    evaluated = tuple(
        agent
        for agent in requested_agents
        if all(
            agent in partition_agents_by_support(bundle, requested_agents)[0]
            for bundle in bundles
        )
    )

    skipped: dict[str, str] = {}
    for bundle in bundles:
        _, bundle_skipped = partition_agents_by_support(bundle, requested_agents)
        skipped.update(bundle_skipped)

    if not skipped:
        return evaluated, skipped

    lines = [
        f"  {agent} needs {algorithm} models "
        f"({RUN_DIRECTORY_FLAG_FOR_ALGORITHM.get(algorithm, 'a run directory')})"
        for agent, algorithm in sorted(skipped.items())
    ]
    message = "Some requested agents have no trained models:\n" + "\n".join(lines)

    if fail_on_incomplete:
        raise SystemExit(message)

    print(f"{message}\nSkipping them. Pass --fail-on-incomplete to treat this "
          "as an error instead.")

    if not evaluated:
        raise SystemExit(
            "No requested agent can be evaluated with the supplied models."
        )

    return evaluated, skipped


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
    attach_model_provenance(args, bundles)

    config = TrainingOpponentEvaluationConfig(
        games_per_matchup=args.games,
        opponents=tuple(args.opponents),
        tested_agents=tuple(args.agents),
        eval_seed_base=args.eval_seed_base,
        output_path=output_path,
        game_config=args.experiment_config.game,
    )

    evaluated_agents, skipped_agents = resolve_agent_support(
        bundles=bundles,
        requested_agents=tuple(args.agents),
        fail_on_incomplete=args.fail_on_incomplete,
    )

    print(
        "Training-opponent evaluation started\n"
        f"training_run_dir={training_run_dir}\n"
        f"q_learning_run_dir={args.q_learning_run_dir or 'not provided'}\n"
        f"sarsa_run_dir={args.sarsa_run_dir or 'not provided'}\n"
        f"double_q_learning_run_dir="
        f"{args.double_q_learning_run_dir or 'not provided'}\n"
        f"bundles={len(bundles)}\n"
        f"training_episodes={sorted({bundle.episode for bundle in bundles})}\n"
        f"seeds={args.seeds or 'auto'}\n"
        f"games_per_matchup={args.games}\n"
        f"requested_agents={args.agents}\n"
        f"evaluated_agents={list(evaluated_agents)}\n"
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
        evaluated_agents=evaluated_agents,
        skipped_agents=skipped_agents,
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

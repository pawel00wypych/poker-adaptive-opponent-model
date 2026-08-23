"""The evaluation CLI should run the documented experiment by default.

Experiment 1 of finalny_zestaw_eksperymentow.md names fifteen agents. The CLI
defaulted to three, so running the documented experiment required undocumented
flags and an incomplete result set was easy to produce by accident.

Widening the default is not a one-line change: an agent whose algorithm was
never trained used to raise deep inside the loaders, turning a missing
--q-learning-run-dir into a crash partway through a run.
"""

import json
import re
from pathlib import Path

import pytest

from src.evaluation.runners.model_evaluator import ModelBundle
from src.experiments.evaluation.run_training_opponent_evaluation import parse_args

GUIDELINES = Path("finalny_zestaw_eksperymentow.md")


def _guideline_agents_for_experiment_one():
    lines = GUIDELINES.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# 1."))
    end = next(
        i for i, line in enumerate(lines) if i > start and line.startswith("# 2.")
    )
    block = "\n".join(lines[start:end])

    return set(re.findall(r"^(\w+) vs \w+$", block, flags=re.M))


def _bundle(*, q_learning=False, sarsa=False, double_q_learning=False):
    path = Path("model.pkl")
    optional = {}

    for enabled, prefix in (
        (q_learning, "q_learning"),
        (sarsa, "sarsa"),
        (double_q_learning, "double_q_learning"),
    ):
        if not enabled:
            continue
        for policy in ("unknown", "tight", "aggressive", "calling"):
            optional[f"{prefix}_{policy}_model_path"] = path

    return ModelBundle(
        training_run_directory=Path("run"),
        seed=42,
        episode=100,
        model_source="final",
        unknown_model_path=path,
        tight_model_path=path,
        aggressive_model_path=path,
        calling_model_path=path,
        **optional,
    )


def test_default_agents_match_the_guideline_experiment():
    """Parsed from the guidelines so the two cannot drift apart."""
    namespace = parse_args(["--training-run-dir", "run"])

    assert set(namespace.agents) == _guideline_agents_for_experiment_one()


def test_the_default_covers_all_four_algorithms():
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
    )

    agents = set(DEFAULT_TRAINING_OPPONENT_AGENTS)

    for algorithm in ("mc", "q_learning", "sarsa", "double_q_learning"):
        assert any(agent.endswith(algorithm) for agent in agents), algorithm


def test_a_monte_carlo_only_run_skips_td_agents_instead_of_crashing(capsys):
    """The constraint that makes this more than a default change.

    bundle.q_learning_agent_paths() raises when the run directory was not
    supplied, and --fail-on-incomplete only skipped incomplete *bundles*, not
    agents of an untrained algorithm.
    """
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
        resolve_agent_support,
    )

    evaluated, skipped = resolve_agent_support(
        bundles=[_bundle()],
        requested_agents=DEFAULT_TRAINING_OPPONENT_AGENTS,
        fail_on_incomplete=False,
    )

    assert evaluated, "a Monte-Carlo-only run must still evaluate something"
    assert skipped

    for agent in evaluated:
        assert "q_learning" not in agent or agent.endswith("_mc")
        assert "sarsa" not in agent

    output = capsys.readouterr().out
    assert "--q-learning-run-dir" in output
    assert "--sarsa-run-dir" in output
    assert "--double-q-learning-run-dir" in output


def test_fail_on_incomplete_turns_a_skip_into_an_error():
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
        resolve_agent_support,
    )

    with pytest.raises(SystemExit, match="no trained models"):
        resolve_agent_support(
            bundles=[_bundle()],
            requested_agents=DEFAULT_TRAINING_OPPONENT_AGENTS,
            fail_on_incomplete=True,
        )


def test_a_complete_bundle_skips_nothing(capsys):
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
        resolve_agent_support,
    )

    evaluated, skipped = resolve_agent_support(
        bundles=[_bundle(q_learning=True, sarsa=True, double_q_learning=True)],
        requested_agents=DEFAULT_TRAINING_OPPONENT_AGENTS,
        fail_on_incomplete=False,
    )

    assert set(evaluated) == set(DEFAULT_TRAINING_OPPONENT_AGENTS)
    assert skipped == {}
    assert capsys.readouterr().out == ""


def test_an_agent_is_dropped_unless_every_bundle_supports_it(capsys):
    """Otherwise one seed contributes fewer agents than the others.

    A result set where seeds cover different agents cannot be compared across
    seeds, which is where every claim in the thesis is made.
    """
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
        resolve_agent_support,
    )

    evaluated, _ = resolve_agent_support(
        bundles=[_bundle(q_learning=True), _bundle()],
        requested_agents=DEFAULT_TRAINING_OPPONENT_AGENTS,
        fail_on_incomplete=False,
    )

    assert "adaptive_q_learning" not in evaluated


def test_a_request_with_no_usable_agent_fails_loudly(capsys):
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        resolve_agent_support,
    )

    with pytest.raises(SystemExit, match="No requested agent can be evaluated"):
        resolve_agent_support(
            bundles=[_bundle()],
            requested_agents=("adaptive_q_learning", "adaptive_sarsa"),
            fail_on_incomplete=False,
        )


@pytest.mark.parametrize(
    "agent",
    ["rule_based", "always_call", "always_raise", "adaptive_mc", "oracle_mc"],
)
def test_monte_carlo_and_baseline_agents_need_no_extra_run_directory(agent):
    from src.evaluation.runners.model_evaluator import bundle_supports_agent

    assert bundle_supports_agent(_bundle(), agent)


@pytest.mark.parametrize(
    ("agent", "flag"),
    [
        ("adaptive_q_learning", "q_learning"),
        ("oracle_sarsa", "sarsa"),
        ("policy_general_double_q_learning", "double_q_learning"),
    ],
)
def test_td_agents_require_their_own_models(agent, flag):
    from src.evaluation.runners.model_evaluator import bundle_supports_agent

    assert not bundle_supports_agent(_bundle(), agent)
    assert bundle_supports_agent(_bundle(**{flag: True}), agent)


def test_partition_reports_which_algorithm_is_missing():
    from src.evaluation.runners.model_evaluator import (
        partition_agents_by_support,
    )

    _, skipped = partition_agents_by_support(
        _bundle(), ("adaptive_q_learning", "adaptive_mc")
    )

    assert skipped == {"adaptive_q_learning": "q_learning"}


def test_the_summary_records_which_agents_were_evaluated(tmp_path):
    """A result file has to state what it actually contains.

    Otherwise a run silently missing three algorithms looks identical to a
    complete one.
    """
    from src.experiments.evaluation.run_training_opponent_evaluation import (
        DEFAULT_TRAINING_OPPONENT_AGENTS,
        save_summary,
    )

    output_path = tmp_path / "results.csv"
    arguments = parse_args(["--training-run-dir", "run"])

    save_summary(
        output_path=output_path,
        arguments=arguments,
        bundle_count=1,
        training_episodes=[100],
        row_count=0,
        duration_seconds=1.0,
        evaluated_agents=("adaptive_mc", "rule_based"),
        skipped_agents={"adaptive_sarsa": "sarsa"},
    )

    summary = json.loads(
        output_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )

    assert summary["evaluated_agents"] == ["adaptive_mc", "rule_based"]
    assert summary["skipped_agents"] == {"adaptive_sarsa": "sarsa"}
    assert set(summary["requested_agents"]) == set(DEFAULT_TRAINING_OPPONENT_AGENTS)

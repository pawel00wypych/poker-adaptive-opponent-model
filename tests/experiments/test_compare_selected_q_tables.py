import pytest

from src.evaluation.diagnostics.q_table_comparator import build_selected_targets
from src.experiments.diagnostics.compare_selected_q_tables import (
    parse_args,
    should_compare_pair,
)


def test_parse_args_accepts_general_q_table_comparison_arguments():
    args = parse_args(
        [
            "--training-run-dir",
            "results/training_runs/state_v2_linear_2000_sqrt_visit",
            "--checkpoint",
            "2000",
            "--seeds",
            "42",
            "123",
            "456",
            "--policies",
            "unknown",
            "tight",
            "aggressive",
            "calling",
            "--output-path",
            "results/evaluation/q_table_comparison.json",
            "--top-n",
            "30",
        ]
    )

    assert args.training_run_dir == (
        "results/training_runs/state_v2_linear_2000_sqrt_visit"
    )
    assert args.checkpoint == 2000
    assert args.seeds == [42, 123, 456]
    assert args.policies == [
        "unknown",
        "tight",
        "aggressive",
        "calling",
    ]
    assert args.output_path == "results/evaluation/q_table_comparison.json"
    assert args.top_n == 30


def test_parse_args_rejects_invalid_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--training-run-dir",
                "results/training_runs/example",
                "--checkpoint",
                "0",
                "--seeds",
                "42",
            ]
        )


def test_parse_args_rejects_negative_seed():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--training-run-dir",
                "results/training_runs/example",
                "--checkpoint",
                "2000",
                "--seeds",
                "-1",
            ]
        )


def test_parse_args_rejects_invalid_top_n():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--training-run-dir",
                "results/training_runs/example",
                "--checkpoint",
                "2000",
                "--seeds",
                "42",
                "--top-n",
                "0",
            ]
        )


def test_should_compare_pair_accepts_same_seed_different_policies(tmp_path):
    unknown_target, calling_target = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown", "calling"],
    )

    assert should_compare_pair(
        unknown_target,
        calling_target,
    )


def test_should_compare_pair_accepts_same_policy_different_seeds(tmp_path):
    first_target, second_target = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42, 123],
        policies=["calling"],
    )

    assert should_compare_pair(
        first_target,
        second_target,
    )


def test_should_compare_pair_rejects_different_seed_different_policy(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42, 123],
        policies=["unknown", "calling"],
    )

    unknown_seed_42 = targets[0]
    calling_seed_123 = targets[3]

    assert not should_compare_pair(
        unknown_seed_42,
        calling_seed_123,
    )


def test_should_compare_pair_rejects_same_seed_same_policy(tmp_path):
    target = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown"],
    )[0]

    assert not should_compare_pair(
        target,
        target,
    )

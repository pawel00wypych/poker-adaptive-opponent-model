import pickle

import pytest

from src.evaluation.q_table_comparator import (
    best_action,
    build_selected_targets,
    checkpoint_filename,
    checkpoint_model_path,
    compare_q_tables,
    is_fully_zero,
    is_tied_best,
    load_q_table,
    normalize_q_table,
    strip_opponent_type,
    summarize_q_table,
    validate_targets_exist,
)


def test_best_action_returns_index_with_highest_q_value():
    assert best_action([0.1, 2.0, 1.0]) == 1


def test_is_fully_zero():
    assert is_fully_zero([0.0, 0.0, 0.0])
    assert not is_fully_zero([0.0, 0.1, 0.0])


def test_is_tied_best():
    assert is_tied_best([1.0, 1.0, 0.0])
    assert not is_tied_best([1.0, 0.5, 0.0])


def test_checkpoint_filename_for_unknown_policy():
    assert checkpoint_filename(
        policy_type="unknown",
        checkpoint_episode=2000,
        seed=42,
    ) == "single_policy_episodes_2000_seed_42.pkl"


def test_checkpoint_filename_for_specialist_policy():
    assert checkpoint_filename(
        policy_type="calling",
        checkpoint_episode=2000,
        seed=123,
    ) == "specialist_calling_episodes_2000_seed_123.pkl"


def test_checkpoint_filename_rejects_unsupported_policy():
    with pytest.raises(ValueError, match="Unsupported policy type"):
        checkpoint_filename(
            policy_type="loose",
            checkpoint_episode=2000,
            seed=42,
        )


def test_checkpoint_model_path_for_unknown_policy(tmp_path):
    path = checkpoint_model_path(
        training_run_directory=tmp_path,
        policy_type="unknown",
        seed=42,
        checkpoint_episode=2000,
    )

    assert path == (
        tmp_path
        / "seed_42"
        / "single_policy"
        / "checkpoints"
        / "single_policy_episodes_2000_seed_42.pkl"
    )


def test_checkpoint_model_path_for_specialist_policy(tmp_path):
    path = checkpoint_model_path(
        training_run_directory=tmp_path,
        policy_type="aggressive",
        seed=456,
        checkpoint_episode=1500,
    )

    assert path == (
        tmp_path
        / "seed_456"
        / "specialist_aggressive"
        / "checkpoints"
        / "specialist_aggressive_episodes_1500_seed_456.pkl"
    )


def test_normalize_q_table_from_list_values():
    raw = {
        (0, 4, 0, 0, 3, 3, 0): [1, 2, 3],
    }

    normalized = normalize_q_table(raw)

    assert normalized[(0, 4, 0, 0, 3, 3, 0)] == [1.0, 2.0, 3.0]


def test_normalize_q_table_from_dict_values():
    raw = {
        (0, 4, 0, 0, 3, 3, 0): {
            0: 1,
            2: 3,
        },
    }

    normalized = normalize_q_table(raw)

    assert normalized[(0, 4, 0, 0, 3, 3, 0)] == [1.0, 0.0, 3.0]


def test_load_q_table_from_payload_dict(tmp_path):
    path = tmp_path / "model.pkl"

    payload = {
        "q_table": {
            (0, 4, 0, 0, 3, 3, 0): [1, 2, 3],
        }
    }

    with path.open("wb") as file:
        pickle.dump(payload, file)

    q_table = load_q_table(path)

    assert q_table[(0, 4, 0, 0, 3, 3, 0)] == [1.0, 2.0, 3.0]


def test_build_selected_targets_uses_all_requested_seeds_and_policies(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42, 123],
        policies=["unknown", "calling"],
    )

    names = [
        target.name
        for target in targets
    ]

    assert names == [
        "policy_general_seed_42_cp_2000",
        "policy_calling_seed_42_cp_2000",
        "policy_general_seed_123_cp_2000",
        "policy_calling_seed_123_cp_2000",
    ]

    assert targets[0].path == (
        tmp_path
        / "seed_42"
        / "single_policy"
        / "checkpoints"
        / "single_policy_episodes_2000_seed_42.pkl"
    )

    assert targets[1].path == (
        tmp_path
        / "seed_42"
        / "specialist_calling"
        / "checkpoints"
        / "specialist_calling_episodes_2000_seed_42.pkl"
    )


def test_build_selected_targets_rejects_unsupported_policy(tmp_path):
    with pytest.raises(ValueError, match="Unsupported policy type"):
        build_selected_targets(
            training_run_directory=tmp_path,
            checkpoint_episode=2000,
            seeds=[42],
            policies=["unknown", "loose"],
        )


def test_build_selected_targets_rejects_negative_seed(tmp_path):
    with pytest.raises(ValueError, match="seed must be non-negative"):
        build_selected_targets(
            training_run_directory=tmp_path,
            checkpoint_episode=2000,
            seeds=[-1],
            policies=["unknown"],
        )


def test_validate_targets_exist_raises_for_missing_files(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown"],
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing Q-table model files",
    ):
        validate_targets_exist(targets)


def test_summarize_q_table_counts_best_actions(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown"],
    )
    target = targets[0]

    q_table = {
        (0, 0, 0, 0, 0, 0, 0): [1.0, 0.0, 0.0],
        (0, 0, 0, 0, 0, 0, 1): [0.0, 2.0, 0.0],
        (0, 0, 0, 0, 0, 0, 2): [0.0, 0.0, 3.0],
        (0, 0, 0, 0, 0, 0, 3): [0.0, 0.0, 0.0],
    }

    summary = summarize_q_table(
        target=target,
        q_table=q_table,
    )

    assert summary.name == "policy_general_seed_42_cp_2000"
    assert summary.policy_type == "unknown"
    assert summary.seed == 42
    assert summary.checkpoint_episode == 2000
    assert summary.states == 4
    assert summary.fully_zero_states == 1
    assert summary.best_action_counts["fold"] == 2
    assert summary.best_action_counts["call"] == 1
    assert summary.best_action_counts["raise"] == 1


def test_compare_q_tables_counts_action_agreement(tmp_path):
    left_target, right_target = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown", "calling"],
    )

    left = {
        ("a",): [1.0, 0.0, 0.0],
        ("b",): [0.0, 2.0, 0.0],
        ("c",): [0.0, 0.0, 3.0],
    }

    right = {
        ("a",): [2.0, 0.0, 0.0],
        ("b",): [0.0, 0.0, 3.0],
        ("d",): [0.0, 5.0, 0.0],
    }

    comparison = compare_q_tables(
        left_target=left_target,
        left_q_table=left,
        right_target=right_target,
        right_q_table=right,
    )

    assert comparison.left_states == 3
    assert comparison.right_states == 3
    assert comparison.common_states == 2
    assert comparison.left_only_states == 1
    assert comparison.right_only_states == 1
    assert comparison.best_action_agreement == 1
    assert comparison.best_action_agreement_rate == pytest.approx(50.0)
    assert comparison.transition_counts["fold"]["fold"] == 1
    assert comparison.transition_counts["call"]["raise"] == 1
    assert comparison.left_policy_type == "unknown"
    assert comparison.right_policy_type == "calling"


def test_strip_opponent_type_removes_last_state_element():
    q_table = {
        (0, 4, 0, 0, 3, 3, 0): [1.0, 0.0, 0.0],
        (1, 2, 4, 1, 2, 2, 0): [0.0, 1.0, 0.0],
    }

    stripped = strip_opponent_type(q_table)

    assert stripped == {
        (0, 4, 0, 0, 3, 3): [1.0, 0.0, 0.0],
        (1, 2, 4, 1, 2, 2): [0.0, 1.0, 0.0],
    }


def test_compare_q_tables_after_stripping_opponent_type_has_common_states(
    tmp_path,
):
    left_target, right_target = build_selected_targets(
        training_run_directory=tmp_path,
        checkpoint_episode=2000,
        seeds=[42],
        policies=["unknown", "calling"],
    )

    unknown = {
        (0, 4, 0, 0, 3, 3, 0): [0.0, 1.0, 0.0],
    }

    calling = {
        (0, 4, 0, 0, 3, 3, 6): [0.0, 0.0, 1.0],
    }

    comparison = compare_q_tables(
        left_target=left_target,
        left_q_table=strip_opponent_type(unknown),
        right_target=right_target,
        right_q_table=strip_opponent_type(calling),
    )

    assert comparison.common_states == 1
    assert comparison.transition_counts["call"]["raise"] == 1
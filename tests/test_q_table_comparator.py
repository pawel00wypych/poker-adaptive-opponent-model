import pickle

import pytest

from src.evaluation.q_table_comparator import (
    best_action,
    build_selected_targets,
    compare_q_tables,
    is_fully_zero,
    is_tied_best,
    load_q_table,
    normalize_q_table,
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


def test_summarize_q_table_counts_best_actions(tmp_path):
    path = tmp_path / "model.pkl"

    q_table = {
        (0, 0, 0, 0, 0, 0, 0): [1.0, 0.0, 0.0],
        (0, 0, 0, 0, 0, 0, 1): [0.0, 2.0, 0.0],
        (0, 0, 0, 0, 0, 0, 2): [0.0, 0.0, 3.0],
        (0, 0, 0, 0, 0, 0, 3): [0.0, 0.0, 0.0],
    }

    summary = summarize_q_table(
        name="test_model",
        path=path,
        q_table=q_table,
    )

    assert summary.states == 4
    assert summary.fully_zero_states == 1
    assert summary.best_action_counts["fold"] == 2
    assert summary.best_action_counts["call"] == 1
    assert summary.best_action_counts["raise"] == 1


def test_compare_q_tables_counts_action_agreement():
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
        left_name="left",
        left_q_table=left,
        right_name="right",
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


def test_build_selected_targets_uses_expected_paths(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        unknown_checkpoint_episode=2000,
        calling_checkpoint_episode=2000,
        unknown_seeds=[42, 456],
        calling_targets=[
            (42, 2000),
            (456, 2000),
            (123, 1500),
        ],
    )

    names = [
        target.name
        for target in targets
    ]

    assert names == [
        "policy_unknown_seed_42_cp_2000",
        "policy_unknown_seed_456_cp_2000",
        "policy_calling_seed_42_cp_2000",
        "policy_calling_seed_456_cp_2000",
        "policy_calling_seed_123_cp_1500",
    ]

    assert targets[0].path == (
        tmp_path
        / "seed_42"
        / "single_policy"
        / "checkpoints"
        / "single_policy_episodes_2000_seed_42.pkl"
    )

    assert targets[4].path == (
        tmp_path
        / "seed_123"
        / "specialist_calling"
        / "checkpoints"
        / "specialist_calling_episodes_1500_seed_123.pkl"
    )


def test_validate_targets_exist_raises_for_missing_files(tmp_path):
    targets = build_selected_targets(
        training_run_directory=tmp_path,
        unknown_checkpoint_episode=2000,
        calling_checkpoint_episode=2000,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing Q-table model files",
    ):
        validate_targets_exist(targets)

from src.evaluation.q_table_comparator import strip_opponent_type


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


def test_compare_q_tables_after_stripping_opponent_type_has_common_states():
    unknown = {
        (0, 4, 0, 0, 3, 3, 0): [0.0, 1.0, 0.0],
    }

    calling = {
        (0, 4, 0, 0, 3, 3, 6): [0.0, 0.0, 1.0],
    }

    comparison = compare_q_tables(
        left_name="unknown",
        left_q_table=strip_opponent_type(unknown),
        right_name="calling",
        right_q_table=strip_opponent_type(calling),
    )

    assert comparison.common_states == 1
    assert comparison.transition_counts["call"]["raise"] == 1
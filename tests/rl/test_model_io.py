import pickle

import pytest

from src.rl.model_io import (
    load_model_metadata,
    load_model_payload,
    save_model_payload,
)


def test_model_io_saves_and_loads_payload(tmp_path):
    path = tmp_path / "models" / "agent.pkl"
    payload = {
        "algorithm": "test_algorithm",
        "metadata": {"seed": 42},
    }

    save_model_payload(
        path=str(path),
        payload=payload,
    )

    assert load_model_payload(
        path=str(path),
        model_name="Test",
    ) == payload


def test_model_io_loads_metadata(tmp_path):
    path = tmp_path / "agent.pkl"
    save_model_payload(
        path=str(path),
        payload={"metadata": {"run": "abc"}},
    )

    assert load_model_metadata(
        path=str(path),
        model_name="Test",
    ) == {"run": "abc"}


def test_model_io_returns_empty_metadata_for_legacy_non_dict_payload(tmp_path):
    path = tmp_path / "legacy.pkl"

    with path.open("wb") as file:
        pickle.dump(["legacy"], file)

    assert load_model_metadata(
        path=str(path),
        model_name="Test",
    ) == {}


def test_model_io_rejects_non_dict_payload(tmp_path):
    path = tmp_path / "legacy.pkl"

    with path.open("wb") as file:
        pickle.dump(["legacy"], file)

    with pytest.raises(
        TypeError,
        match="Unsupported Test model payload",
    ):
        load_model_payload(
            path=str(path),
            model_name="Test",
        )


def test_model_io_missing_file_raises_clear_error(tmp_path):
    path = tmp_path / "missing.pkl"

    with pytest.raises(
        FileNotFoundError,
        match="Test model does not exist",
    ):
        load_model_payload(
            path=str(path),
            model_name="Test",
        )


def test_load_model_payload_accepts_matching_algorithm(tmp_path):
    path = tmp_path / "agent.pkl"
    save_model_payload(
        path=str(path),
        payload={"algorithm": "q_learning"},
    )

    payload = load_model_payload(
        path=str(path),
        model_name="Q-learning",
        expected_algorithm="q_learning",
    )

    assert payload["algorithm"] == "q_learning"


def test_load_model_payload_rejects_mismatched_algorithm(tmp_path):
    path = tmp_path / "agent.pkl"
    save_model_payload(
        path=str(path),
        payload={"algorithm": "first_visit_monte_carlo_control"},
    )

    with pytest.raises(ValueError) as error:
        load_model_payload(
            path=str(path),
            model_name="Q-learning",
            expected_algorithm="q_learning",
        )

    message = str(error.value)
    assert "expected 'q_learning'" in message
    assert "found 'first_visit_monte_carlo_control'" in message
    assert "agent.pkl" in message


def test_load_model_payload_rejects_payload_without_algorithm_key(tmp_path):
    path = tmp_path / "agent.pkl"
    save_model_payload(
        path=str(path),
        payload={"q_table": {}},
    )

    with pytest.raises(
        ValueError,
        match="does not declare the algorithm",
    ):
        load_model_payload(
            path=str(path),
            model_name="SARSA",
            expected_algorithm="sarsa",
        )


def test_load_model_payload_skips_algorithm_check_when_not_requested(tmp_path):
    path = tmp_path / "agent.pkl"
    save_model_payload(
        path=str(path),
        payload={"algorithm": "anything"},
    )

    assert load_model_payload(
        path=str(path),
        model_name="Test",
    ) == {"algorithm": "anything"}

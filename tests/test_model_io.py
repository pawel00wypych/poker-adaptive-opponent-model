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

import pickle
from pathlib import Path
from typing import Any


def save_model_payload(
    path: str,
    payload: dict[str, Any],
) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with model_path.open("wb") as file:
        pickle.dump(payload, file)


def load_model_payload(
    path: str,
    model_name: str,
) -> dict[str, Any]:
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model does not exist: {model_path}"
        )

    with model_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Unsupported {model_name} model payload: "
            f"{type(payload)}"
        )

    return payload


def load_model_metadata(
    path: str,
    model_name: str,
) -> dict:
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model does not exist: {model_path}"
        )

    with model_path.open("rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        return {}

    return payload.get("metadata", {})

import pickle
from pathlib import Path
from typing import Any

from src.rl.constants import ALGORITHM_KEY


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


def validate_payload_algorithm(
    payload: dict[str, Any],
    *,
    expected_algorithm: str,
    model_name: str,
    path: str | Path,
) -> None:
    """Fail when a payload was produced by a different algorithm.

    Every agent writes its algorithm identifier into the payload. Without this
    check, pointing an evaluation run at another algorithm's directory loads
    successfully and silently attributes those results to the wrong algorithm.
    """
    found_algorithm = payload.get(ALGORITHM_KEY)

    if found_algorithm is None:
        raise ValueError(
            f"{model_name} model does not declare the algorithm that produced "
            f"it (missing {ALGORITHM_KEY!r} key): {path}. The file is either "
            "corrupted or was written by an older, unsupported version."
        )

    if found_algorithm != expected_algorithm:
        raise ValueError(
            f"{model_name} model was produced by a different algorithm: "
            f"expected {expected_algorithm!r}, found {found_algorithm!r} in "
            f"{path}. Check that the training-run directory passed on the "
            "command line matches the algorithm being evaluated."
        )


def load_model_payload(
    path: str,
    model_name: str,
    expected_algorithm: str | None = None,
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

    if expected_algorithm is not None:
        validate_payload_algorithm(
            payload,
            expected_algorithm=expected_algorithm,
            model_name=model_name,
            path=model_path,
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

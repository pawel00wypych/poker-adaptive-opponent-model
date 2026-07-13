import json
from pathlib import Path
from typing import Any


def save_json(
    path: str | Path,
    data: dict[str, Any],
) -> None:
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
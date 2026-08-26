import json
from pathlib import Path
from typing import Any

from src.experiment_protocol import (
    ProtocolProvenance,
    protocol_metadata,
)


def add_protocol_metadata(
    metadata: dict[str, Any],
    provenance: ProtocolProvenance | None,
) -> dict[str, Any]:
    if provenance is None:
        return metadata
    return {
        **metadata,
        **protocol_metadata(provenance),
    }


def save_protocol_snapshot(
    directory: str | Path,
    provenance: ProtocolProvenance,
) -> Path:
    output_path = Path(directory) / "experiment_config.json"
    save_json(output_path, protocol_metadata(provenance))
    return output_path


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

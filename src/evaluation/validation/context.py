from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.algorithm_metadata import AlgorithmValidationSpec


@dataclass(frozen=True)
class EvaluationManifest:
    path: Path
    values: dict[str, object]

    @classmethod
    def load_for_csv(cls, input_path: str | Path) -> EvaluationManifest | None:
        summary_path = Path(input_path).with_suffix(".summary.json")
        if not summary_path.exists():
            return None
        with summary_path.open(encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict):
            raise TypeError(f"Evaluation summary must be an object: {summary_path}")
        return cls(path=summary_path, values=values)

    def positive_int(self, key: str) -> int | None:
        value = self.values.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value


@dataclass(frozen=True)
class EvaluationContext:
    input_path: Path
    validation_mode: str
    raw_games: pd.DataFrame
    aggregated: pd.DataFrame
    seed_rows: pd.DataFrame | None
    replicate_rows: pd.DataFrame | None
    manifest: EvaluationManifest | None
    algorithm_specs: tuple[AlgorithmValidationSpec, ...]
    selected_training_episode: int | None
    model_selection: str

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

from src.evaluation.constants import (
    ACTION_LABELS,
    CHECKPOINT_PREFIXES,
    MODEL_DIRECTORIES,
    STATE_V2_FIELDS,
    SUPPORTED_POLICY_TYPES,
)


@dataclass(frozen=True)
class QTableTarget:
    name: str
    path: Path
    policy_type: str
    seed: int
    checkpoint_episode: int


@dataclass(frozen=True)
class ActionStats:
    action: str
    mean_q: float
    median_q: float
    std_q: float
    min_q: float
    max_q: float
    zero_count: int
    zero_rate: float


@dataclass(frozen=True)
class QTableSummary:
    name: str
    path: str
    policy_type: str
    seed: int
    checkpoint_episode: int
    states: int
    fully_zero_states: int
    fully_zero_rate: float
    tied_best_states: int
    tied_best_rate: float
    best_action_counts: dict[str, int]
    best_action_rates: dict[str, float]
    action_stats: list[ActionStats]


@dataclass(frozen=True)
class PairwiseComparison:
    left_name: str
    right_name: str
    left_policy_type: str
    right_policy_type: str
    left_seed: int
    right_seed: int
    left_checkpoint_episode: int
    right_checkpoint_episode: int
    left_states: int
    right_states: int
    common_states: int
    left_only_states: int
    right_only_states: int
    best_action_agreement: int
    best_action_agreement_rate: float
    transition_counts: dict[str, dict[str, int]]
    mean_abs_q_delta_by_action: dict[str, float]
    mean_max_abs_q_delta: float


def checkpoint_filename(
    policy_type: str,
    checkpoint_episode: int,
    seed: int,
) -> str:
    validate_policy_type(policy_type)

    prefix = CHECKPOINT_PREFIXES[policy_type]

    return (
        f"{prefix}"
        f"_episodes_{checkpoint_episode}"
        f"_seed_{seed}.pkl"
    )


def checkpoint_model_path(
    training_run_directory: str | Path,
    policy_type: str,
    seed: int,
    checkpoint_episode: int,
) -> Path:
    validate_policy_type(policy_type)

    root = Path(training_run_directory)

    return (
        root
        / f"seed_{seed}"
        / MODEL_DIRECTORIES[policy_type]
        / "checkpoints"
        / checkpoint_filename(
            policy_type=policy_type,
            checkpoint_episode=checkpoint_episode,
            seed=seed,
        )
    )


def public_policy_label(policy_type: str) -> str:
    """
    Return the public report label for a persisted policy type.

    Internally, the general fixed policy is still stored as policy_type
    "unknown" because it uses the unknown/general opponent-state encoding.
    Public reports should call it "general" to avoid suggesting that this
    is an unknown-behaviour agent.
    """
    validate_policy_type(policy_type)

    if policy_type == "unknown":
        return "general"

    return policy_type


def target_name(
    policy_type: str,
    seed: int,
    checkpoint_episode: int,
) -> str:
    return (
        f"policy_{public_policy_label(policy_type)}"
        f"_seed_{seed}"
        f"_cp_{checkpoint_episode}"
    )


def validate_policy_type(policy_type: str) -> None:
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            "Unsupported policy type: "
            f"{policy_type}. "
            f"Supported values: {list(SUPPORTED_POLICY_TYPES)}"
        )


def build_selected_targets(
    training_run_directory: str | Path,
    checkpoint_episode: int,
    seeds: Iterable[int],
    policies: Iterable[str],
) -> list[QTableTarget]:
    targets: list[QTableTarget] = []

    for policy_type in policies:
        validate_policy_type(policy_type)

    for seed in seeds:
        if seed < 0:
            raise ValueError("seed must be non-negative")

        for policy_type in policies:
            targets.append(
                QTableTarget(
                    name=target_name(
                        policy_type=policy_type,
                        seed=seed,
                        checkpoint_episode=checkpoint_episode,
                    ),
                    path=checkpoint_model_path(
                        training_run_directory=training_run_directory,
                        policy_type=policy_type,
                        seed=seed,
                        checkpoint_episode=checkpoint_episode,
                    ),
                    policy_type=policy_type,
                    seed=seed,
                    checkpoint_episode=checkpoint_episode,
                )
            )

    return targets


def strip_opponent_type(
    q_table: dict[tuple, list[float]],
) -> dict[tuple, list[float]]:
    stripped = {}

    for state, q_values in q_table.items():
        if len(state) < 2:
            stripped[state] = q_values
        else:
            stripped[state[:-1]] = q_values

    return stripped


def load_q_table(path: str | Path) -> dict[tuple, list[float]]:
    model_path = Path(path)

    with model_path.open("rb") as file:
        payload = pickle.load(file)

    if hasattr(payload, "q_table"):
        raw_q_table = payload.q_table
    elif isinstance(payload, dict) and "q_table" in payload:
        raw_q_table = payload["q_table"]
    elif isinstance(payload, dict):
        raw_q_table = payload
    else:
        raise TypeError(
            f"Unsupported model payload type: {type(payload)}"
        )

    return normalize_q_table(raw_q_table)


def normalize_q_table(raw_q_table: dict[Any, Any]) -> dict[tuple, list[float]]:
    normalized: dict[tuple, list[float]] = {}

    for raw_state, raw_values in raw_q_table.items():
        state = tuple(raw_state)

        if isinstance(raw_values, dict):
            q_values = [
                float(raw_values.get(action_id, 0.0))
                for action_id in ACTION_LABELS
            ]
        else:
            q_values = [
                float(value)
                for value in list(raw_values)
            ]

            if len(q_values) < len(ACTION_LABELS):
                q_values = q_values + [0.0] * (
                    len(ACTION_LABELS) - len(q_values)
                )

            if len(q_values) > len(ACTION_LABELS):
                q_values = q_values[: len(ACTION_LABELS)]

        normalized[state] = q_values

    return normalized


def best_action(q_values: list[float]) -> int:
    return max(
        ACTION_LABELS.keys(),
        key=lambda action_id: q_values[action_id],
    )


def is_fully_zero(q_values: list[float], tolerance: float = 1e-12) -> bool:
    return all(
        abs(value) <= tolerance
        for value in q_values
    )


def is_tied_best(q_values: list[float], tolerance: float = 1e-12) -> bool:
    best = max(q_values)

    tied = [
        value
        for value in q_values
        if abs(value - best) <= tolerance
    ]

    return len(tied) > 1


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator * 100


def describe_state(state: tuple) -> dict[str, Any]:
    return {
        field: state[index] if index < len(state) else None
        for index, field in enumerate(STATE_V2_FIELDS)
    }


def summarize_q_table(
    target: QTableTarget,
    q_table: dict[tuple, list[float]],
) -> QTableSummary:
    states = len(q_table)

    best_action_counts = {
        label: 0
        for label in ACTION_LABELS.values()
    }

    fully_zero_states = 0
    tied_best_states = 0

    values_by_action = {
        action_id: []
        for action_id in ACTION_LABELS
    }

    for q_values in q_table.values():
        selected_action = best_action(q_values)
        selected_label = ACTION_LABELS[selected_action]
        best_action_counts[selected_label] += 1

        if is_fully_zero(q_values):
            fully_zero_states += 1

        if is_tied_best(q_values):
            tied_best_states += 1

        for action_id in ACTION_LABELS:
            values_by_action[action_id].append(q_values[action_id])

    best_action_rates = {
        action: safe_rate(count, states)
        for action, count in best_action_counts.items()
    }

    action_stats = []

    for action_id, values in values_by_action.items():
        if values:
            zero_count = sum(
                1
                for value in values
                if abs(value) <= 1e-12
            )

            action_stats.append(
                ActionStats(
                    action=ACTION_LABELS[action_id],
                    mean_q=mean(values),
                    median_q=median(values),
                    std_q=pstdev(values),
                    min_q=min(values),
                    max_q=max(values),
                    zero_count=zero_count,
                    zero_rate=safe_rate(zero_count, len(values)),
                )
            )
        else:
            action_stats.append(
                ActionStats(
                    action=ACTION_LABELS[action_id],
                    mean_q=0.0,
                    median_q=0.0,
                    std_q=0.0,
                    min_q=0.0,
                    max_q=0.0,
                    zero_count=0,
                    zero_rate=0.0,
                )
            )

    return QTableSummary(
        name=target.name,
        path=str(target.path),
        policy_type=target.policy_type,
        seed=target.seed,
        checkpoint_episode=target.checkpoint_episode,
        states=states,
        fully_zero_states=fully_zero_states,
        fully_zero_rate=safe_rate(fully_zero_states, states),
        tied_best_states=tied_best_states,
        tied_best_rate=safe_rate(tied_best_states, states),
        best_action_counts=best_action_counts,
        best_action_rates=best_action_rates,
        action_stats=action_stats,
    )


def compare_q_tables(
    left_target: QTableTarget,
    left_q_table: dict[tuple, list[float]],
    right_target: QTableTarget,
    right_q_table: dict[tuple, list[float]],
) -> PairwiseComparison:
    left_states = set(left_q_table.keys())
    right_states = set(right_q_table.keys())
    common_states = sorted(left_states & right_states)

    transition_counts = {
        left_label: {
            right_label: 0
            for right_label in ACTION_LABELS.values()
        }
        for left_label in ACTION_LABELS.values()
    }

    agreement = 0

    abs_delta_by_action = {
        action_id: []
        for action_id in ACTION_LABELS
    }

    max_abs_deltas = []

    for state in common_states:
        left_values = left_q_table[state]
        right_values = right_q_table[state]

        left_action = best_action(left_values)
        right_action = best_action(right_values)

        left_label = ACTION_LABELS[left_action]
        right_label = ACTION_LABELS[right_action]

        transition_counts[left_label][right_label] += 1

        if left_action == right_action:
            agreement += 1

        state_deltas = []

        for action_id in ACTION_LABELS:
            delta = abs(left_values[action_id] - right_values[action_id])
            abs_delta_by_action[action_id].append(delta)
            state_deltas.append(delta)

        max_abs_deltas.append(max(state_deltas))

    mean_abs_q_delta_by_action = {
        ACTION_LABELS[action_id]: (
            mean(values)
            if values
            else 0.0
        )
        for action_id, values in abs_delta_by_action.items()
    }

    return PairwiseComparison(
        left_name=left_target.name,
        right_name=right_target.name,
        left_policy_type=left_target.policy_type,
        right_policy_type=right_target.policy_type,
        left_seed=left_target.seed,
        right_seed=right_target.seed,
        left_checkpoint_episode=left_target.checkpoint_episode,
        right_checkpoint_episode=right_target.checkpoint_episode,
        left_states=len(left_states),
        right_states=len(right_states),
        common_states=len(common_states),
        left_only_states=len(left_states - right_states),
        right_only_states=len(right_states - left_states),
        best_action_agreement=agreement,
        best_action_agreement_rate=safe_rate(
            agreement,
            len(common_states),
        ),
        transition_counts=transition_counts,
        mean_abs_q_delta_by_action=mean_abs_q_delta_by_action,
        mean_max_abs_q_delta=mean(max_abs_deltas) if max_abs_deltas else 0.0,
    )


def find_largest_disagreements(
    left_target: QTableTarget,
    left_q_table: dict[tuple, list[float]],
    right_target: QTableTarget,
    right_q_table: dict[tuple, list[float]],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    common_states = sorted(
        set(left_q_table.keys()) & set(right_q_table.keys())
    )

    rows = []

    for state in common_states:
        left_values = left_q_table[state]
        right_values = right_q_table[state]

        left_action = best_action(left_values)
        right_action = best_action(right_values)

        max_abs_delta = max(
            abs(left_values[action_id] - right_values[action_id])
            for action_id in ACTION_LABELS
        )

        rows.append(
            {
                "state": state,
                "state_description": describe_state(state),
                "left_name": left_target.name,
                "right_name": right_target.name,
                "left_policy_type": left_target.policy_type,
                "right_policy_type": right_target.policy_type,
                "left_seed": left_target.seed,
                "right_seed": right_target.seed,
                "left_checkpoint_episode": left_target.checkpoint_episode,
                "right_checkpoint_episode": right_target.checkpoint_episode,
                "left_q_values": left_values,
                "right_q_values": right_values,
                "left_best_action": ACTION_LABELS[left_action],
                "right_best_action": ACTION_LABELS[right_action],
                "same_best_action": left_action == right_action,
                "max_abs_q_delta": max_abs_delta,
            }
        )

    rows.sort(
        key=lambda row: row["max_abs_q_delta"],
        reverse=True,
    )

    return rows[:top_n]


def validate_targets_exist(targets: Iterable[QTableTarget]) -> None:
    missing = [
        target
        for target in targets
        if not target.path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"{target.name}: {target.path}"
            for target in missing
        )

        raise FileNotFoundError(
            "Missing Q-table model files:\n"
            f"{formatted}"
        )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, list):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def save_json(
    path: str | Path,
    data: Any,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            to_jsonable(data),
            file,
            indent=2,
            ensure_ascii=False,
        )
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.experiment_protocol import (
    FINAL_EXPERIMENT_CONFIG,
    HEAD_TO_HEAD_EVALUATION,
    experiment_config_hash_from_snapshot,
    training_config_hash_from_snapshot,
)
from src.evaluation.validation.config import IntegrityRequirements
from src.evaluation.validation.context import EvaluationManifest
from src.evaluation.validation.models import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    CheckKind,
    ValidationCheckResult,
)

BASELINE_SANITY_MODE = "baseline-sanity"
HEAD_TO_HEAD_MODE = "head-to-head"

COMMON_REQUIRED_COLUMNS = {
    "agent_name",
    "opponent_name",
    "experiment_name",
    "game_id",
    "matchup_game_index",
    "evaluation_seed",
    "final_stack",
    "initial_stack",
    "profit",
    "profit_bb",
    "hands_played",
    "won_game",
    "busted",
    "ended_by_bust",
    "ended_by_round_limit",
    "classified_decisions",
    "correct_classifications",
    "incorrect_classifications",
    "unknown_classifications",
    "other_classifications",
    "classifier_accuracy",
    "classifier_coverage",
    "policy_switches",
    "policy_decisions",
    "unseen_state_decisions",
    "untried_action_selections",
    "unseen_state_decision_rate",
    "untried_action_selection_rate",
}

MODEL_REQUIRED_COLUMNS = {
    "model_seed",
    "model_source",
    "training_episode",
}

BASELINE_REQUIRED_COLUMNS = {
    "evaluation_replicate_id",
}

FINITE_NUMERIC_COLUMNS = {
    "game_id",
    "matchup_game_index",
    "evaluation_seed",
    "final_stack",
    "initial_stack",
    "profit",
    "profit_bb",
    "hands_played",
    "won_game",
    "busted",
    "ended_by_bust",
    "ended_by_round_limit",
    "classified_decisions",
    "correct_classifications",
    "incorrect_classifications",
    "unknown_classifications",
    "other_classifications",
    "classifier_accuracy",
    "classifier_coverage",
    "policy_switches",
    "policy_decisions",
    "unseen_state_decisions",
    "untried_action_selections",
    "unseen_state_decision_rate",
    "untried_action_selection_rate",
}

BINARY_COLUMNS = {
    "won_game",
    "busted",
    "ended_by_bust",
    "ended_by_round_limit",
}

NON_NEGATIVE_COLUMNS = {
    "final_stack",
    "initial_stack",
    "hands_played",
    "classified_decisions",
    "correct_classifications",
    "incorrect_classifications",
    "unknown_classifications",
    "other_classifications",
    "policy_switches",
    "policy_decisions",
    "unseen_state_decisions",
    "untried_action_selections",
}

UNIT_INTERVAL_COLUMNS = {
    "classifier_accuracy",
    "classifier_coverage",
    "unseen_state_decision_rate",
    "untried_action_selection_rate",
}


def _replicate_rows(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> pd.DataFrame:
    if validation_mode == BASELINE_SANITY_MODE:
        return raw_games
    if (
        validation_mode == HEAD_TO_HEAD_MODE
        and "evaluation_replicate_id" in raw_games.columns
    ):
        return raw_games[raw_games["evaluation_replicate_id"].notna()]
    return raw_games.iloc[0:0]


def _model_rows(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> pd.DataFrame:
    if validation_mode == BASELINE_SANITY_MODE:
        return raw_games.iloc[0:0]
    replicate_indices = _replicate_rows(raw_games, validation_mode).index
    return raw_games.drop(index=replicate_indices)


def _integrity_result(
    *,
    check_id: str,
    check_name: str,
    status: str,
    message: str,
    category: str,
    observed_value: float | None = None,
    threshold: float | None = None,
    details: dict[str, object] | None = None,
) -> ValidationCheckResult:
    return ValidationCheckResult(
        check_id=check_id,
        check_name=check_name,
        check_type=CheckKind.INTEGRITY,
        status=status,
        message=message,
        category=category,
        observed_value=observed_value,
        threshold=threshold,
        details=details,
    )


def validate_required_columns(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    required = set(COMMON_REQUIRED_COLUMNS)
    if validation_mode == BASELINE_SANITY_MODE:
        required.update(BASELINE_REQUIRED_COLUMNS)
    else:
        required.update(MODEL_REQUIRED_COLUMNS)
        if validation_mode == HEAD_TO_HEAD_MODE:
            required.update(BASELINE_REQUIRED_COLUMNS)
    missing = sorted(required.difference(raw_games.columns))
    return [
        _integrity_result(
            check_id="required_columns",
            check_name="Required evaluation columns",
            status=STATUS_FAIL if missing else STATUS_PASS,
            category="schema_integrity",
            message=(
                f"Missing required columns: {missing}."
                if missing
                else "All required evaluation columns are present."
            ),
            details={
                "required_columns": sorted(required),
                "missing_columns": missing,
            },
        )
    ]


def validate_finite_numeric_values(
    raw_games: pd.DataFrame,
) -> list[ValidationCheckResult]:
    checked_columns = sorted(FINITE_NUMERIC_COLUMNS.intersection(raw_games.columns))
    invalid_by_column: dict[str, int] = {}
    for column in checked_columns:
        numeric = pd.to_numeric(raw_games[column], errors="coerce")
        invalid_count = int((numeric.isna() | ~np.isfinite(numeric)).sum())
        if invalid_count:
            invalid_by_column[column] = invalid_count

    return [
        _integrity_result(
            check_id="finite_numeric_values",
            check_name="Finite numeric evaluation values",
            status=STATUS_FAIL if invalid_by_column else STATUS_PASS,
            category="value_integrity",
            message=(
                f"Invalid numeric values found: {invalid_by_column}."
                if invalid_by_column
                else "All required numeric values are finite."
            ),
            details={
                "checked_columns": checked_columns,
                "invalid_counts": invalid_by_column,
            },
        )
    ]


def validate_numeric_domains(
    raw_games: pd.DataFrame,
) -> list[ValidationCheckResult]:
    invalid_by_column: dict[str, int] = {}
    for column in sorted(BINARY_COLUMNS.intersection(raw_games.columns)):
        numeric = pd.to_numeric(raw_games[column], errors="coerce")
        invalid = ~numeric.isin((0, 1))
        if invalid.any():
            invalid_by_column[column] = int(invalid.sum())
    for column in sorted(NON_NEGATIVE_COLUMNS.intersection(raw_games.columns)):
        numeric = pd.to_numeric(raw_games[column], errors="coerce")
        invalid = numeric.lt(0)
        if invalid.any():
            invalid_by_column[column] = int(invalid.sum())
    for column in sorted(UNIT_INTERVAL_COLUMNS.intersection(raw_games.columns)):
        numeric = pd.to_numeric(raw_games[column], errors="coerce")
        invalid = numeric.lt(0.0) | numeric.gt(1.0)
        if invalid.any():
            invalid_by_column[column] = int(invalid.sum())

    return [
        _integrity_result(
            check_id="numeric_value_domains",
            check_name="Numeric evaluation domains",
            status=STATUS_FAIL if invalid_by_column else STATUS_PASS,
            category="value_integrity",
            message=(
                f"Out-of-domain numeric values found: {invalid_by_column}."
                if invalid_by_column
                else "Binary, count, and rate values are in valid domains."
            ),
            details={"invalid_counts": invalid_by_column},
        )
    ]
def validate_game_identity_uniqueness(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    identity_definitions: list[tuple[pd.DataFrame, list[str]]] = []
    model_rows = _model_rows(raw_games, validation_mode)
    replicate_rows = _replicate_rows(raw_games, validation_mode)
    if not model_rows.empty:
        identity_definitions.append(
            (
                model_rows,
                [
                    "model_seed",
                    "agent_name",
                    "opponent_name",
                    "matchup_game_index",
                ],
            )
        )
    if not replicate_rows.empty:
        identity_definitions.append(
            (
                replicate_rows,
                [
                    "evaluation_replicate_id",
                    "agent_name",
                    "opponent_name",
                    "matchup_game_index",
                ],
            )
        )

    duplicate_count = 0
    checked_identities: list[list[str]] = []
    for rows, identity_columns in identity_definitions:
        if not set(identity_columns).issubset(rows.columns):
            continue
        checked_identities.append(identity_columns)
        duplicate_count += int(rows.duplicated(identity_columns, keep=False).sum())
    return [
        _integrity_result(
            check_id="unique_game_identity",
            check_name="Unique game identity",
            status=STATUS_FAIL if duplicate_count else STATUS_PASS,
            category="row_identity",
            message=(
                f"Found {duplicate_count} rows with duplicate game identity."
                if duplicate_count
                else "Every game identity is unique."
            ),
            observed_value=float(duplicate_count),
            threshold=0.0,
            details={
                "identity_columns": checked_identities,
                "duplicate_row_count": duplicate_count,
            },
        )
    ]


def validate_profit_consistency(
    raw_games: pd.DataFrame,
) -> list[ValidationCheckResult]:
    required = {"final_stack", "initial_stack", "profit"}
    if not required.issubset(raw_games.columns):
        return []

    final_stack = pd.to_numeric(raw_games["final_stack"], errors="coerce")
    initial_stack = pd.to_numeric(raw_games["initial_stack"], errors="coerce")
    profit = pd.to_numeric(raw_games["profit"], errors="coerce")
    inconsistent = ~(profit.eq(final_stack - initial_stack))
    inconsistent_count = int(inconsistent.sum())
    return [
        _integrity_result(
            check_id="profit_consistency",
            check_name="Profit and stack consistency",
            status=STATUS_FAIL if inconsistent_count else STATUS_PASS,
            category="value_integrity",
            message=(
                f"Found {inconsistent_count} rows where profit != final - initial."
                if inconsistent_count
                else "Profit equals final stack minus initial stack in every row."
            ),
            observed_value=float(inconsistent_count),
            threshold=0.0,
        )
    ]


def validate_final_model_rows(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    model_rows = _model_rows(raw_games, validation_mode)
    if model_rows.empty:
        return []

    invalid_sources = (
        sorted(
            {
                "<missing>" if pd.isna(value) else str(value)
                for value in model_rows["model_source"]
                if pd.isna(value) or str(value) != "final"
            }
        )
        if "model_source" in model_rows.columns
        else ["<missing column>"]
    )
    missing_training_episode = (
        int(model_rows["training_episode"].isna().sum())
        if "training_episode" in model_rows.columns
        else len(model_rows)
    )
    checkpoint_rows = (
        int(model_rows["checkpoint_episode"].notna().sum())
        if "checkpoint_episode" in model_rows.columns
        else 0
    )
    valid = not invalid_sources and not missing_training_episode and not checkpoint_rows
    return [
        _integrity_result(
            check_id="final_model_rows",
            check_name="Final-model row contract",
            status=STATUS_PASS if valid else STATUS_FAIL,
            category="model_source_integrity",
            message=(
                "All rows describe final models."
                if valid
                else "Evaluation contains invalid final-model metadata."
            ),
            details={
                "invalid_model_sources": invalid_sources,
                "missing_training_episode_rows": missing_training_episode,
                "checkpoint_rows": checkpoint_rows,
            },
        )
    ]


def validate_model_metadata_values(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    model_rows = _model_rows(raw_games, validation_mode)
    if model_rows.empty:
        return []

    invalid_by_column: dict[str, int] = {}
    for column, minimum in (("model_seed", 0), ("training_episode", 1)):
        if column not in model_rows.columns:
            invalid_by_column[column] = len(model_rows)
            continue
        numeric = pd.to_numeric(model_rows[column], errors="coerce")
        invalid = (
            numeric.isna()
            | ~np.isfinite(numeric)
            | numeric.lt(minimum)
            | numeric.ne(np.floor(numeric))
        )
        invalid_count = int(invalid.sum())
        if invalid_count:
            invalid_by_column[column] = invalid_count

    return [
        _integrity_result(
            check_id="model_metadata_values",
            check_name="Model metadata value contract",
            status=STATUS_FAIL if invalid_by_column else STATUS_PASS,
            category="model_source_integrity",
            message=(
                f"Invalid model metadata values found: {invalid_by_column}."
                if invalid_by_column
                else "Model seeds and training episodes are valid integers."
            ),
            details={"invalid_counts": invalid_by_column},
        )
    ]


def validate_replicate_metadata_values(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    replicate_rows = _replicate_rows(raw_games, validation_mode)
    if replicate_rows.empty:
        return []

    numeric = pd.to_numeric(
        replicate_rows["evaluation_replicate_id"],
        errors="coerce",
    )
    invalid_replicates = (
        numeric.isna()
        | ~np.isfinite(numeric)
        | numeric.lt(0)
        | numeric.ne(np.floor(numeric))
    )
    model_metadata_columns = [
        column
        for column in (
            "model_seed",
            "model_source",
            "training_episode",
            "checkpoint_episode",
        )
        if column in replicate_rows.columns
    ]
    legacy_model_rows = (
        int(replicate_rows[model_metadata_columns].notna().any(axis=1).sum())
        if model_metadata_columns
        else 0
    )
    invalid_count = int(invalid_replicates.sum())
    valid = not invalid_count and not legacy_model_rows

    return [
        _integrity_result(
            check_id="replicate_metadata_values",
            check_name="Evaluation-replicate metadata contract",
            status=STATUS_PASS if valid else STATUS_FAIL,
            category="replicate_metadata_integrity",
            message=(
                "Evaluation replicate identifiers are valid."
                if valid
                else "Invalid or legacy baseline replicate metadata was found."
            ),
            details={
                "invalid_evaluation_replicate_ids": invalid_count,
                "legacy_model_seed_rows": legacy_model_rows,
            },
        )
    ]


def validate_metadata_domain_exclusivity(
    raw_games: pd.DataFrame,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    if "evaluation_replicate_id" not in raw_games.columns:
        return []

    replicate_mask = raw_games["evaluation_replicate_id"].notna()
    model_metadata_columns = [
        column
        for column in ("model_seed", "model_source", "training_episode")
        if column in raw_games.columns
    ]
    model_mask = (
        raw_games[model_metadata_columns].notna().any(axis=1)
        if model_metadata_columns
        else pd.Series(False, index=raw_games.index)
    )
    mixed_domain_rows = int((replicate_mask & model_mask).sum())
    unexpected_replicate_rows = (
        int(replicate_mask.sum())
        if validation_mode not in {BASELINE_SANITY_MODE, HEAD_TO_HEAD_MODE}
        else 0
    )
    valid = not mixed_domain_rows and not unexpected_replicate_rows

    return [
        _integrity_result(
            check_id="metadata_domain_exclusivity",
            check_name="Model and replicate metadata domains",
            status=STATUS_PASS if valid else STATUS_FAIL,
            category="model_source_integrity",
            message=(
                "Model and replicate metadata are mutually exclusive."
                if valid
                else "Rows mix incompatible model and replicate metadata."
            ),
            details={
                "mixed_domain_rows": mixed_domain_rows,
                "unexpected_replicate_rows": unexpected_replicate_rows,
            },
        )
    ]


def _expected_seed_values(
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> tuple[int, ...] | None:
    if requirements.expected_model_seeds is not None:
        return requirements.expected_model_seeds
    if manifest is None:
        return None
    seeds = manifest.values.get("seeds", manifest.values.get("model_seeds"))
    if not isinstance(seeds, list):
        return None
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        return None
    return tuple(seeds)


def validate_expected_seed_sets(
    raw_games: pd.DataFrame,
    validation_mode: str,
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> list[ValidationCheckResult]:
    model_rows = _model_rows(raw_games, validation_mode)
    if model_rows.empty or "model_seed" not in model_rows:
        return []
    expected = _expected_seed_values(requirements, manifest)
    if expected is None:
        return []

    expected_set = set(expected)
    mismatches: list[dict[str, object]] = []
    for (agent_name, opponent_name), rows in model_rows.groupby(
        ["agent_name", "opponent_name"],
        dropna=False,
    ):
        numeric = pd.to_numeric(rows["model_seed"], errors="coerce")
        actual = {
            int(seed)
            for seed in numeric
            if np.isfinite(seed) and seed >= 0 and seed == np.floor(seed)
        }
        if actual != expected_set:
            mismatches.append(
                {
                    "agent_name": str(agent_name),
                    "opponent_name": str(opponent_name),
                    "missing_seeds": sorted(expected_set - actual),
                    "unexpected_seeds": sorted(actual - expected_set),
                }
            )

    return [
        _integrity_result(
            check_id="expected_model_seeds",
            check_name="Expected model-seed sets",
            status=STATUS_FAIL if mismatches else STATUS_PASS,
            category="seed_coverage",
            message=(
                f"{len(mismatches)} matchup(s) do not have the expected seed set."
                if mismatches
                else "Every matchup has the expected model-seed set."
            ),
            details={
                "expected_model_seeds": list(expected),
                "mismatches": mismatches,
            },
        )
    ]


def _expected_games(
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> int | None:
    if requirements.expected_games_per_matchup is not None:
        return requirements.expected_games_per_matchup
    return manifest.positive_int("games") if manifest is not None else None


def validate_games_per_matchup(
    raw_games: pd.DataFrame,
    validation_mode: str,
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> list[ValidationCheckResult]:
    expected = _expected_games(requirements, manifest)
    if expected is None:
        return []

    group_definitions: list[tuple[pd.DataFrame, list[str]]] = []
    model_rows = _model_rows(raw_games, validation_mode)
    replicate_rows = _replicate_rows(raw_games, validation_mode)
    if not model_rows.empty:
        group_definitions.append(
            (model_rows, ["model_seed", "agent_name", "opponent_name"])
        )
    if not replicate_rows.empty:
        group_definitions.append(
            (
                replicate_rows,
                ["evaluation_replicate_id", "agent_name", "opponent_name"],
            )
        )

    mismatches: list[dict[str, object]] = []
    for rows, group_columns in group_definitions:
        if not set(group_columns).issubset(rows.columns):
            continue
        counts = rows.groupby(group_columns, dropna=False).size()
        invalid = counts[counts != expected]
        for key, count in invalid.items():
            key_values = key if isinstance(key, tuple) else (key,)
            identity = {
                column: (
                    value.item() if isinstance(value, np.generic) else value
                )
                for column, value in zip(
                    group_columns,
                    key_values,
                    strict=True,
                )
            }
            mismatches.append(
                {
                    **identity,
                    "observed_games": int(count),
                    "expected_games": expected,
                }
            )
    return [
        _integrity_result(
            check_id="games_per_matchup",
            check_name="Games per matchup",
            status=STATUS_FAIL if mismatches else STATUS_PASS,
            category="game_count_integrity",
            message=(
                f"{len(mismatches)} matchup block(s) have an invalid game count."
                if mismatches
                else f"Every matchup block contains {expected} games."
            ),
            observed_value=float(len(mismatches)),
            threshold=0.0,
            details={
                "expected_games_per_matchup": expected,
                "mismatches": mismatches,
            },
        )
    ]


def validate_expected_evaluation_replicates(
    raw_games: pd.DataFrame,
    validation_mode: str,
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> list[ValidationCheckResult]:
    replicate_rows = _replicate_rows(raw_games, validation_mode)
    if replicate_rows.empty or "evaluation_replicate_id" not in replicate_rows:
        return []
    expected = requirements.expected_evaluation_replicates
    if expected is None and manifest is not None:
        expected = manifest.positive_int("evaluation_replicates")
    minimum = requirements.min_evaluation_replicates_per_matchup

    mismatches: list[dict[str, object]] = []
    for (agent_name, opponent_name), rows in replicate_rows.groupby(
        ["agent_name", "opponent_name"],
        dropna=False,
    ):
        actual = int(rows["evaluation_replicate_id"].nunique())
        if actual < minimum or (expected is not None and actual != expected):
            mismatches.append(
                {
                    "agent_name": str(agent_name),
                    "opponent_name": str(opponent_name),
                    "observed_replicates": actual,
                    "expected_replicates": expected,
                    "minimum_replicates": minimum,
                }
            )

    return [
        _integrity_result(
            check_id="expected_evaluation_replicates",
            check_name="Expected evaluation-replicate coverage",
            status=STATUS_FAIL if mismatches else STATUS_PASS,
            category="evaluation_replicate_integrity",
            message=(
                f"{len(mismatches)} matchup(s) have an invalid replicate count."
                if mismatches
                else (
                    f"Every baseline matchup has at least {minimum} replicates"
                    + (
                        f" and exactly {expected} as required."
                        if expected is not None
                        else "."
                    )
                )
            ),
            details={
                "expected_evaluation_replicates": expected,
                "minimum_evaluation_replicates": minimum,
                "mismatches": mismatches,
            },
        )
    ]


def validate_manifest(
    raw_games: pd.DataFrame,
    manifest: EvaluationManifest | None,
    requirements: IntegrityRequirements,
    validation_mode: str,
) -> list[ValidationCheckResult]:
    if manifest is None:
        return [
            _integrity_result(
                check_id="evaluation_manifest",
                check_name="Evaluation summary manifest",
                status=STATUS_FAIL if requirements.require_manifest else STATUS_SKIPPED,
                category="manifest_integrity",
                message="Companion .summary.json was not found.",
            )
        ]

    manifest_rows = manifest.positive_int("row_count")
    manifest_games = manifest.positive_int("games")
    errors: list[str] = []
    if manifest_rows is None:
        errors.append("row_count must be a positive integer")
    elif manifest_rows != len(raw_games):
        errors.append(
            f"row_count={manifest_rows} does not match CSV rows={len(raw_games)}"
        )
    if manifest_games is None:
        errors.append("games must be a positive integer")
    elif (
        requirements.expected_games_per_matchup is not None
        and manifest_games != requirements.expected_games_per_matchup
    ):
        errors.append(
            f"games={manifest_games} does not match required "
            f"{requirements.expected_games_per_matchup}"
        )

    model_rows = _model_rows(raw_games, validation_mode)
    seed_key = (
        "seeds"
        if "seeds" in manifest.values
        else "model_seeds" if "model_seeds" in manifest.values else None
    )
    manifest_seeds = manifest.values.get(seed_key) if seed_key is not None else None
    if not model_rows.empty and seed_key is None and requirements.require_manifest:
        errors.append("manifest is missing seeds/model_seeds")
    elif manifest_seeds is None:
        if (
            not model_rows.empty
            and requirements.require_manifest
            and requirements.expected_model_seeds is None
        ):
            errors.append(
                "manifest seeds are unavailable and no required seed set was supplied"
            )
    elif not isinstance(manifest_seeds, list):
        errors.append("seeds must be a list of integers or null")
    else:
        if any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in manifest_seeds
        ):
            errors.append("seeds must contain integers")
        elif "model_seed" not in model_rows.columns:
            errors.append("CSV is missing model_seed required by manifest seeds")
        else:
            numeric = pd.to_numeric(model_rows["model_seed"], errors="coerce")
            actual_seeds = {
                int(seed)
                for seed in numeric.dropna()
                if np.isfinite(seed) and seed == np.floor(seed)
            }
            if actual_seeds != set(manifest_seeds):
                errors.append(
                    f"manifest seeds={sorted(manifest_seeds)} do not match "
                    f"CSV seeds={sorted(actual_seeds)}"
                )
            if (
                requirements.expected_model_seeds is not None
                and set(manifest_seeds) != set(requirements.expected_model_seeds)
            ):
                errors.append(
                    "manifest seeds do not match the required model-seed set"
                )

    replicate_rows = _replicate_rows(raw_games, validation_mode)
    replicate_key_present = "evaluation_replicates" in manifest.values
    manifest_replicates = manifest.values.get("evaluation_replicates")
    if (
        not replicate_rows.empty
        and requirements.require_manifest
        and not replicate_key_present
    ):
        errors.append("manifest is missing evaluation_replicates")
    elif manifest_replicates is not None:
        if (
            isinstance(manifest_replicates, bool)
            or not isinstance(manifest_replicates, int)
            or manifest_replicates <= 0
        ):
            errors.append("evaluation_replicates must be a positive integer")
        elif "evaluation_replicate_id" not in replicate_rows.columns:
            errors.append(
                "CSV is missing evaluation_replicate_id required by manifest"
            )
        else:
            actual_replicates = (
                int(replicate_rows["evaluation_replicate_id"].nunique())
                if not replicate_rows.empty
                else 0
            )
            if actual_replicates != manifest_replicates:
                errors.append(
                    f"evaluation_replicates={manifest_replicates} does not "
                    f"match CSV replicates={actual_replicates}"
                )
            if (
                requirements.expected_evaluation_replicates is not None
                and manifest_replicates
                != requirements.expected_evaluation_replicates
            ):
                errors.append(
                    "manifest evaluation_replicates does not match the "
                    "required count"
                )
    elif not replicate_rows.empty and requirements.require_manifest:
        errors.append("manifest evaluation_replicates is unavailable")

    protocol_fields = (
        "protocol_id",
        "preset_name",
        "experiment_config_hash",
        "training_config_hash",
        "experiment_config",
        "source_revision",
        "source_dirty",
    )
    present_protocol_fields = [
        field for field in protocol_fields if field in manifest.values
    ]
    if present_protocol_fields and len(present_protocol_fields) != len(
        protocol_fields
    ):
        missing = sorted(set(protocol_fields) - set(present_protocol_fields))
        errors.append(f"protocol provenance is incomplete: missing {missing}")
    elif present_protocol_fields:
        snapshot = manifest.values["experiment_config"]
        if not isinstance(snapshot, dict):
            errors.append("experiment_config must be an object")
        else:
            try:
                calculated_experiment_hash = (
                    experiment_config_hash_from_snapshot(snapshot)
                )
                calculated_training_hash = training_config_hash_from_snapshot(
                    snapshot
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"experiment_config is malformed: {error}")
            else:
                if (
                    manifest.values["experiment_config_hash"]
                    != calculated_experiment_hash
                ):
                    errors.append(
                        "experiment_config_hash does not match experiment_config"
                    )
                if (
                    manifest.values["training_config_hash"]
                    != calculated_training_hash
                ):
                    errors.append(
                        "training_config_hash does not match experiment_config"
                    )
        source_revision = manifest.values.get("source_revision")
        if not isinstance(source_revision, str) or not source_revision.strip():
            errors.append("source_revision must be a non-empty string")
        source_dirty = manifest.values.get("source_dirty")
        if source_dirty is not None and not isinstance(source_dirty, bool):
            errors.append("source_dirty must be a boolean or null")

        model_training_hash = manifest.values.get(
            "model_training_config_hash"
        )
        if not model_rows.empty:
            if model_training_hash is None:
                if requirements.enforce_frozen_final_protocol:
                    errors.append(
                        "frozen final validation requires model protocol metadata"
                    )
            elif model_training_hash != manifest.values["training_config_hash"]:
                errors.append(
                    "model_training_config_hash does not match the evaluation "
                    "training_config_hash"
                )
            if requirements.enforce_frozen_final_protocol:
                if manifest.values.get("model_source_dirty") != [False]:
                    errors.append(
                        "frozen final validation requires models from a clean "
                        "source tree"
                    )
    elif requirements.enforce_frozen_final_protocol:
        errors.append("frozen final validation requires protocol provenance")

    if requirements.enforce_frozen_final_protocol and present_protocol_fields:
        if manifest.values.get("source_dirty") is not False:
            errors.append(
                "frozen final validation requires a clean source tree"
            )
        if manifest.values.get("protocol_id") != (
            FINAL_EXPERIMENT_CONFIG.protocol_id
        ):
            errors.append(
                "protocol_id is not "
                f"{FINAL_EXPERIMENT_CONFIG.protocol_id}"
            )
        if manifest.values.get("experiment_config_hash") != (
            FINAL_EXPERIMENT_CONFIG.config_hash
        ):
            errors.append(
                "experiment_config_hash does not match the frozen final protocol"
            )
        if manifest.values.get("training_config_hash") != (
            FINAL_EXPERIMENT_CONFIG.training_config_hash
        ):
            errors.append(
                "training_config_hash does not match the frozen final protocol"
            )
        expected_mode = (
            HEAD_TO_HEAD_EVALUATION
            if validation_mode == BASELINE_SANITY_MODE
            else validation_mode
        )
        try:
            expected_namespace = (
                FINAL_EXPERIMENT_CONFIG.evaluation.seed_namespace(expected_mode)
            )
        except ValueError:
            expected_namespace = None
        observed_namespace = manifest.values.get(
            "evaluation_seed_namespace",
            manifest.values.get("eval_seed_base"),
        )
        if (
            expected_namespace is not None
            and observed_namespace != expected_namespace
        ):
            errors.append(
                f"evaluation seed namespace must be {expected_namespace}"
            )
        skipped_agents = manifest.values.get("skipped_agents")
        if isinstance(skipped_agents, dict) and skipped_agents:
            errors.append(
                "frozen final validation does not allow skipped agents"
            )
        if not model_rows.empty:
            if "training_episode" not in model_rows.columns:
                observed_training_episodes: set[int] = set()
            else:
                observed_training_episodes = {
                    int(value)
                    for value in pd.to_numeric(
                        model_rows["training_episode"],
                        errors="coerce",
                    ).dropna()
                }
            if observed_training_episodes != {
                FINAL_EXPERIMENT_CONFIG.training.episodes
            }:
                errors.append(
                    "training_episode must equal the frozen 10000-episode budget"
                )

    valid = not errors
    return [
        _integrity_result(
            check_id="evaluation_manifest",
            check_name="Evaluation summary manifest",
            status=STATUS_PASS if valid else STATUS_FAIL,
            category="manifest_integrity",
            message=(
                "Evaluation summary manifest matches the CSV and protocol."
                if valid
                else "Evaluation summary manifest is invalid or inconsistent."
            ),
            observed_value=float(len(raw_games)),
            threshold=float(manifest_rows) if manifest_rows is not None else None,
            details={
                "manifest_path": str(manifest.path),
                "manifest_row_count": manifest_rows,
                "manifest_games": manifest_games,
                "csv_row_count": len(raw_games),
                "errors": errors,
            },
        )
    ]


def validate_raw_evaluation_integrity(
    raw_games: pd.DataFrame,
    *,
    validation_mode: str,
    requirements: IntegrityRequirements,
    manifest: EvaluationManifest | None,
) -> list[ValidationCheckResult]:
    checks = validate_required_columns(raw_games, validation_mode)
    if checks[0].status == STATUS_FAIL:
        checks.extend(
            validate_manifest(
                raw_games,
                manifest,
                requirements,
                validation_mode,
            )
        )
        return checks

    validators: Iterable[list[ValidationCheckResult]] = (
        validate_finite_numeric_values(raw_games),
        validate_numeric_domains(raw_games),
        validate_game_identity_uniqueness(raw_games, validation_mode),
        validate_profit_consistency(raw_games),
        validate_final_model_rows(raw_games, validation_mode),
        validate_model_metadata_values(raw_games, validation_mode),
        validate_replicate_metadata_values(raw_games, validation_mode),
        validate_metadata_domain_exclusivity(raw_games, validation_mode),
        validate_expected_seed_sets(
            raw_games,
            validation_mode,
            requirements,
            manifest,
        ),
        validate_games_per_matchup(
            raw_games,
            validation_mode,
            requirements,
            manifest,
        ),
        validate_expected_evaluation_replicates(
            raw_games,
            validation_mode,
            requirements,
            manifest,
        ),
        validate_manifest(
            raw_games,
            manifest,
            requirements,
            validation_mode,
        ),
    )
    for results in validators:
        checks.extend(results)
    return checks

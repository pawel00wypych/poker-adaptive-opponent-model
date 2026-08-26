from dataclasses import replace

import pytest

from src.experiment_protocol import (
    CROSS_PLAY_EVALUATION,
    EXTENDED_EXPERIMENT_CONFIG,
    FINAL_EXPERIMENT_CONFIG,
    GENERALIZATION_EVALUATION,
    HEAD_TO_HEAD_EVALUATION,
    LEARNING_CURVE_EVALUATION,
    STRESS_TEST_EVALUATION,
    TRAINING_OPPONENT_EVALUATION,
    VERIFICATION_EXPERIMENT_CONFIG,
    build_protocol_provenance,
    resolve_effective_config,
    validate_protocol_provenance,
)


def test_frozen_protocol_budgets_and_namespaces():
    assert FINAL_EXPERIMENT_CONFIG.training.episodes == 10_000
    assert FINAL_EXPERIMENT_CONFIG.training.seeds == (42, 123, 456, 789, 2026)
    assert FINAL_EXPERIMENT_CONFIG.evaluation.games_per_matchup == 500
    assert FINAL_EXPERIMENT_CONFIG.training.alpha_mode == "sqrt_visit"
    assert (
        FINAL_EXPERIMENT_CONFIG.evaluation.baseline_evaluation_replicates == 5
    )
    assert VERIFICATION_EXPERIMENT_CONFIG.training.episodes == 500
    assert VERIFICATION_EXPERIMENT_CONFIG.evaluation.games_per_matchup == 200
    assert EXTENDED_EXPERIMENT_CONFIG.evaluation.games_per_matchup == 1_000

    evaluation = FINAL_EXPERIMENT_CONFIG.evaluation
    assert evaluation.seed_namespace(TRAINING_OPPONENT_EVALUATION) == 1
    assert evaluation.seed_namespace(LEARNING_CURVE_EVALUATION) == 1
    assert evaluation.seed_namespace(HEAD_TO_HEAD_EVALUATION) == 2
    assert evaluation.seed_namespace(GENERALIZATION_EVALUATION) == 3
    assert evaluation.seed_namespace(STRESS_TEST_EVALUATION) == 4
    assert evaluation.seed_namespace(CROSS_PLAY_EVALUATION) == 5


def test_protocol_hash_is_deterministic_and_path_independent():
    changed_root = replace(
        FINAL_EXPERIMENT_CONFIG,
        training=replace(
            FINAL_EXPERIMENT_CONFIG.training,
            model_root_directory="D:/different/runtime/path",
        ),
    )

    assert changed_root.config_hash == FINAL_EXPERIMENT_CONFIG.config_hash
    assert (
        changed_root.training_config_hash
        == FINAL_EXPERIMENT_CONFIG.training_config_hash
    )


def test_extended_and_final_share_the_training_hash():
    assert (
        EXTENDED_EXPERIMENT_CONFIG.training_config_hash
        == FINAL_EXPERIMENT_CONFIG.training_config_hash
    )
    assert (
        EXTENDED_EXPERIMENT_CONFIG.config_hash
        != FINAL_EXPERIMENT_CONFIG.config_hash
    )


def test_scientific_override_is_labelled_custom():
    custom = resolve_effective_config(
        "final",
        training=replace(FINAL_EXPERIMENT_CONFIG.training, alpha=0.2),
    )

    assert custom.preset_name == "custom"
    assert custom.protocol_id == "custom-from-thesis-final-v2"


def test_provenance_contains_snapshot_hashes_and_source_revision():
    provenance = build_protocol_provenance(
        FINAL_EXPERIMENT_CONFIG,
        source_revision="abc123",
        source_dirty=False,
    )

    assert provenance.protocol_id == "thesis-final-v2"
    assert provenance.experiment_config_hash == FINAL_EXPERIMENT_CONFIG.config_hash
    assert provenance.training_config_hash == (
        FINAL_EXPERIMENT_CONFIG.training_config_hash
    )
    assert provenance.experiment_config["evaluation"]["games_per_matchup"] == 500
    assert provenance.source_revision == "abc123"
    assert provenance.source_dirty is False


def test_injected_provenance_must_match_the_effective_training_config():
    provenance = build_protocol_provenance(
        FINAL_EXPERIMENT_CONFIG,
        source_revision="revision",
        source_dirty=False,
    )
    custom = resolve_effective_config(
        "final",
        training=replace(FINAL_EXPERIMENT_CONFIG.training, alpha=0.2),
    )

    with pytest.raises(ValueError, match="effective run"):
        validate_protocol_provenance(
            provenance,
            custom,
            verify_source=False,
        )

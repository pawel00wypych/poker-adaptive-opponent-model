import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.experiment_protocol import (
    FINAL_EXPERIMENT_CONFIG,
    EXTENDED_EXPERIMENT_CONFIG,
    VERIFICATION_EXPERIMENT_CONFIG,
    build_protocol_provenance,
    resolve_effective_config,
)
from src.evaluation.runners.model_evaluator import ModelBundle
from src.experiments.evaluation.protocol_cli import attach_model_provenance
from src.experiments.evaluation.run_cross_play_evaluation import (
    parse_args as parse_cross_play,
)
from src.experiments.evaluation.run_generalization_evaluation import (
    parse_args as parse_generalization,
)
from src.experiments.evaluation.run_head_to_head_evaluation import (
    parse_args as parse_head_to_head,
)
from src.experiments.evaluation.run_learning_curve_evaluation import (
    parse_args as parse_learning_curve,
)
from src.experiments.evaluation.run_stress_test_evaluation import (
    parse_args as parse_stress_test,
)
from src.experiments.evaluation.run_training_opponent_evaluation import (
    parse_args as parse_training_opponent,
)
from src.experiments.evaluation.run_training_opponent_evaluation import (
    save_summary as save_training_summary,
)


@pytest.mark.parametrize(
    ("parser", "arguments", "expected_namespace"),
    [
        (parse_training_opponent, ["--training-run-dir", "run"], 1),
        (
            parse_learning_curve,
            [
                "--training-run-dir",
                "run",
                "--checkpoint-episodes",
                "1000",
            ],
            1,
        ),
        (parse_head_to_head, ["--training-run-dir", "run"], 2),
        (parse_generalization, ["--training-run-dir", "run"], 3),
        (parse_stress_test, ["--training-run-dir", "run"], 4),
        (parse_cross_play, ["--training-run-dir", "run"], 5),
    ],
)
def test_final_evaluation_defaults_use_frozen_budget_and_small_namespaces(
    parser,
    arguments,
    expected_namespace,
):
    args = parser(arguments)

    assert args.games == (
        200 if parser is parse_learning_curve else 500
    )
    assert args.eval_seed_base == expected_namespace
    assert args.experiment_config.preset_name == "final"
    assert args.protocol_provenance.experiment_config_hash == (
        FINAL_EXPERIMENT_CONFIG.config_hash
    )


def test_verification_and_extended_evaluation_budgets():
    verification = parse_generalization(
        ["--training-run-dir", "run", "--config", "verification"]
    )
    extended = parse_generalization(
        ["--training-run-dir", "run", "--config", "extended"]
    )
    verification_head = parse_head_to_head(
        ["--training-run-dir", "run", "--config", "verification"]
    )

    assert verification.games == 200
    assert verification.protocol_provenance.experiment_config_hash == (
        VERIFICATION_EXPERIMENT_CONFIG.config_hash
    )
    assert verification_head.evaluation_replicates == 3
    assert extended.games == 1_000


def test_scientific_evaluation_override_is_labelled_custom():
    args = parse_stress_test(
        [
            "--training-run-dir",
            "run",
            "--games",
            "321",
            "--eval-seed-namespace",
            "9",
        ]
    )

    assert args.games == 321
    assert args.eval_seed_base == 9
    assert args.experiment_config.preset_name == "custom"
    assert args.protocol_provenance.protocol_id.startswith("custom-from-")


def test_evaluation_summary_persists_protocol_snapshot(tmp_path):
    args = parse_training_opponent(["--training-run-dir", "run"])
    output_path = tmp_path / "results.csv"

    save_training_summary(
        output_path=output_path,
        arguments=args,
        bundle_count=5,
        training_episodes=[10_000],
        row_count=10,
        duration_seconds=1.0,
        evaluated_agents=("adaptive_mc",),
        skipped_agents={},
    )

    summary = json.loads(
        output_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert summary["games"] == 500
    assert summary["eval_seed_base"] == 1
    assert summary["evaluation_seed_namespace"] == 1
    assert summary["protocol_id"] == "thesis-final-v2"
    assert summary["experiment_config_hash"] == (
        FINAL_EXPERIMENT_CONFIG.config_hash
    )
    assert summary["experiment_config"]["representation"]["reward_version"] == (
        "reward_bb_v1"
    )


def _bundle_with_protocol(config, *, source_dirty=False):
    provenance = build_protocol_provenance(
        config,
        source_revision="model-revision",
        source_dirty=source_dirty,
    )
    path = Path("model.pkl")
    return ModelBundle(
        training_run_directory=Path("run"),
        seed=42,
        episode=config.training.episodes,
        model_source="final",
        unknown_model_path=path,
        tight_model_path=path,
        aggressive_model_path=path,
        calling_model_path=path,
        protocol_id=provenance.protocol_id,
        preset_name=provenance.preset_name,
        experiment_config_hash=provenance.experiment_config_hash,
        training_config_hash=provenance.training_config_hash,
        source_revision=provenance.source_revision,
        source_dirty=provenance.source_dirty,
        experiment_config=provenance.experiment_config,
    )


def test_custom_trained_models_rebase_evaluation_to_custom_protocol():
    custom_training = resolve_effective_config(
        "final",
        training=replace(FINAL_EXPERIMENT_CONFIG.training, alpha=0.2),
    )
    args = parse_generalization(["--training-run-dir", "run"])

    attach_model_provenance(
        args,
        [_bundle_with_protocol(custom_training)],
    )

    assert args.experiment_config.preset_name == "custom"
    assert args.protocol_provenance.training_config_hash == (
        custom_training.training_config_hash
    )
    assert args.protocol_provenance.protocol_id.startswith(
        "custom-evaluation-of-"
    )


@pytest.mark.parametrize(
    ("evaluation_preset", "model_config"),
    [
        ("verification", VERIFICATION_EXPERIMENT_CONFIG),
        ("final", FINAL_EXPERIMENT_CONFIG),
        ("extended", FINAL_EXPERIMENT_CONFIG),
    ],
)
def test_evaluation_labels_models_with_current_preset(
    evaluation_preset,
    model_config,
):
    args = parse_generalization(
        ["--training-run-dir", "run", "--config", evaluation_preset]
    )
    source_bundle = _bundle_with_protocol(model_config)

    labelled_bundles = attach_model_provenance(args, [source_bundle])

    assert labelled_bundles[0].preset_name == model_config.preset_name
    assert labelled_bundles[0].evaluation_run_name == evaluation_preset


def test_clean_and_unknown_model_dirtiness_cannot_be_mixed():
    clean = _bundle_with_protocol(FINAL_EXPERIMENT_CONFIG, source_dirty=False)
    unknown = replace(clean, seed=123, source_dirty=None)
    args = parse_generalization(["--training-run-dir", "run"])

    with pytest.raises(ValueError, match="clean and dirty"):
        attach_model_provenance(args, [clean, unknown])


def test_custom_models_may_differ_only_in_historical_evaluation_settings():
    custom_training = replace(
        FINAL_EXPERIMENT_CONFIG.training,
        alpha=0.2,
    )
    custom_final = resolve_effective_config(
        "final",
        training=custom_training,
    )
    custom_extended = resolve_effective_config(
        "extended",
        training=custom_training,
        evaluation=EXTENDED_EXPERIMENT_CONFIG.evaluation,
    )
    args = parse_generalization(["--training-run-dir", "run"])

    attach_model_provenance(
        args,
        [
            _bundle_with_protocol(custom_final),
            replace(
                _bundle_with_protocol(custom_extended),
                seed=123,
            ),
        ],
    )

    assert args.experiment_config.preset_name == "custom"
    assert args.protocol_provenance.training_config_hash == (
        custom_final.training_config_hash
    )

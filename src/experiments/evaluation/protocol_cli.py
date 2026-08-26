from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.experiment_protocol import (
    CUSTOM_PRESET,
    DEFAULT_EXPERIMENT_CONFIG_PRESET,
    EXPERIMENT_CONFIG_PRESETS,
    ProtocolProvenance,
    build_protocol_provenance,
    experiment_config_from_snapshot,
    experiment_config_for,
    protocol_metadata,
    resolve_effective_config,
)
from src.training.training_metadata import save_json


def add_evaluation_protocol_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_evaluation_replicates: bool = False,
) -> None:
    parser.add_argument(
        "--config",
        choices=sorted(EXPERIMENT_CONFIG_PRESETS),
        default=DEFAULT_EXPERIMENT_CONFIG_PRESET,
        help=(
            "Versioned experiment protocol. Scientific overrides are allowed "
            "but the resulting run is labelled custom."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Games per matchup. Defaults to the selected protocol.",
    )
    parser.add_argument(
        "--eval-seed-namespace",
        "--eval-seed-base",
        dest="eval_seed_base",
        type=int,
        default=None,
        help=(
            "Small namespace identifier used to derive deterministic game "
            "seeds. Defaults to the selected evaluation type."
        ),
    )
    if include_evaluation_replicates:
        parser.add_argument(
            "--evaluation-replicates",
            type=int,
            default=None,
            help=(
                "Independent baseline-only simulation replicates. Defaults "
                "to the selected protocol."
            ),
        )


def resolve_evaluation_protocol(
    args: argparse.Namespace,
    *,
    evaluation_type: str,
) -> argparse.Namespace:
    base = experiment_config_for(args.config)
    configured_evaluation = base.evaluation
    games = (
        args.games
        if args.games is not None
        else configured_evaluation.games_for(evaluation_type)
    )
    namespace = (
        args.eval_seed_base
        if args.eval_seed_base is not None
        else configured_evaluation.seed_namespace(evaluation_type)
    )
    replicates = getattr(args, "evaluation_replicates", None)
    if hasattr(args, "evaluation_replicates") and replicates is None:
        replicates = configured_evaluation.baseline_evaluation_replicates

    namespaces = dict(configured_evaluation.seed_namespaces)
    namespaces[evaluation_type] = namespace
    effective_evaluation = replace(
        configured_evaluation,
        games_per_matchup=(
            configured_evaluation.games_per_matchup
            if evaluation_type == "learning-curve"
            else games
        ),
        learning_curve_games_per_matchup=(
            games
            if evaluation_type == "learning-curve"
            else configured_evaluation.learning_curve_games_per_matchup
        ),
        baseline_evaluation_replicates=(
            replicates
            if replicates is not None
            else configured_evaluation.baseline_evaluation_replicates
        ),
        seed_namespaces=tuple(namespaces.items()),
    )
    effective_config = resolve_effective_config(
        args.config,
        evaluation=effective_evaluation,
    )

    args.games = games
    args.eval_seed_base = namespace
    if hasattr(args, "evaluation_replicates"):
        args.evaluation_replicates = replicates
    args.experiment_config = effective_config
    args.protocol_provenance = build_protocol_provenance(effective_config)
    return args


def save_evaluation_summary(
    *,
    output_path: str | Path,
    summary: dict[str, object],
    provenance: ProtocolProvenance,
) -> Path:
    path = Path(output_path).with_suffix(".summary.json")
    save_json(
        path,
        {
            **summary,
            "evaluation_seed_namespace": summary.get("eval_seed_base"),
            **protocol_metadata(provenance),
        },
    )
    return path


def attach_model_provenance(
    args: argparse.Namespace,
    bundles,
) -> list:
    """Validate model provenance and label bundles with the evaluation preset.

    Model provenance remains unchanged. This distinction matters for the
    ``extended`` preset, which evaluates models whose source preset is ``final``.
    """
    bundle_list = list(bundles)
    hashes = {
        bundle.training_config_hash
        for bundle in bundle_list
        if bundle.training_config_hash is not None
    }
    legacy_count = sum(
        bundle.training_config_hash is None for bundle in bundle_list
    )
    if hashes and legacy_count:
        raise ValueError(
            "Evaluation bundles mix legacy and protocol-aware model metadata."
        )
    if len(hashes) > 1:
        raise ValueError(
            "Evaluation bundles use multiple training_config_hash values."
        )
    if hashes:
        model_hash = next(iter(hashes))
        if model_hash != args.protocol_provenance.training_config_hash:
            snapshots = [
                bundle.experiment_config
                for bundle in bundle_list
                if bundle.experiment_config is not None
            ]
            if len(snapshots) != len(bundle_list):
                raise ValueError(
                    "Custom model bundles do not all expose experiment_config."
                )
            model_config = experiment_config_from_snapshot(
                snapshots[0]
            )
            combined_opponents = replace(
                args.experiment_config.opponents,
                training=model_config.opponents.training,
            )
            combined = replace(
                model_config,
                preset_name=CUSTOM_PRESET,
                protocol_id=(
                    f"custom-evaluation-of-{model_config.protocol_id}"
                ),
                evaluation=args.experiment_config.evaluation,
                opponents=combined_opponents,
            )
            args.experiment_config = combined
            args.protocol_provenance = build_protocol_provenance(combined)
            if (
                model_hash
                != args.protocol_provenance.training_config_hash
            ):
                raise ValueError(
                    "Custom model snapshot does not reproduce its "
                    "training_config_hash."
                )
        args.model_training_config_hash = model_hash
        revisions = {
            bundle.source_revision
            for bundle in bundle_list
            if bundle.source_revision is not None
        }
        dirty_values = {bundle.source_dirty for bundle in bundle_list}
        if len(revisions) > 1:
            raise ValueError(
                "Evaluation bundles use multiple source revisions."
            )
        if len(dirty_values) > 1:
            raise ValueError(
                "Evaluation bundles mix clean and dirty source provenance."
            )
        args.model_source_revisions = sorted(revisions)
        args.model_source_dirty = sorted(dirty_values)
    else:
        args.model_training_config_hash = None
        args.model_source_revisions = []
        args.model_source_dirty = []

    return [
        replace(
            bundle,
            evaluation_run_name=args.experiment_config.preset_name,
        )
        for bundle in bundle_list
    ]


def model_provenance_summary(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    return {
        "model_training_config_hash": getattr(
            arguments,
            "model_training_config_hash",
            None,
        ),
        "model_source_revisions": getattr(
            arguments,
            "model_source_revisions",
            [],
        ),
        "model_source_dirty": getattr(
            arguments,
            "model_source_dirty",
            [],
        ),
    }

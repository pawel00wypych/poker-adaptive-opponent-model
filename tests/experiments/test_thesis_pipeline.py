from __future__ import annotations

import json
import importlib
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.evaluation.algorithm_metadata import (
    ADAPTIVE_AGENTS,
    GENERAL_POLICY_AGENTS,
)
from src.experiments import run_thesis_pipeline as pipeline


def _context(tmp_path, *extra_args):
    args = pipeline.parse_args(
        [
            "--config",
            "verification",
            "--output-dir",
            str(tmp_path / "pipeline"),
            *extra_args,
        ]
    )
    return pipeline.build_context(args)


def _option_values(command, option, next_option):
    start = command.index(option) + 1
    end = command.index(next_option)
    return tuple(command[start:end])


def test_parser_defaults_to_safe_verification_profile():
    args = pipeline.parse_args([])

    assert args.config == "verification"
    assert args.workers == 1
    assert args.heartbeat_seconds == 60
    assert args.resume is False


def test_verification_stage_graph_has_expected_order_and_reports(tmp_path):
    context = _context(tmp_path)
    stages = pipeline.build_stages(context)
    stage_ids = [stage.stage_id for stage in stages]

    assert stage_ids[:4] == [
        "train_monte_carlo",
        "train_q_learning",
        "train_sarsa",
        "train_double_q_learning",
    ]
    assert stage_ids[-1] == "finalize_pipeline"
    assert len(stages) == 33
    assert {
        "report_learning_curve",
        "report_training_opponents",
        "report_algorithm_comparison",
        "report_seed_stability",
        "report_classifier_quality",
        "report_experiment_summary",
        "report_generalization_overview",
        "report_stress_overview",
        "report_cross_play_overview",
        "report_baseline_overview",
    }.issubset(stage_ids)
    assert {
        "validate_training",
        "validate_generalization",
        "validate_stress",
        "validate_cross_play",
        "validate_baseline",
    }.issubset(stage_ids)


def test_extended_reuses_final_models_and_skips_training_diagnostics(tmp_path):
    args = pipeline.parse_args(
        [
            "--config",
            "extended",
            "--output-dir",
            str(tmp_path / "extended"),
            "--final-pipeline-dir",
            str(tmp_path / "final"),
        ]
    )
    context = pipeline.build_context(args)
    stages = pipeline.build_stages(context)
    stage_ids = {stage.stage_id for stage in stages}

    assert "verify_final_models" in stage_ids
    assert not any(stage_id.startswith("train_") for stage_id in stage_ids)
    assert "evaluate_learning_curve" not in stage_ids
    assert "report_learning_curve" not in stage_ids
    assert len(stages) == 28
    assert context.model_paths.monte_carlo == (
        (tmp_path / "final" / "models" / "monte_carlo").resolve()
    )


def test_cross_play_commands_cover_all_requested_matrices(tmp_path):
    stages = pipeline.build_stages(_context(tmp_path))
    by_id = {stage.stage_id: stage for stage in stages}

    adaptive = by_id["evaluate_cross_play_adaptive"].command
    general = by_id["evaluate_cross_play_general"].command
    assert set(
        _option_values(adaptive, "--agents", "--opponent-agents")
    ) == set(ADAPTIVE_AGENTS)
    assert set(
        _option_values(general, "--agents", "--opponent-agents")
    ) == set(GENERAL_POLICY_AGENTS)

    for algorithm, adaptive_agent, general_agent in zip(
        pipeline.ALGORITHM_KEYS,
        ADAPTIVE_AGENTS,
        GENERAL_POLICY_AGENTS,
        strict=True,
    ):
        command = by_id[f"evaluate_cross_play_paired_{algorithm}"].command
        agents = _option_values(command, "--agents", "--opponent-agents")
        assert set(agents) == {adaptive_agent, general_agent}
        assert "--no-include-self-play" in command


def test_training_commands_use_preset_budget_and_quiet_progress_interval(tmp_path):
    context = _context(tmp_path)
    stages = {stage.stage_id: stage for stage in pipeline.build_stages(context)}
    mc_command = stages["train_monte_carlo"].command
    q_command = stages["train_q_learning"].command

    assert mc_command[mc_command.index("--config") + 1] == "verification"
    assert mc_command[mc_command.index("--episodes") + 1] == "500"
    assert mc_command[mc_command.index("--log-interval") + 1] == "100"
    assert "--rerun-existing" in mc_command
    assert q_command[q_command.index("--alpha-mode") + 1] == "sqrt_visit"
    assert q_command[q_command.index("--log-interval") + 1] == "100"


def test_every_generated_subprocess_command_is_accepted_by_its_cli(
    tmp_path,
    monkeypatch,
):
    for stage in pipeline.build_stages(_context(tmp_path)):
        if not stage.command:
            continue
        assert stage.command[1:3] == ("-u", "-m")
        module_name = stage.command[3]
        arguments = list(stage.command[4:])
        module = importlib.import_module(module_name)
        parse_args = module.parse_args
        if inspect.signature(parse_args).parameters:
            parse_args(arguments)
        else:
            monkeypatch.setattr(
                sys,
                "argv",
                [module_name, *arguments],
            )
            parse_args()


def test_pipeline_tees_messages_to_console_and_log(tmp_path, capsys):
    log_path = tmp_path / "pipeline.log"

    with pipeline.TeeLogger(log_path) as logger:
        logger.emit("stage started")

    assert "stage started" in capsys.readouterr().out
    assert "stage started" in log_path.read_text(encoding="utf-8")


def test_resume_skips_complete_stages_and_reruns_invalidated_descendants(
    tmp_path,
    monkeypatch,
):
    context = _context(tmp_path)
    first_output = context.paths.root / "first.txt"
    second_output = context.paths.root / "second.txt"
    stages = (
        pipeline.Stage(
            "first",
            "First",
            (),
            (first_output,),
            "internal",
            1,
            command=("fake-first",),
        ),
        pipeline.Stage(
            "second",
            "Second",
            ("first",),
            (second_output,),
            "internal",
            1,
            command=("fake-second",),
        ),
    )
    calls = []

    def fake_run(context, stage, logger):
        calls.append(stage.stage_id)
        stage.outputs[0].parent.mkdir(parents=True, exist_ok=True)
        stage.outputs[0].write_text(stage.stage_id, encoding="utf-8")

    monkeypatch.setattr(pipeline, "run_subprocess_stage", fake_run)
    assert pipeline.run_pipeline(context, stages) == 0
    assert calls == ["first", "second"]

    resume_context = replace(context, resume=True)
    assert pipeline.run_pipeline(resume_context, stages) == 0
    assert calls == ["first", "second"]

    first_output.unlink()
    assert pipeline.run_pipeline(resume_context, stages) == 0
    assert calls == ["first", "second", "first", "second"]


def test_pipeline_stops_on_first_failed_stage(tmp_path, monkeypatch):
    context = _context(tmp_path)
    stages = (
        pipeline.Stage(
            "failure",
            "Failure",
            (),
            (context.paths.root / "failure.txt",),
            "internal",
            1,
            command=("fail",),
        ),
        pipeline.Stage(
            "never",
            "Never",
            ("failure",),
            (context.paths.root / "never.txt",),
            "internal",
            1,
            command=("never",),
        ),
    )
    calls = []

    def fake_run(context, stage, logger):
        calls.append(stage.stage_id)
        raise pipeline.PipelineError("boom")

    monkeypatch.setattr(pipeline, "run_subprocess_stage", fake_run)

    assert pipeline.run_pipeline(context, stages) == 1
    assert calls == ["failure"]
    summary = json.loads(
        context.paths.summary.read_text(encoding="utf-8")
    )
    assert summary["pipeline_status"] == "failed"
    assert summary["failed_stages"] == ["failure"]


def test_changed_command_prevents_resume(tmp_path, monkeypatch):
    context = _context(tmp_path)
    output = context.paths.root / "stage.txt"
    original = (
        pipeline.Stage(
            "stage",
            "Stage",
            (),
            (output,),
            "internal",
            1,
            command=("first-command",),
        ),
    )
    calls = []

    def fake_run(context, stage, logger):
        calls.append(stage.command)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("done", encoding="utf-8")

    monkeypatch.setattr(pipeline, "run_subprocess_stage", fake_run)
    assert pipeline.run_pipeline(context, original) == 0
    changed = (
        replace(original[0], command=("changed-command",)),
    )
    assert pipeline.run_pipeline(replace(context, resume=True), changed) == 0
    assert calls == [("first-command",), ("changed-command",)]


def test_csv_and_summary_merge_preserve_union_and_provenance(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.csv"
    first.write_text("a,b\n1,2\n", encoding="utf-8")
    second.write_text("a,c\n3,4\n", encoding="utf-8")
    common = {
        "protocol_id": "test",
        "preset_name": "verification",
        "experiment_config_hash": "hash",
        "training_config_hash": "training",
        "experiment_config": {},
        "source_revision": "revision",
        "source_dirty": False,
        "model_training_config_hash": "training",
        "model_source_revisions": ["revision"],
        "model_source_dirty": [False],
        "model_source": "final",
        "training_episodes": [500],
        "seeds": [42, 123, 456],
        "games": 200,
        "eval_seed_base": 1,
        "bundle_count": 3,
        "row_count": 1,
        "duration_seconds": 1.0,
    }
    first.with_suffix(".summary.json").write_text(
        json.dumps(common),
        encoding="utf-8",
    )
    second.with_suffix(".summary.json").write_text(
        json.dumps(common),
        encoding="utf-8",
    )

    assert pipeline._merge_csv_files([first, second], output) == 2
    pipeline._merge_summary_files(
        [
            first.with_suffix(".summary.json"),
            second.with_suffix(".summary.json"),
        ],
        output,
        evaluation_type="combined",
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "a,b,c",
        "1,2,",
        "3,,4",
    ]
    summary = json.loads(
        output.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert summary["row_count"] == 2
    assert summary["protocol_id"] == "test"


def test_dry_run_does_not_create_manifest(tmp_path, capsys):
    context = replace(_context(tmp_path), dry_run=True)
    stages = pipeline.build_stages(context)

    assert pipeline.run_pipeline(context, stages) == 0
    assert not context.paths.manifest.exists()
    assert not context.paths.root.exists()
    assert "train_monte_carlo" in capsys.readouterr().out


def test_progress_estimator_uses_completed_rate():
    stage = pipeline.Stage(
        "train",
        "Train",
        (),
        (Path("out"),),
        "training",
        100,
        command=("train",),
    )
    estimator = pipeline.ProgressEstimator(
        (stage,),
        {
            "train": {
                "status": "success",
                "duration_seconds": 50,
            }
        },
    )

    assert estimator.estimate_remaining({"train"}) == pytest.approx(50)


def test_final_summary_renders_finalization_as_success(tmp_path):
    context = _context(tmp_path)
    output = context.paths.reports / "final_thesis_summary_g200.md"
    manifest = {
        "stages": {
            "finalize_pipeline": {
                "title": "Finalize",
                "status": "running",
                "duration_seconds": 0.0,
                "outputs": [str(output)],
            }
        }
    }

    pipeline._finalize_pipeline(context, manifest)

    assert "| `finalize_pipeline` | success |" in output.read_text(
        encoding="utf-8"
    )


def test_multi_output_report_stages_declare_json_csv_and_latex(tmp_path):
    stages = {
        stage.stage_id: stage for stage in pipeline.build_stages(_context(tmp_path))
    }

    algorithm_outputs = {
        path.relative_to(
            _context(tmp_path).paths.reports / "algorithm_comparison"
        ).as_posix()
        for path in stages["report_algorithm_comparison"].outputs
    }
    assert {
        "algorithm_comparison.md",
        "algorithm_comparison.json",
        "algorithm_global_ranking.csv",
        "algorithm_global_ranking.tex",
        "charts/algorithm_global_mean_profit.png",
    }.issubset(algorithm_outputs)
    learning_outputs = {
        path.relative_to(
            _context(tmp_path).paths.reports / "learning_curve"
        ).as_posix()
        for path in stages["report_learning_curve"].outputs
    }
    assert {
        "learning_curve_report.md",
        "learning_curve_report.html",
        "plots/checkpoint_mean_profit_bb.png",
        "plots/checkpoint_global_classifier_coverage.png",
    }.issubset(learning_outputs)


def test_merge_preserves_multiple_seed_namespaces(tmp_path):
    first = tmp_path / "first.summary.json"
    second = tmp_path / "second.summary.json"
    output = tmp_path / "primary.csv"
    common = {
        "protocol_id": "test",
        "preset_name": "verification",
        "experiment_config_hash": "hash",
        "training_config_hash": "training",
        "experiment_config": {},
        "source_revision": "revision",
        "source_dirty": False,
        "model_training_config_hash": "training",
        "model_source_revisions": ["revision"],
        "model_source_dirty": [False],
        "model_source": "final",
        "training_episodes": [500],
        "seeds": [42, 123, 456],
        "games": 200,
        "bundle_count": 3,
        "row_count": 1,
        "duration_seconds": 1.0,
    }
    first.write_text(
        json.dumps(
            {
                **common,
                "eval_seed_base": 1,
                "evaluation_seed_namespace": 1,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                **common,
                "eval_seed_base": 3,
                "evaluation_seed_namespace": 3,
            }
        ),
        encoding="utf-8",
    )

    pipeline._merge_summary_files(
        [first, second],
        output,
        evaluation_type="primary",
    )

    merged = json.loads(
        output.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert merged["evaluation_seed_namespace"] is None
    assert merged["evaluation_seed_namespaces"] == [1, 3]

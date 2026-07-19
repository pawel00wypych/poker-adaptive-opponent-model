from src.experiments.create_checkpoint_report import parse_args as parse_checkpoint_args
from src.experiments.create_q_table_report import parse_args as parse_q_table_args
from src.experiments.compare_selected_q_tables import parse_args as parse_compare_args


def test_create_checkpoint_report_parser_accepts_format():
    args = parse_checkpoint_args(
        [
            "--input-path",
            "results/evaluation/results.csv",
            "--output-dir",
            "reports/checkpoint",
            "--format",
            "html",
        ]
    )

    assert args.format == "html"
    assert args.input_path == "results/evaluation/results.csv"


def test_create_q_table_report_parser_accepts_disagreement_limit():
    args = parse_q_table_args(
        [
            "--input-path",
            "results/evaluation/q.json",
            "--output-dir",
            "reports/q",
            "--disagreement-limit",
            "3",
        ]
    )

    assert args.disagreement_limit == 3


def test_compare_selected_q_tables_parser_accepts_checkpoints():
    args = parse_compare_args(
        [
            "--training-run-dir",
            "results/training_runs/state_v2_linear_2000_sqrt_visit",
            "--unknown-checkpoint",
            "2000",
            "--calling-checkpoint",
            "2000",
            "--unknown-seeds",
            "42",
            "456",
            "--calling-seeds",
            "42",
            "123",
            "456",
        ]
    )

    assert args.training_run_dir == "results/training_runs/state_v2_linear_2000_sqrt_visit"
    assert args.unknown_checkpoint == 2000
    assert args.calling_checkpoint == 2000
    assert args.unknown_seeds == [42, 456]
    assert args.calling_seeds == [42, 123, 456]

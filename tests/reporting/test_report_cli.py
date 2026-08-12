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


def test_compare_selected_q_tables_parser_accepts_general_arguments():
    args = parse_compare_args(
        [
            "--training-run-dir",
            "results/training_runs/state_v2_linear_2000_sqrt_visit",
            "--checkpoint",
            "2000",
            "--seeds",
            "42",
            "123",
            "456",
            "--policies",
            "unknown",
            "tight",
            "aggressive",
            "calling",
            "--output-path",
            "results/evaluation/q_table_comparison_all_policies.json",
            "--top-n",
            "30",
        ]
    )

    assert args.training_run_dir == (
        "results/training_runs/state_v2_linear_2000_sqrt_visit"
    )
    assert args.checkpoint == 2000
    assert args.seeds == [42, 123, 456]
    assert args.policies == [
        "unknown",
        "tight",
        "aggressive",
        "calling",
    ]
    assert args.output_path == (
        "results/evaluation/q_table_comparison_all_policies.json"
    )
    assert args.top_n == 30
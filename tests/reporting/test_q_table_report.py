import json

from src.evaluation.reporting.q_table_report import (
    comparisons_to_dataframe,
    summaries_to_dataframe,
    write_q_table_html_report,
)


def sample_q_report():
    return {
        "summaries": [
            {
                "name": "policy_calling_seed_42_cp_2000",
                "states": 10,
                "fully_zero_states": 0,
                "fully_zero_rate": 0.0,
                "tied_best_states": 1,
                "tied_best_rate": 10.0,
                "best_action_counts": {
                    "fold": 1,
                    "call": 6,
                    "raise": 3,
                },
                "best_action_rates": {
                    "fold": 10.0,
                    "call": 60.0,
                    "raise": 30.0,
                },
                "action_stats": [
                    {
                        "action": "fold",
                        "mean_q": -1.0,
                        "median_q": -1.0,
                        "std_q": 0.1,
                        "min_q": -2.0,
                        "max_q": 0.0,
                        "zero_count": 1,
                        "zero_rate": 10.0,
                    },
                    {
                        "action": "call",
                        "mean_q": 0.5,
                        "median_q": 0.4,
                        "std_q": 0.2,
                        "min_q": -1.0,
                        "max_q": 2.0,
                        "zero_count": 0,
                        "zero_rate": 0.0,
                    },
                    {
                        "action": "raise",
                        "mean_q": 0.7,
                        "median_q": 0.6,
                        "std_q": 0.3,
                        "min_q": -1.0,
                        "max_q": 3.0,
                        "zero_count": 0,
                        "zero_rate": 0.0,
                    },
                ],
            },
            {
                "name": "policy_calling_seed_456_cp_2000",
                "states": 10,
                "fully_zero_states": 0,
                "fully_zero_rate": 0.0,
                "tied_best_states": 0,
                "tied_best_rate": 0.0,
                "best_action_counts": {
                    "fold": 2,
                    "call": 5,
                    "raise": 3,
                },
                "best_action_rates": {
                    "fold": 20.0,
                    "call": 50.0,
                    "raise": 30.0,
                },
                "action_stats": [],
            },
        ],
        "comparisons": [
            {
                "left_name": "policy_calling_seed_42_cp_2000",
                "right_name": "policy_calling_seed_456_cp_2000",
                "left_states": 10,
                "right_states": 10,
                "common_states": 9,
                "left_only_states": 1,
                "right_only_states": 1,
                "best_action_agreement": 7,
                "best_action_agreement_rate": 77.777,
                "transition_counts": {},
                "mean_abs_q_delta_by_action": {
                    "fold": 0.1,
                    "call": 0.2,
                    "raise": 0.3,
                },
                "mean_max_abs_q_delta": 0.4,
            }
        ],
        "largest_disagreements": {
            "a__vs__b": [
                {
                    "left_best_action": "call",
                    "right_best_action": "raise",
                    "max_abs_q_delta": 2.0,
                    "state_description": {"street": 0},
                }
            ]
        },
    }


def test_summaries_to_dataframe_expands_action_rates():
    df = summaries_to_dataframe(sample_q_report())

    assert len(df) == 2
    assert df.iloc[0]["best_call_rate"] == 60.0
    assert df.iloc[0]["q_raise_mean"] == 0.7


def test_comparisons_to_dataframe_expands_delta_columns():
    df = comparisons_to_dataframe(sample_q_report())

    assert len(df) == 1
    assert df.iloc[0]["mean_abs_q_delta_raise"] == 0.3


def test_write_q_table_html_report_creates_file_and_plots(tmp_path):
    input_path = tmp_path / "q_report.json"
    output_dir = tmp_path / "report"
    input_path.write_text(json.dumps(sample_q_report()), encoding="utf-8")

    output_path = write_q_table_html_report(input_path, output_dir)

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Q-table comparison report" in html
    assert "State encoding glossary" in html
    assert any((output_dir / "plots").glob("*.png"))

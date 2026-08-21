import pandas as pd

from src.evaluation.metrics.oracle_gap import (
    ORACLE_GAP_BB_COLUMN,
    calculate_oracle_gap_bb,
)


def test_oracle_gap_is_oracle_minus_adaptive():
    assert ORACLE_GAP_BB_COLUMN == "oracle_gap_bb"
    assert calculate_oracle_gap_bb(12.0, 8.0) == 4.0
    assert calculate_oracle_gap_bb(8.0, 12.0) == -4.0
    assert calculate_oracle_gap_bb(10.0, 10.0) == 0.0


def test_oracle_gap_supports_report_series():
    oracle = pd.Series([12.0, 8.0, 10.0])
    adaptive = pd.Series([8.0, 12.0, 10.0])

    result = calculate_oracle_gap_bb(oracle, adaptive)

    assert result.tolist() == [4.0, -4.0, 0.0]

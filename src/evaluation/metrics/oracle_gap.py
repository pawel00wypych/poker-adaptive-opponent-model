from __future__ import annotations

import pandas as pd

ORACLE_GAP_BB_COLUMN = "oracle_gap_bb"

OracleGapOperand = float | pd.Series


def calculate_oracle_gap_bb(
    oracle_mean_profit_bb: OracleGapOperand,
    adaptive_mean_profit_bb: OracleGapOperand,
) -> OracleGapOperand:
    """Return Oracle minus adaptive profit.

    A positive value means that the adaptive agent underperforms Oracle.
    A negative value means that the adaptive agent outperforms Oracle.
    """

    return oracle_mean_profit_bb - adaptive_mean_profit_bb

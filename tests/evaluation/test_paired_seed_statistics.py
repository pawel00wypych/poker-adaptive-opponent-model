import pandas as pd
import pytest

from src.evaluation.metrics.paired_seed_statistics import (
    PairedSeedStatisticsError,
    calculate_paired_seed_statistics,
)


def make_seed_row(
    agent_name: str,
    seed: int,
    mean_profit_bb: float,
    *,
    opponent_name: str = "calling",
    checkpoint_episode: int = 1000,
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "checkpoint_episode": checkpoint_episode,
        "model_seed": seed,
        "mean_profit_bb": mean_profit_bb,
    }


def test_paired_seed_statistics_use_student_t_interval_for_deltas():
    rows = pd.DataFrame(
        [
            make_seed_row("adaptive_mc", seed, delta)
            for seed, delta in enumerate((1.0, 2.0, 3.0, 4.0), start=1)
        ]
        + [
            make_seed_row("rule_based", seed, 0.0)
            for seed in range(1, 5)
        ]
    )

    statistics = calculate_paired_seed_statistics(
        rows,
        left_agent_name="adaptive_mc",
        right_agent_name="rule_based",
        opponent_name="calling",
        checkpoint_episode=1000,
    )

    assert statistics.common_seeds == (1, 2, 3, 4)
    assert statistics.deltas_by_seed == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
    }
    assert statistics.mean_delta == 2.5
    assert statistics.standard_deviation == pytest.approx(
        1.2909944487358056
    )
    assert statistics.standard_error == pytest.approx(
        0.6454972243679028
    )
    assert statistics.ci_lower == pytest.approx(0.4457397432394794)
    assert statistics.ci_upper == pytest.approx(4.554260256760521)


def test_paired_seed_statistics_report_unmatched_seeds_without_using_them():
    rows = pd.DataFrame(
        [
            make_seed_row("adaptive_mc", 1, 5.0),
            make_seed_row("adaptive_mc", 2, 8.0),
            make_seed_row("adaptive_mc", 3, 100.0),
            make_seed_row("rule_based", 1, 2.0),
            make_seed_row("rule_based", 2, 4.0),
            make_seed_row("rule_based", 4, -100.0),
        ]
    )

    statistics = calculate_paired_seed_statistics(
        rows,
        left_agent_name="adaptive_mc",
        right_agent_name="rule_based",
        opponent_name="calling",
        checkpoint_episode=1000,
    )

    assert statistics.common_seeds == (1, 2)
    assert statistics.left_only_seeds == (3,)
    assert statistics.right_only_seeds == (4,)
    assert statistics.deltas_by_seed == {1: 3.0, 2: 4.0}
    assert statistics.mean_delta == 3.5


def test_paired_seed_statistics_reject_duplicate_seed_rows():
    rows = pd.DataFrame(
        [
            make_seed_row("adaptive_mc", 1, 5.0),
            make_seed_row("adaptive_mc", 1, 6.0),
            make_seed_row("rule_based", 1, 2.0),
        ]
    )

    with pytest.raises(
        PairedSeedStatisticsError,
        match="duplicate model_seed values",
    ):
        calculate_paired_seed_statistics(
            rows,
            left_agent_name="adaptive_mc",
            right_agent_name="rule_based",
            opponent_name="calling",
            checkpoint_episode=1000,
        )

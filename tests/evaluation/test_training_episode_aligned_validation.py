import pandas as pd

from src.evaluation.algorithm_metadata import ALGORITHM_VALIDATION_SPECS
from src.evaluation.validation import (
    STATUS_FAIL,
    STATUS_SKIPPED,
    STATUS_WARNING,
    ValidationThresholds,
    validate_adaptive_beats_rule_based,
    validate_oracle_not_worse_than_adaptive,
)
from src.evaluation.validation.generalization_validation import (
    validate_generalization_adaptive_beats_agent,
)
from src.evaluation.validation.head_to_head_validation import (
    validate_adaptive_not_worse_than_general_rule_based,
)

SPEC = ALGORITHM_VALIDATION_SPECS[0]


def make_row(
    agent_name: str,
    opponent_name: str,
    training_episode: int,
    mean_profit_bb: float,
) -> dict[str, object]:
    return {
        "agent_name": agent_name,
        "opponent_name": opponent_name,
        "model_source": "final",
        "training_episode": training_episode,
        "mean_profit_bb": mean_profit_bb,
    }


def test_adaptive_rule_based_delta_uses_latest_common_training_episode():
    rows = pd.DataFrame(
        [
            make_row(SPEC.adaptive_agent, "calling", 1000, 20.0),
            make_row("rule_based", "calling", 1000, 0.0),
            make_row(SPEC.adaptive_agent, "calling", 2000, -2.0),
            make_row("rule_based", "calling", 2000, 3.0),
        ]
    )

    check = validate_adaptive_beats_rule_based(
        rows,
        ValidationThresholds(),
        opponents=("calling",),
        algorithm_specs=(SPEC,),
    )[0]

    assert check.status == STATUS_FAIL
    assert check.training_episode == 2000
    assert check.observed_value == -5.0
    assert check.details["rule_based_training_episode"] == 2000


def test_oracle_gap_skips_rows_without_a_common_training_episode():
    rows = pd.DataFrame(
        [
            make_row(SPEC.adaptive_agent, "tight", 1000, 5.0),
            make_row(SPEC.oracle_agent, "tight", 2000, 7.0),
        ]
    )

    check = validate_oracle_not_worse_than_adaptive(
        rows,
        ValidationThresholds(),
        opponents=("tight",),
        algorithm_specs=(SPEC,),
    )[0]

    assert check.status == STATUS_SKIPPED
    assert check.training_episode is None
    assert "No common training_episode" in check.message
    assert check.details["training_episodes_by_matchup"] == {
        f"{SPEC.oracle_agent} vs tight": [2000],
        f"{SPEC.adaptive_agent} vs tight": [1000],
    }


def test_generalization_delta_uses_latest_common_training_episode():
    opponent_name = "calling_loose"
    rows = pd.DataFrame(
        [
            make_row(SPEC.adaptive_agent, opponent_name, 1000, 10.0),
            make_row(SPEC.general_policy_agent, opponent_name, 1000, 0.0),
            make_row(SPEC.adaptive_agent, opponent_name, 2000, 1.0),
            make_row(SPEC.general_policy_agent, opponent_name, 2000, 4.0),
        ]
    )

    check = validate_generalization_adaptive_beats_agent(
        rows,
        ValidationThresholds(),
        min_successful_variants=1,
        check_name="Adaptive beats fixed general",
        category="generalization_adaptive_delta_vs_general",
        opponents=(opponent_name,),
        algorithm_specs=(SPEC,),
    )[0]

    assert check.status == STATUS_FAIL
    assert check.observed_value == 0.0
    assert check.details["deltas_by_variant"] == {opponent_name: -3.0}
    assert check.details["training_episodes_by_variant"] == {opponent_name: 2000}


def test_head_to_head_gap_uses_latest_common_training_episode():
    rows = pd.DataFrame(
        [
            make_row(SPEC.adaptive_agent, "rule_based", 1000, 10.0),
            make_row(SPEC.general_policy_agent, "rule_based", 1000, 0.0),
            make_row(SPEC.adaptive_agent, "rule_based", 2000, 1.0),
            make_row(SPEC.general_policy_agent, "rule_based", 2000, 4.0),
        ]
    )

    check = validate_adaptive_not_worse_than_general_rule_based(
        rows,
        ValidationThresholds(),
        algorithm_specs=(SPEC,),
    )[0]

    assert check.status == STATUS_WARNING
    assert check.training_episode == 2000
    assert check.observed_value == -3.0
    assert check.details["general_training_episode"] == 2000

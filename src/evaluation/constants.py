from src.poker.constants import (
    OPPONENT_TYPE_AGGRESSIVE,
    OPPONENT_TYPE_CALLING,
    OPPONENT_TYPE_FISH,
    OPPONENT_TYPE_UNKNOWN,
)

MODEL_DIRECTORY_BY_POLICY_TYPE = {
    OPPONENT_TYPE_UNKNOWN: "single_policy",
    OPPONENT_TYPE_FISH: "specialist_fish",
    OPPONENT_TYPE_AGGRESSIVE: "specialist_aggressive",
    OPPONENT_TYPE_CALLING: "specialist_calling",
}

CHECKPOINT_PREFIX_BY_POLICY_TYPE = {
    OPPONENT_TYPE_UNKNOWN: "single_policy",
    OPPONENT_TYPE_FISH: "specialist_fish",
    OPPONENT_TYPE_AGGRESSIVE: "specialist_aggressive",
    OPPONENT_TYPE_CALLING: "specialist_calling",
}

CROSS_POLICY_AGENT_TO_POLICY_TYPE = {
    "policy_unknown": OPPONENT_TYPE_UNKNOWN,
    "policy_fish": OPPONENT_TYPE_FISH,
    "policy_aggressive": OPPONENT_TYPE_AGGRESSIVE,
    "policy_calling": OPPONENT_TYPE_CALLING,
}

SUPPORTED_TESTED_AGENTS = {
    "rule_based",
    "single_policy_mc",
    "adaptive_mc",
    "oracle_adaptive",
    *CROSS_POLICY_AGENT_TO_POLICY_TYPE.keys(),
}

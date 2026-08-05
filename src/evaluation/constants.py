ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2

ACTION_LABELS = {
    ACTION_FOLD: "fold",
    ACTION_CALL: "call",
    ACTION_RAISE: "raise",
}

UNKNOWN_POLICY = "unknown"
FISH_POLICY = "fish"
AGGRESSIVE_POLICY = "aggressive"
CALLING_POLICY = "calling"

SUPPORTED_POLICY_TYPES = (
    UNKNOWN_POLICY,
    FISH_POLICY,
    AGGRESSIVE_POLICY,
    CALLING_POLICY,
)

SPECIALIST_POLICY_TYPES = (
    FISH_POLICY,
    AGGRESSIVE_POLICY,
    CALLING_POLICY,
)

MODEL_DIRECTORIES = {
    UNKNOWN_POLICY: "single_policy",
    FISH_POLICY: "specialist_fish",
    AGGRESSIVE_POLICY: "specialist_aggressive",
    CALLING_POLICY: "specialist_calling",
}

# Backward-compatible alias used by checkpoint_evaluator.py
MODEL_DIRECTORY_BY_POLICY_TYPE = MODEL_DIRECTORIES

CHECKPOINT_PREFIXES = {
    UNKNOWN_POLICY: "single_policy",
    FISH_POLICY: "specialist_fish",
    AGGRESSIVE_POLICY: "specialist_aggressive",
    CALLING_POLICY: "specialist_calling",
}

# Backward-compatible alias used by checkpoint_evaluator.py
CHECKPOINT_PREFIX_BY_POLICY_TYPE = CHECKPOINT_PREFIXES

POLICY_DISPLAY_NAMES = {
    UNKNOWN_POLICY: "General policy",
    FISH_POLICY: "Fish specialist",
    AGGRESSIVE_POLICY: "Aggressive specialist",
    CALLING_POLICY: "Calling specialist",
}

STATE_V2_FIELDS = (
    "street",
    "hand_strength_bin",
    "pair_strength_bin",
    "pot_bucket",
    "pot_odds_bin",
    "spr_bin",
    "opponent_type_id",
)

STATE_V2_ABSTRACT_FIELDS = STATE_V2_FIELDS[:-1]


RULE_BASED_AGENT = "rule_based"
ALWAYS_RAISE_AGENT = "always_raise"
ALWAYS_CALL_AGENT = "always_call"
SINGLE_POLICY_MC_AGENT = "single_policy_mc"
ADAPTIVE_MC_AGENT = "adaptive_mc"
ORACLE_ADAPTIVE_AGENT = "oracle_adaptive"

POLICY_UNKNOWN_AGENT = "policy_unknown"
POLICY_FISH_AGENT = "policy_fish"
POLICY_AGGRESSIVE_AGENT = "policy_aggressive"
POLICY_CALLING_AGENT = "policy_calling"

SUPPORTED_TESTED_AGENTS = {
    RULE_BASED_AGENT,
    ALWAYS_RAISE_AGENT,
    ALWAYS_CALL_AGENT,
    SINGLE_POLICY_MC_AGENT,
    ADAPTIVE_MC_AGENT,
    ORACLE_ADAPTIVE_AGENT,
    POLICY_UNKNOWN_AGENT,
    POLICY_FISH_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
}

CROSS_POLICY_AGENT_TO_POLICY_TYPE = {
    POLICY_UNKNOWN_AGENT: UNKNOWN_POLICY,
    POLICY_FISH_AGENT: FISH_POLICY,
    POLICY_AGGRESSIVE_AGENT: AGGRESSIVE_POLICY,
    POLICY_CALLING_AGENT: CALLING_POLICY,
}

AGENT_DISPLAY_NAMES = {
    RULE_BASED_AGENT: "Rule-based baseline",
    ALWAYS_RAISE_AGENT: "Always-raise baseline",
    ALWAYS_CALL_AGENT: "Always-call baseline",
    SINGLE_POLICY_MC_AGENT: "Single-policy Monte Carlo",
    ADAPTIVE_MC_AGENT: "Adaptive Monte Carlo",
    ORACLE_ADAPTIVE_AGENT: "Oracle adaptive",
    POLICY_UNKNOWN_AGENT: "Fixed general policy",
    POLICY_FISH_AGENT: "Fixed fish specialist",
    POLICY_AGGRESSIVE_AGENT: "Fixed aggressive specialist",
    POLICY_CALLING_AGENT: "Fixed calling specialist",
}
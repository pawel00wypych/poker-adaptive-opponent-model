ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_RAISE = 2

ACTION_LABELS = {
    ACTION_FOLD: "fold",
    ACTION_CALL: "call",
    ACTION_RAISE: "raise",
}

UNKNOWN_POLICY = "unknown"
TIGHT_POLICY = "tight"
AGGRESSIVE_POLICY = "aggressive"
CALLING_POLICY = "calling"

SUPPORTED_POLICY_TYPES = (
    UNKNOWN_POLICY,
    TIGHT_POLICY,
    AGGRESSIVE_POLICY,
    CALLING_POLICY,
)

SPECIALIST_POLICY_TYPES = (
    TIGHT_POLICY,
    AGGRESSIVE_POLICY,
    CALLING_POLICY,
)

MODEL_DIRECTORIES = {
    UNKNOWN_POLICY: "general_policy",
    TIGHT_POLICY: "specialist_tight",
    AGGRESSIVE_POLICY: "specialist_aggressive",
    CALLING_POLICY: "specialist_calling",
}

CHECKPOINT_PREFIXES = {
    UNKNOWN_POLICY: "general_policy",
    TIGHT_POLICY: "specialist_tight",
    AGGRESSIVE_POLICY: "specialist_aggressive",
    CALLING_POLICY: "specialist_calling",
}

POLICY_DISPLAY_NAMES = {
    UNKNOWN_POLICY: "General policy",
    TIGHT_POLICY: "Tight specialist",
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
    "is_small_blind",
)


RULE_BASED_AGENT = "rule_based"
ALWAYS_RAISE_AGENT = "always_raise"
ALWAYS_CALL_AGENT = "always_call"
POLICY_GENERAL_MC_AGENT = "policy_general_mc"
ADAPTIVE_MC_AGENT = "adaptive_mc"
ORACLE_MC_AGENT = "oracle_mc"
ORACLE_Q_LEARNING_AGENT = "oracle_q_learning"
ORACLE_SARSA_AGENT = "oracle_sarsa"
ORACLE_DOUBLE_Q_LEARNING_AGENT = "oracle_double_q_learning"
ADAPTIVE_Q_LEARNING_AGENT = "adaptive_q_learning"
ADAPTIVE_SARSA_AGENT = "adaptive_sarsa"
ADAPTIVE_DOUBLE_Q_LEARNING_AGENT = "adaptive_double_q_learning"

POLICY_GENERAL_Q_LEARNING_AGENT = "policy_general_q_learning"
POLICY_GENERAL_SARSA_AGENT = "policy_general_sarsa"
POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT = "policy_general_double_q_learning"

# Fixed specialist policies: a single specialist model played against every
# opponent, with no classification and no switching. They answer "does a
# specialist generalise outside the family it was trained on?", which is the
# empirical case for adaptation. The unsuffixed names are Monte Carlo for
# backwards compatibility with existing result files.
POLICY_TIGHT_AGENT = "policy_tight"
POLICY_AGGRESSIVE_AGENT = "policy_aggressive"
POLICY_CALLING_AGENT = "policy_calling"

POLICY_TIGHT_Q_LEARNING_AGENT = "policy_tight_q_learning"
POLICY_AGGRESSIVE_Q_LEARNING_AGENT = "policy_aggressive_q_learning"
POLICY_CALLING_Q_LEARNING_AGENT = "policy_calling_q_learning"

POLICY_TIGHT_SARSA_AGENT = "policy_tight_sarsa"
POLICY_AGGRESSIVE_SARSA_AGENT = "policy_aggressive_sarsa"
POLICY_CALLING_SARSA_AGENT = "policy_calling_sarsa"

POLICY_TIGHT_DOUBLE_Q_LEARNING_AGENT = "policy_tight_double_q_learning"
POLICY_AGGRESSIVE_DOUBLE_Q_LEARNING_AGENT = "policy_aggressive_double_q_learning"
POLICY_CALLING_DOUBLE_Q_LEARNING_AGENT = "policy_calling_double_q_learning"

MONTE_CARLO_SPECIALIST_AGENTS = (
    POLICY_TIGHT_AGENT,
    POLICY_AGGRESSIVE_AGENT,
    POLICY_CALLING_AGENT,
)

Q_LEARNING_SPECIALIST_AGENTS = (
    POLICY_TIGHT_Q_LEARNING_AGENT,
    POLICY_AGGRESSIVE_Q_LEARNING_AGENT,
    POLICY_CALLING_Q_LEARNING_AGENT,
)

SARSA_SPECIALIST_AGENTS = (
    POLICY_TIGHT_SARSA_AGENT,
    POLICY_AGGRESSIVE_SARSA_AGENT,
    POLICY_CALLING_SARSA_AGENT,
)

DOUBLE_Q_LEARNING_SPECIALIST_AGENTS = (
    POLICY_TIGHT_DOUBLE_Q_LEARNING_AGENT,
    POLICY_AGGRESSIVE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_CALLING_DOUBLE_Q_LEARNING_AGENT,
)

FIXED_SPECIALIST_AGENTS = (
    *MONTE_CARLO_SPECIALIST_AGENTS,
    *Q_LEARNING_SPECIALIST_AGENTS,
    *SARSA_SPECIALIST_AGENTS,
    *DOUBLE_Q_LEARNING_SPECIALIST_AGENTS,
)

SUPPORTED_TESTED_AGENTS = {
    RULE_BASED_AGENT,
    ALWAYS_RAISE_AGENT,
    ALWAYS_CALL_AGENT,
    ADAPTIVE_MC_AGENT,
    ORACLE_MC_AGENT,
    ORACLE_Q_LEARNING_AGENT,
    ORACLE_SARSA_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT,
    POLICY_GENERAL_MC_AGENT,
    POLICY_GENERAL_Q_LEARNING_AGENT,
    POLICY_GENERAL_SARSA_AGENT,
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT,
    *FIXED_SPECIALIST_AGENTS,
}

# One mapping per algorithm, each covering the same four policy types. The
# Monte Carlo one used to be called CROSS_POLICY_..., which read as if it
# spanned algorithms; it does not, and that name is part of why the missing
# TD specialists went unnoticed.
MONTE_CARLO_POLICY_AGENT_TO_POLICY_TYPE = {
    POLICY_GENERAL_MC_AGENT: UNKNOWN_POLICY,
    POLICY_TIGHT_AGENT: TIGHT_POLICY,
    POLICY_AGGRESSIVE_AGENT: AGGRESSIVE_POLICY,
    POLICY_CALLING_AGENT: CALLING_POLICY,
}

Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE = {
    POLICY_GENERAL_Q_LEARNING_AGENT: UNKNOWN_POLICY,
    POLICY_TIGHT_Q_LEARNING_AGENT: TIGHT_POLICY,
    POLICY_AGGRESSIVE_Q_LEARNING_AGENT: AGGRESSIVE_POLICY,
    POLICY_CALLING_Q_LEARNING_AGENT: CALLING_POLICY,
}

SARSA_POLICY_AGENT_TO_POLICY_TYPE = {
    POLICY_GENERAL_SARSA_AGENT: UNKNOWN_POLICY,
    POLICY_TIGHT_SARSA_AGENT: TIGHT_POLICY,
    POLICY_AGGRESSIVE_SARSA_AGENT: AGGRESSIVE_POLICY,
    POLICY_CALLING_SARSA_AGENT: CALLING_POLICY,
}

DOUBLE_Q_LEARNING_POLICY_AGENT_TO_POLICY_TYPE = {
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT: UNKNOWN_POLICY,
    POLICY_TIGHT_DOUBLE_Q_LEARNING_AGENT: TIGHT_POLICY,
    POLICY_AGGRESSIVE_DOUBLE_Q_LEARNING_AGENT: AGGRESSIVE_POLICY,
    POLICY_CALLING_DOUBLE_Q_LEARNING_AGENT: CALLING_POLICY,
}

AGENT_DISPLAY_NAMES = {
    RULE_BASED_AGENT: "Rule-based baseline",
    ALWAYS_RAISE_AGENT: "Always-raise baseline",
    ALWAYS_CALL_AGENT: "Always-call baseline",
    ADAPTIVE_MC_AGENT: "Adaptive Monte Carlo",
    ORACLE_MC_AGENT: "Oracle Monte Carlo",
    ORACLE_Q_LEARNING_AGENT: "Oracle Q-learning",
    ORACLE_SARSA_AGENT: "Oracle SARSA",
    ORACLE_DOUBLE_Q_LEARNING_AGENT: "Oracle Double Q-learning",
    ADAPTIVE_Q_LEARNING_AGENT: "Adaptive Q-learning",
    ADAPTIVE_SARSA_AGENT: "Adaptive SARSA",
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT: "Adaptive Double Q-learning",
    POLICY_GENERAL_MC_AGENT: "Fixed general Monte Carlo policy",
    POLICY_GENERAL_Q_LEARNING_AGENT: "Fixed general Q-learning policy",
    POLICY_GENERAL_SARSA_AGENT: "Fixed general SARSA policy",
    POLICY_GENERAL_DOUBLE_Q_LEARNING_AGENT: "Fixed general Double Q-learning policy",
    POLICY_TIGHT_AGENT: "Fixed tight specialist (Monte Carlo)",
    POLICY_AGGRESSIVE_AGENT: "Fixed aggressive specialist (Monte Carlo)",
    POLICY_CALLING_AGENT: "Fixed calling specialist (Monte Carlo)",
    POLICY_TIGHT_Q_LEARNING_AGENT: "Fixed tight specialist (Q-learning)",
    POLICY_AGGRESSIVE_Q_LEARNING_AGENT: "Fixed aggressive specialist (Q-learning)",
    POLICY_CALLING_Q_LEARNING_AGENT: "Fixed calling specialist (Q-learning)",
    POLICY_TIGHT_SARSA_AGENT: "Fixed tight specialist (SARSA)",
    POLICY_AGGRESSIVE_SARSA_AGENT: "Fixed aggressive specialist (SARSA)",
    POLICY_CALLING_SARSA_AGENT: "Fixed calling specialist (SARSA)",
    POLICY_TIGHT_DOUBLE_Q_LEARNING_AGENT: (
        "Fixed tight specialist (Double Q-learning)"
    ),
    POLICY_AGGRESSIVE_DOUBLE_Q_LEARNING_AGENT: (
        "Fixed aggressive specialist (Double Q-learning)"
    ),
    POLICY_CALLING_DOUBLE_Q_LEARNING_AGENT: (
        "Fixed calling specialist (Double Q-learning)"
    ),
}

ADAPTIVE_AGENT_TO_ORACLE_AGENT = {
    ADAPTIVE_MC_AGENT: ORACLE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT: ORACLE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT: ORACLE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT: ORACLE_DOUBLE_Q_LEARNING_AGENT,
}

ORACLE_AGENT_TO_ADAPTIVE_AGENT = {
    oracle_agent: adaptive_agent
    for adaptive_agent, oracle_agent in ADAPTIVE_AGENT_TO_ORACLE_AGENT.items()
}

ORACLE_AGENTS = tuple(ORACLE_AGENT_TO_ADAPTIVE_AGENT.keys())

# The Oracle gap measures what an agent loses by classifying the opponent
# imperfectly and switching policies late. It is only defined for agents that
# actually switch. From final_experiment_guidelines.md:
#
#   "The Oracle is not meaningful for policy_general_*, rule_based,
#    always_call, and always_raise, because these agents do not switch
#    between specialists."
#
# The same reasoning excludes the fixed specialists: they play one policy for
# the whole game and never classify anything. Do not re-add them - a number
# computed for those rows has no interpretation, and a reader cannot tell that
# from looking at it.
AGENT_TO_ORACLE_AGENT = {
    ADAPTIVE_MC_AGENT: ORACLE_MC_AGENT,
    ORACLE_MC_AGENT: ORACLE_MC_AGENT,
    ADAPTIVE_Q_LEARNING_AGENT: ORACLE_Q_LEARNING_AGENT,
    ORACLE_Q_LEARNING_AGENT: ORACLE_Q_LEARNING_AGENT,
    ADAPTIVE_SARSA_AGENT: ORACLE_SARSA_AGENT,
    ORACLE_SARSA_AGENT: ORACLE_SARSA_AGENT,
    ADAPTIVE_DOUBLE_Q_LEARNING_AGENT: ORACLE_DOUBLE_Q_LEARNING_AGENT,
    ORACLE_DOUBLE_Q_LEARNING_AGENT: ORACLE_DOUBLE_Q_LEARNING_AGENT,
}

POLICY_SWITCHING_AGENTS = tuple(AGENT_TO_ORACLE_AGENT.keys())


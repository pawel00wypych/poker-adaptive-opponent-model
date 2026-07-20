METRIC_DESCRIPTIONS = {
    "checkpoint_episode": "Number of training episodes completed when the checkpoint was saved.",
    "model_seed": "Random seed used for training a given model checkpoint.",
    "games": "Number of evaluated games in the group.",
    "total_profit_bb": "Total profit measured in big blinds across all evaluated games.",
    "mean_profit_bb": "Average profit per game, measured in big blinds. Higher is better.",
    "std_profit_bb": "Standard deviation of per-game profit in big blinds. Higher values mean less stable results.",
    "bb_per_100": "Profit normalized to 100 played hands. Higher is better.",
    "win_rate": "Percentage of evaluated games won by the tested agent.",
    "bust_rate": "Percentage of games in which the tested agent busted.",
    "ended_by_bust_rate": "Percentage of games that ended because a player busted.",
    "ended_by_round_limit_rate": "Percentage of games that ended by reaching the configured round limit.",
    "standard_error": "Estimated uncertainty of mean_profit_bb for a group of games.",
    "ci_95_lower": "Lower bound of the approximate 95% confidence interval for mean_profit_bb.",
    "ci_95_upper": "Upper bound of the approximate 95% confidence interval for mean_profit_bb.",
    "global_classifier_accuracy": "Classifier accuracy computed from total correct and incorrect classifications. Applies mainly to adaptive agents.",
    "global_classifier_coverage": "Percentage of classification opportunities where the classifier returned a known opponent type instead of unknown.",
    "mean_policy_switches": "Average number of policy switches per game. Applies mainly to adaptive agents.",
    "mean_first_classification_hand": "Average hand number of the first known opponent classification.",
    "mean_first_correct_classification_hand": "Average hand number of the first correct known opponent classification.",
    "states": "Number of unique states stored in the Q-table.",
    "fully_zero_states": "States where all action values are exactly zero.",
    "tied_best_states": "States where at least two actions share the same best Q-value.",
    "best_action_agreement_rate": "Percentage of common states where two policies choose the same best action.",
    "mean_max_abs_q_delta": "Average largest absolute Q-value difference per common state.",
}

AGENT_LABELS = {
    "adaptive_mc": "Adaptive Monte Carlo",
    "oracle_adaptive": "Oracle adaptive",
    "single_policy_mc": "Single-policy Monte Carlo",
    "policy_unknown": "Fixed unknown policy",
    "policy_calling": "Fixed calling specialist",
    "policy_fish": "Fixed fish specialist",
    "policy_aggressive": "Fixed aggressive specialist",
    "rule_based": "Rule-based baseline",
}

ACTION_LABELS = {
    "fold": "Fold",
    "call": "Call",
    "raise": "Raise",
}

STATE_FIELD_DESCRIPTIONS = {
    "street": "Betting street encoded as a small integer, for example preflop/flop/turn/river.",
    "hand_strength_bin": "Discretized hand strength bucket estimated from cards and board.",
    "pair_strength_bin": "Contextual pair-strength bucket, for example no pair, top pair, overpair, or two-pair-or-better.",
    "pot_bucket": "Discretized pot-size bucket.",
    "pot_odds_bin": "Discretized cost-to-call bucket relative to the pot.",
    "spr_bin": "Stack-to-pot-ratio bucket. Lower SPR means stacks are shallow relative to the pot.",
    "opponent_type_id": "Encoded opponent type used in the Q-table state. Reports can strip this field when comparing policies across opponent labels.",
}

REPORT_INTRODUCTION = (
    "This report is generated from raw experiment outputs. CSV and JSON files remain "
    "the source of truth; the report only aggregates them into tables, descriptions, "
    "and plots that are easier to inspect and include in thesis notes."
)

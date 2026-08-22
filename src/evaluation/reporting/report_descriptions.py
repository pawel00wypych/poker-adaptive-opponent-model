METRIC_DESCRIPTIONS = {
    "training_episode": (
        "Number of training episodes completed by the evaluated final model."
    ),
    "checkpoint_episode": (
        "Number of training episodes completed when a diagnostic checkpoint was saved."
    ),
    "model_seed": "Random seed used for training a given model.",
    "games": "Number of evaluated games in the group.",
    "total_profit_bb": (
        "Total profit measured in big blinds across all evaluated games."
    ),
    "mean_profit_bb": (
        "Average profit per game, measured in big blinds. Higher is better."
    ),
    "oracle_gap_bb": (
        "Oracle mean profit minus adaptive mean profit, measured in big "
        "blinds per game. Positive values indicate adaptive "
        "underperformance relative to Oracle."
    ),
    "std_profit_bb": (
        "Standard deviation of per-game profit in big blinds. Higher values mean less "
        "stable results."
    ),
    "bb_per_100": "Profit normalized to 100 played hands. Higher is better.",
    "win_rate": "Percentage of evaluated games won by the tested agent.",
    "bust_rate": "Percentage of games in which the tested agent busted.",
    "ended_by_bust_rate": "Percentage of games that ended because a player busted.",
    "ended_by_round_limit_rate": (
        "Percentage of games that ended by reaching the configured round limit."
    ),
    "standard_error": "Estimated uncertainty of mean_profit_bb for a group of games.",
    "ci_95_lower": (
        "Lower bound of the approximate 95% confidence interval for mean_profit_bb."
    ),
    "ci_95_upper": (
        "Upper bound of the approximate 95% confidence interval for mean_profit_bb."
    ),
    "mean_profit_bb_std_across_seeds": (
        "Sample standard deviation of per-seed mean profit across independent training "
        "seeds."
    ),
    "mean_profit_bb_standard_error_across_seeds": (
        "Standard error of mean_profit_bb calculated from equally weighted "
        "training-seed means."
    ),
    "mean_profit_bb_ci_95_lower_across_seeds": (
        "Lower bound of the 95% Student-t confidence interval across training seeds."
    ),
    "mean_profit_bb_ci_95_upper_across_seeds": (
        "Upper bound of the 95% Student-t confidence interval across training seeds."
    ),
    "mean_profit_bb_ci_95_margin_across_seeds": (
        "Margin of error of the 95% Student-t confidence interval across training "
        "seeds."
    ),
    "mean_profit_bb_min_across_seeds": (
        "Lowest per-seed mean profit in the aggregated group."
    ),
    "mean_profit_bb_max_across_seeds": (
        "Highest per-seed mean profit in the aggregated group."
    ),
    "mean_profit_bb_seed_spread": (
        "Difference between the highest and lowest per-seed mean profit."
    ),
    "global_classifier_accuracy": (
        "Classifier accuracy computed from total correct and incorrect "
        "classifications. Applies mainly to adaptive agents."
    ),
    "global_classifier_coverage": (
        "Percentage of classification opportunities where a specialist policy was "
        "selected. Decisions classified as unknown or other stay in the denominator "
        "because no specialist was used."
    ),
    "global_other_rate": (
        "Percentage of classification opportunities where the opponent matched no "
        "known family (other), so the general policy was used."
    ),
    "unseen_state_decision_rate": (
        "Percentage of decisions taken in a state where nothing had been "
        "learned. All Q-values are zero there, so the greedy choice "
        "degenerates into a uniform random pick over the legal actions."
    ),
    "untried_action_selection_rate": (
        "Percentage of decisions that selected an action never tried in that "
        "state. Zero-initialised entries outrank actions already learned to "
        "be losing, so this measures how often the choice reflects "
        "initialisation rather than experience."
    ),
    "mean_policy_switches": (
        "Average number of policy switches per game. Applies mainly to adaptive "
        "agents."
    ),
    "mean_first_classification_hand": (
        "Average hand number of the first known opponent classification."
    ),
    "mean_first_correct_classification_hand": (
        "Average hand number of the first correct known opponent classification."
    ),
    "states": "Number of unique states stored in the Q-table.",
    "fully_zero_states": "States where all action values are exactly zero.",
    "tied_best_states": (
        "States where at least two actions share the same best Q-value."
    ),
    "best_action_agreement_rate": (
        "Percentage of common states where two policies choose the same best action."
    ),
    "mean_max_abs_q_delta": (
        "Average largest absolute Q-value difference per common state."
    ),
}

AGENT_LABELS = {
    "adaptive_mc": "Adaptive Monte Carlo",
    "adaptive_q_learning": "Adaptive Q-learning",
    "adaptive_sarsa": "Adaptive SARSA",
    "adaptive_double_q_learning": "Adaptive Double Q-learning",
    "oracle_mc": "Oracle Monte Carlo",
    "oracle_q_learning": "Oracle Q-learning",
    "oracle_sarsa": "Oracle SARSA",
    "oracle_double_q_learning": "Oracle Double Q-learning",
    "policy_general_mc": "Fixed general Monte Carlo policy",
    "policy_general_q_learning": "Fixed general Q-learning policy",
    "policy_general_sarsa": "Fixed general SARSA policy",
    "policy_general_double_q_learning": "Fixed general Double Q-learning policy",
    "policy_calling": "Fixed calling specialist",
    "policy_tight": "Fixed tight specialist",
    "policy_aggressive": "Fixed aggressive specialist",
    "rule_based": "Rule-based baseline",
    "always_raise": "Always-raise baseline",
    "always_call": "Always-call baseline",
}

ACTION_LABELS = {
    "fold": "Fold",
    "call": "Call",
    "raise": "Raise",
}

STATE_FIELD_DESCRIPTIONS = {
    "street": (
        "Betting street encoded as a small integer, for example "
        "preflop/flop/turn/river."
    ),
    "hand_strength_bin": (
        "Discretized hand strength bucket estimated from cards and board."
    ),
    "pair_strength_bin": (
        "Contextual pair-strength bucket, for example no pair, top pair, overpair, or "
        "two-pair-or-better."
    ),
    "pot_bucket": "Discretized pot-size bucket.",
    "pot_odds_bin": "Discretized cost-to-call bucket relative to the pot.",
    "spr_bin": (
        "Stack-to-pot-ratio bucket. Lower SPR means stacks are shallow relative to the "
        "pot."
    ),
    "opponent_type_id": (
        "Encoded opponent type used in the Q-table state. Reports can strip this field "
        "when comparing policies across opponent labels."
    ),
}

REPORT_INTRODUCTION = (
    "This report is generated from raw experiment outputs. CSV and JSON files remain "
    "the source of truth; the report only aggregates them into tables, descriptions, "
    "and plots that are easier to inspect and include in thesis notes."
)

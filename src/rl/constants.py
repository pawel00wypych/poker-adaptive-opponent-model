DEFAULT_NUM_ACTIONS = 3
Q_TABLE_KEY = "q_table"
VISIT_COUNTS_KEY = "visit_counts"
METADATA_KEY = "metadata"
ALGORITHM_KEY = "algorithm"

Q1_TABLE_KEY = "q1_table"
Q2_TABLE_KEY = "q2_table"
Q1_VISIT_COUNTS_KEY = "q1_visit_counts"
Q2_VISIT_COUNTS_KEY = "q2_visit_counts"

# Identifies the algorithm that produced a saved model. It is written into the
# pickle payload and into the sidecar metadata, and verified on load so that a
# model cannot be silently attributed to another algorithm.
ALGORITHM_MONTE_CARLO = "first_visit_monte_carlo_control"
ALGORITHM_Q_LEARNING = "q_learning"
ALGORITHM_SARSA = "sarsa"
ALGORITHM_DOUBLE_Q_LEARNING = "double_q_learning"

SUPPORTED_MODEL_ALGORITHMS = (
    ALGORITHM_MONTE_CARLO,
    ALGORITHM_Q_LEARNING,
    ALGORITHM_SARSA,
    ALGORITHM_DOUBLE_Q_LEARNING,
)

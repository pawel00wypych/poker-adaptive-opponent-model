EPSILON_SCHEDULE_LINEAR = "linear"
EPSILON_SCHEDULE_EXPONENTIAL = "exponential"

SUPPORTED_EPSILON_SCHEDULES = (
    EPSILON_SCHEDULE_LINEAR,
    EPSILON_SCHEDULE_EXPONENTIAL,
)

ALPHA_MODE_CONSTANT = "constant"
ALPHA_MODE_VISIT_COUNT = "visit_count"
ALPHA_MODE_SQRT_VISIT = "sqrt_visit"

MODEL_TYPE_GENERAL_POLICY = "general_policy"
MODEL_TYPE_SPECIALIST = "specialist"

# Directory segment separating one algorithm's artefacts from another's.
# These must stay equal to the ALGORITHM_KEY_* values in
# src/evaluation/algorithm_metadata.py; src/training does not import from
# src/evaluation, so tests/training/test_model_paths.py keeps them in step.
ALGORITHM_KEY_MONTE_CARLO = "monte_carlo"
ALGORITHM_KEY_Q_LEARNING = "q_learning"
ALGORITHM_KEY_SARSA = "sarsa"
ALGORITHM_KEY_DOUBLE_Q_LEARNING = "double_q_learning"

ALGORITHM_KEYS = (
    ALGORITHM_KEY_MONTE_CARLO,
    ALGORITHM_KEY_Q_LEARNING,
    ALGORITHM_KEY_SARSA,
    ALGORITHM_KEY_DOUBLE_Q_LEARNING,
)

SUPPORTED_ALPHA_MODES = (
    ALPHA_MODE_CONSTANT,
    ALPHA_MODE_VISIT_COUNT,
    ALPHA_MODE_SQRT_VISIT,
)

# Poker Adaptive Opponent Model

Research project for testing adaptive strategy selection in heads-up Texas Hold'em.
The project combines behavioural opponent classification with tabular
reinforcement-learning agents.

The project is developed as part of a master's thesis:

> Dynamic strategy adaptation in imperfect-information games using
> reinforcement learning and behavioural classification

## Project Goal

The main goal is to evaluate whether an agent that identifies an opponent's
playing style and switches to a matching policy can perform better than
non-adaptive policies and simple rule-based baselines.

The experiments are run in heads-up poker, where one evaluated agent plays
against one fixed opponent. This keeps the environment focused on opponent
modelling, strategy adaptation and algorithm-level RL comparison instead of
multi-player table dynamics.

The current experimental setup also includes sanity checks and generalization
tests. These checks are used to verify whether the observed results come from
meaningful adaptation and learning, or from exploiting overly simple scripted
opponents.

## Implemented Scope

- Heads-up Texas Hold'em simulation based on a local `PyPokerEngine` dependency.
- Shared tabular RL utilities for Q-tables, legal-action handling,
  epsilon-greedy action selection and model I/O.
- Tabular Monte Carlo control agent.
- Tabular Q-learning agent.
- Tabular SARSA agent.
- Tabular Double Q-learning agent.
- Single-policy agents trained against mixed opponents.
- Specialist policies trained against one base opponent type.
- Adaptive player that uses behavioural classification to select a policy.
- Oracle adaptive baseline for selected evaluation setups.
- Fixed-policy evaluation players for unknown, fish, aggressive and calling policies.
- Rule-based baseline player.
- Always-call and always-raise sanity baselines.
- Fixed base opponent strategies: fish, aggressive and calling.
- Named generalization opponents:
  - `tight_extreme`
  - `calling_extreme`
  - `aggressive_extreme`
- Discretised state representation with hand strength, poker context, pot
  information, stack-to-pot ratio and opponent type.
- Multi-seed training, checkpoint saving and checkpoint evaluation.
- Generalization and stress-test evaluation on base and variant opponents.
- Direct head-to-head sanity evaluation.
- Experiment summary reports with rankings, baseline deltas, traffic-light
  quality statuses and automatic main findings.
- Dedicated RL algorithm comparison report for Monte Carlo, Q-learning, SARSA
  and Double Q-learning.
- CSV, JSON, Markdown and LaTeX result outputs.
- Plots for experiment analysis, including mean-profit and seed-stability charts.
- Automated validation checks for experiment quality and result stability.
- Automated tests for agents, state encoding, metrics, reports, validation,
  opponent variants and evaluation helpers.

## Project Structure

```text
src/
├── agents/           # Reinforcement-learning agents
├── cards/            # Card and hand-strength utilities
├── evaluation/       # Metrics, evaluation, validation, reports and plots
├── experiments/      # CLI scripts for training, evaluation and reporting
├── features/         # State encoding and opponent statistics
├── opponent_model/   # Behavioural opponent classifier
├── players/          # PyPokerEngine player implementations
├── poker/            # Poker action and round-state helpers
├── rl/               # Shared tabular RL utilities
├── training/         # Training schedules, checkpointing and metadata
└── config.py         # Main game, training and evaluation configuration

tests/                # Automated test suite
PyPokerEngine/        # Local poker engine dependency
results/              # Generated models and experiment results
reports/              # Generated readable reports
docs/                 # Thesis and experiment notes
```

Shared enum-like values are kept in small `constants.py` files inside the
relevant packages, for example `src/rl/constants.py`, `src/poker/constants.py`,
`src/training/constants.py`, `src/experiments/constants.py`,
`src/evaluation/constants.py` and `src/players/constants.py`.

## Requirements

- Python 3.11 (enforced — see below)
- pip
- Git

The supported interpreter is checked at import time in `src/__init__.py` and
declared as `requires-python = ">=3.11,<3.12"` in `pyproject.toml`. Running any
command or test on a different Python version fails immediately with an
explanatory error rather than producing results that silently differ between
versions.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd poker-adaptive-opponent-model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs `PyPokerEngine` from the pinned Git revision listed in
`requirements.txt`; no separate engine checkout is needed.

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate poker_env_311
```

Verify the installation:

```bash
python -m pytest -q
python -m ruff check .
```

Some dependencies are not imported by this project directly but are still
required:

- `scipy` — Student-t statistics used for seed-level confidence intervals.
- `jinja2` — required by pandas for `DataFrame.to_latex()`, which renders the
  LaTeX result tables.
- `ruff` — enforces the lint configuration declared in `pyproject.toml`.

All commands below should be run from the project root.

## Configuration

Default parameters are defined in `src/config.py`:

- `GameConfig`: number of rounds, initial stack and blind size.
- `TrainingConfig`: episodes, alpha, epsilon settings, seed, checkpoints and
  model paths.
- `EvaluationConfig`: number of evaluation games and output CSV path.

Most training and evaluation scripts expose CLI flags such as `--episodes`,
`--seeds`, `--epsilon-schedule`, `--checkpoint-episodes`, `--games`,
`--output-path`, `--training-run-dir`, `--q-learning-run-dir`,
`--sarsa-run-dir`, `--double-q-learning-run-dir` and `--workers`.

## RL Algorithms

The project currently compares four tabular reinforcement-learning algorithms.
All of them use the same environment, state representation, action mapping,
legal-action handling and reward definition.

### Monte Carlo

Monte Carlo is the original baseline used in the project. It updates action
values from complete episodes. This keeps the implementation simple and
interpretable, but learning can be slower because value updates are delayed
until an episode is finished.

### Q-learning

Q-learning is an off-policy temporal-difference method. It updates action values
step by step using the maximum estimated value of the next state. In the current
experiments, Q-learning provides the strongest average adaptive performance.

### SARSA

SARSA is an on-policy temporal-difference method. It updates action values using
the actually selected next action. This can produce more conservative behaviour
than Q-learning because the update follows the policy currently being executed.

### Double Q-learning

Double Q-learning uses two value tables, `Q1` and `Q2`. On each transition, one
table is updated while the other table is used to evaluate the selected next
action. This reduces overestimation bias compared with standard Q-learning. In
the current algorithm-level comparison, Double Q-learning is especially useful
against the most aggressive opponent variant.

## Training

### Monte Carlo training suite

Run a reproducible multi-seed Monte Carlo training suite for the general policy
and all specialists:

```powershell
python -m src.experiments.run_training_suite `
  --episodes 1000 `
  --seeds 42 123 456 `
  --models single_policy fish aggressive calling `
  --checkpoint-episodes 1000 `
  --epsilon-schedule linear `
  --alpha-mode sqrt_visit `
  --output-root results/training_runs `
  --experiment-name state_v2_linear_1000_sqrt_visit `
  --workers 4
```

### Q-learning training

```powershell
python -m src.experiments.run_q_learning_training `
  --episodes 1000 `
  --seeds 42 123 456 `
  --output-dir results/training_runs/q_learning_1000 `
  --checkpoint-episodes 1000 `
  --models single_policy fish aggressive calling
```

### SARSA training

```powershell
python -m src.experiments.run_sarsa_training `
  --episodes 1000 `
  --seeds 42 123 456 `
  --output-dir results/training_runs/sarsa_1000 `
  --checkpoint-episodes 1000 `
  --models single_policy fish aggressive calling
```

### Double Q-learning training

```powershell
python -m src.experiments.run_double_q_learning_training `
  --episodes 1000 `
  --seeds 42 123 456 `
  --output-dir results/training_runs/double_q_learning_1000 `
  --checkpoint-episodes 1000 `
  --models single_policy fish aggressive calling
```

Supported specialist opponents are `fish`, `aggressive` and `calling`.
Generalization variants are intentionally not used as specialist training
opponents.

## Evaluation

### General checkpoint evaluation

```powershell
python -m src.experiments.run_checkpoint_evaluation `
  --training-run-dir results/training_runs/state_v2_linear_1000_sqrt_visit `
  --q-learning-run-dir results/training_runs/q_learning_1000 `
  --sarsa-run-dir results/training_runs/sarsa_1000 `
  --double-q-learning-run-dir results/training_runs/double_q_learning_1000 `
  --checkpoint-episodes 1000 `
  --seeds 42 123 456 `
  --games 200 `
  --agents adaptive_mc adaptive_q_learning adaptive_sarsa adaptive_double_q_learning policy_unknown_mc policy_unknown_q_learning policy_unknown_sarsa policy_unknown_double_q_learning rule_based always_call always_raise `
  --output-path results/evaluation/checkpoint_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000.csv
```

### Generalization evaluation

```powershell
python -m src.experiments.run_generalization_evaluation `
  --training-run-dir results/training_runs/state_v2_linear_1000_sqrt_visit `
  --q-learning-run-dir results/training_runs/q_learning_1000 `
  --sarsa-run-dir results/training_runs/sarsa_1000 `
  --double-q-learning-run-dir results/training_runs/double_q_learning_1000 `
  --checkpoint-episodes 1000 `
  --seeds 42 123 456 `
  --games 200 `
  --agents adaptive_mc adaptive_q_learning adaptive_sarsa adaptive_double_q_learning policy_unknown_mc policy_unknown_q_learning policy_unknown_sarsa policy_unknown_double_q_learning rule_based always_call always_raise `
  --output-path results/evaluation/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000.csv
```

The generalization setup evaluates agents trained on the base opponents against
a small set of base and variant opponents used for robustness and stress testing.
It does not train new variant-specific specialist policies.

## Reports and Plots

### Experiment summary

Create an experiment summary report with rankings, deltas, traffic-light
statuses, main findings, tables and charts:

```powershell
python -m src.experiments.create_experiment_summary `
  --input-path results/evaluation/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000.csv `
  --output-dir reports/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000 `
  --format all `
  --include-charts
```

The experiment summary can generate:

```text
experiment_summary.md
experiment_summary.json
agent_ranking.csv
agent_ranking.tex
deltas.csv
deltas.tex
quality_flags.csv
quality_flags.tex
charts/mean_profit_ci_by_opponent.png
charts/seed_stability_by_opponent.png
```

### RL algorithm comparison report

Create a dedicated report comparing only the adaptive RL algorithms:

```powershell
python -m src.experiments.create_algorithm_comparison `
  --input-path results/evaluation/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000.csv `
  --output-dir reports/algorithm_comparison_1000 `
  --format all `
  --include-charts
```

The algorithm comparison report can generate:

```text
algorithm_comparison.md
algorithm_comparison.json
algorithm_global_ranking.csv
algorithm_global_ranking.tex
algorithm_by_opponent.csv
algorithm_by_opponent.tex
algorithm_deltas.csv
algorithm_deltas.tex
charts/algorithm_mean_profit_by_opponent.png
charts/algorithm_seed_stability_by_opponent.png
charts/algorithm_global_mean_profit.png
```

In the algorithm comparison overview:

- `source_raw_games` is the number of games in the source evaluation CSV.
- `algorithm_summary_rows` is the number of filtered algorithm-level rows used
  in the dedicated algorithm comparison.

## Experiment Validation

Run validation checks for generalization results:

```powershell
python -m src.experiments.validate_checkpoint_evaluation `
  --input-path results/evaluation/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000.csv `
  --output-dir reports/generalization_mc_vs_q_learning_vs_sarsa_vs_double_q_learning_1000_validation `
  --validation-mode generalization `
  --format both
```

Validation checks produce `PASS`, `WARNING`, `FAIL` or `SKIPPED` statuses. They
are used to detect unstable results, weak baselines, extreme BB/100 values,
large seed variance and suspicious dominance of simple sanity baselines.

The current validation checks are still centred mostly on `adaptive_mc`, because
Monte Carlo was the original adaptive baseline. Therefore, a validation `FAIL`
can mean that the original Monte Carlo agent failed a sanity threshold even if
Q-learning, SARSA or Double Q-learning performed better.

## Metrics

The main evaluation metrics are:

- `mean_profit_bb`: average game-level profit in big blinds.
- `bb_per_100`: profit normalised to 100 hands.
- `win_rate`: fraction of games won.
- `bust_rate`: fraction of games where the evaluated agent lost its stack.
- `mean_profit_bb_std_across_seeds`: stability of mean profit across training
  seeds.
- `standard_error`: uncertainty of the sample mean.
- `ci_95_lower` and `ci_95_upper`: approximate 95% confidence interval.
- `classifier_accuracy`: correctness of classified non-unknown decisions.
- `classifier_coverage`: fraction of decisions where the classifier returned a
  known opponent type.
- `policy_switches`: number of times the adaptive player changed active policy.
- `delta_vs_monte_carlo`: difference in mean profit compared with the adaptive
  Monte Carlo baseline.
- `delta_vs_rule_based`: difference in mean profit compared with the rule-based
  baseline.
- `delta_vs_oracle`: difference in mean profit compared with the oracle adaptive
  baseline, if oracle rows are available.

`mean_profit_bb` is the main game-level profitability metric. `BB/100` is useful
for normalising by the number of hands, but it can become inflated when games end
after very few hands. For this reason, BB/100 should be interpreted together with
`mean_hands_played`, confidence intervals and seed stability.

## Current Experimental Status

The current pipeline supports the full flow from multi-seed training to
checkpoint evaluation, generalization evaluation, validation, experiment summary
reports and dedicated RL algorithm comparison reports.

The latest algorithm-level comparison at the 1000-episode checkpoint indicates
that temporal-difference methods outperform Monte Carlo under the same training
budget. Q-learning has the best average mean profit across evaluated opponents,
while Double Q-learning is the only adaptive algorithm with non-negative mean
profit in all evaluated matchups. SARSA is competitive and achieves the best
result against the `strong_calling` opponent.

The always-call and always-raise baselines are included as sanity checks. Strong
performance of a trivial baseline against a scripted opponent indicates that the
opponent may be too exploitable and should not be treated as sufficient evidence
of robust poker strategy.

## Current Limitations

- The poker environment is simplified and depends on the local `PyPokerEngine`
  implementation.
- The state space is discretised, so it does not capture every poker detail.
- The classifier needs several observed actions before it can identify an
  opponent type.
- The adaptive agent may lose chips before enough opponent information is
  available.
- Hand-strength evaluation is still simplified compared with full poker
  strategy concepts such as nuanced kicker comparison and all draw types.
- Opponent variants are scripted behavioural approximations, not human-level
  poker opponents.
- The always-raise baseline shows that some scripted opponents can be exploited
  by simple aggression.
- `strong_calling` remains vulnerable to trivial aggression in some evaluations.
- `aggressive_extreme` remains difficult even for the stronger TD-based agents.
- Double Q-learning improves robustness against aggressive variants but can be
  less stable against the calling opponent.
- Results in poker have high variance, so meaningful conclusions require
  multiple seeds and enough evaluation games.
- Performance against predefined opponents or variants does not imply
  performance against human players.

## Research Hypothesis

An adaptive reinforcement-learning agent that uses behavioural opponent
classification should achieve better long-term results than a non-adaptive
single-policy agent when opponents follow distinct and recognisable strategies.

The additional algorithm-comparison experiments test whether the choice of
learning algorithm itself affects robustness, generalization and stability under
the same state representation and reward definition.

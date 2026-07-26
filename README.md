# Poker Adaptive Opponent Model

Research project for testing dynamic strategy adaptation in heads-up Texas
Hold'em. The project combines tabular Monte Carlo reinforcement learning with
behavioural opponent classification.

The project is developed as part of a master's thesis:

> Dynamic strategy adaptation in imperfect-information games using
> reinforcement learning and behavioural classification

## Project Goal

The main goal is to evaluate whether an agent that identifies the opponent's
playing style and switches to a matching policy can perform better than
non-adaptive agents, specialist policies and rule-based baselines.

The experiments are run in heads-up poker, where one evaluated agent plays
against one fixed opponent. This keeps the environment focused on opponent
modelling instead of multi-player table dynamics.

The current experimental setup also includes sanity checks and generalization
tests. These are used to verify whether the observed results come from meaningful
adaptation or from exploiting overly simple scripted opponents.

## Implemented Scope

- Heads-up Texas Hold'em simulation based on a local `PyPokerEngine` dependency.
- First-visit tabular Monte Carlo control agent.
- Single-policy Monte Carlo agent trained against mixed opponents.
- Specialist Monte Carlo policies trained against one base opponent type.
- Adaptive player that uses behavioural classification to select a policy.
- Oracle adaptive baseline that knows the base opponent family from the start.
- Fixed-policy evaluation players for unknown, fish, aggressive and calling policies.
- Rule-based baseline player.
- Always-raise sanity baseline.
- Fixed base opponent strategies: fish, aggressive and calling.
- Parametrized unseen opponent variants for generalization tests:
  - `calling_weak`
  - `calling_medium`
  - `calling_strong`
  - `aggressive_light`
  - `aggressive_extreme`
- Opponent variants with street-based behaviour profiles and hand-strength-based
  raise probability adjustment.
- Discretised state representation with hand strength, poker context, pot
  information, stack-to-pot ratio and opponent type.
- Checkpoint training, checkpoint evaluation and readable HTML/Markdown reports.
- Direct head-to-head evaluation against handcrafted and out-of-distribution
  baselines.
- Generalization evaluation:
  trained on base opponents -> evaluated on unseen opponent variants.
- Experiment summary reports with rankings, baseline deltas, traffic-light
  quality statuses and automatic main findings.
- CSV, JSON, Markdown, HTML and LaTeX result outputs.
- Plots for checkpoint and experiment analysis, including confidence intervals
  and seed-stability charts.
- Automated validation checks for experiment quality and result stability.
- Automated tests for agents, state encoding, metrics, reports, validation,
  opponent variants and evaluation helpers.

## Project Structure

```text
src/
├── agents/           # Reinforcement learning agents
├── cards/            # Card and hand-strength utilities
├── evaluation/       # Metrics, evaluation, validation, reports and plots
├── experiments/      # CLI scripts for training, evaluation and reporting
├── features/         # State encoding and opponent statistics
├── opponent_model/   # Behavioural opponent classifier
├── players/          # PyPokerEngine player implementations
├── poker/            # Poker action and round-state helpers
├── training/         # Training schedules, checkpointing and metadata
└── config.py         # Main game, training and evaluation configuration

tests/                # Automated test suite
PyPokerEngine/        # Local poker engine dependency
results/              # Generated models and experiment results
reports/              # Generated readable reports
```

Shared enum-like values are kept in small `constants.py` files inside the
relevant packages, for example `src/poker/constants.py`,
`src/training/constants.py`, `src/experiments/constants.py`,
`src/evaluation/constants.py` and `src/players/constants.py`.

## Requirements

- Python 3.11
- pip
- Git

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
pip install -e ./PyPokerEngine
```

All commands below should be run from the project root.

## Configuration

Default parameters are defined in `src/config.py`:

- `GameConfig`: number of rounds, initial stack and blind size.
- `TrainingConfig`: episodes, alpha, epsilon settings, seed, checkpoints and
  model paths.
- `EvaluationConfig`: number of evaluation games and output CSV path.

Most training and evaluation scripts expose CLI flags such as `--episodes`,
`--seed`, `--epsilon-schedule`, `--alpha-mode`, `--checkpoint-episodes`,
`--games`, `--output-path`, `--checkpoint-directory`, `--training-run-dir` and
`--workers`.

## Training

Train the general single-policy model:

```bash
python -m src.experiments.run_single_policy_training
```

Train one specialist policy:

```bash
python -m src.experiments.run_specialist_training --opponent calling
```

Supported specialist opponents are `fish`, `aggressive` and `calling`.
Opponent variants are intentionally not used as specialist training opponents.
They are reserved for generalization evaluation.

Run a reproducible multi-seed training suite for the general policy and all
specialists:

```bash
python -m src.experiments.run_training_suite \
  --seeds 42 123 456 \
  --episodes 10000 \
  --checkpoint-episodes 1000 2500 5000 7500 10000 \
  --epsilon-schedule linear \
  --alpha-mode sqrt_visit
```

Useful options:

- `--models single_policy fish aggressive calling` chooses which policies to
  train.
- `--workers 4` runs multiple training jobs in parallel.
- `--rerun-existing` retrains jobs even if their final model already exists.
- `--no-progress` disables periodic progress logs.
- `--player-verbose` enables detailed player decisions.

## Evaluation

Run the default agent comparison:

```bash
python -m src.experiments.run_agent_comparison
```

Show aggregated comparison metrics:

```bash
python -m src.experiments.show_agent_comparison
```

Evaluate checkpoint models from a training suite:

```bash
python -m src.experiments.run_checkpoint_evaluation \
  --training-run-dir results/training_runs/<experiment_name> \
  --checkpoint-episodes 1000 2500 5000 7500 10000 \
  --games 200
```

Display checkpoint evaluation results:

```bash
python -m src.experiments.show_checkpoint_evaluation \
  --input-path results/training_runs/<experiment_name>/checkpoint_evaluation.csv
```

Run direct head-to-head evaluation against non-training baselines:

```bash
python -m src.experiments.run_head_to_head_evaluation \
  --training-run-dir results/training_runs/<experiment_name> \
  --checkpoint-episodes 10000 \
  --seeds 42 123 456 \
  --games 200 \
  --output-path results/evaluation/head_to_head_vs_baselines.csv
```

Run generalization evaluation on unseen opponent variants:

```bash
python -m src.experiments.run_generalization_evaluation \
  --training-run-dir results/training_runs/<experiment_name> \
  --checkpoint-episodes 10000 \
  --seeds 42 123 456 \
  --games 200 \
  --output-path results/evaluation/generalization_evaluation.csv
```

The generalization setup evaluates agents trained on the base opponents against
unseen behavioural variants. It does not train new variant-specific specialist
policies.

## Reports and Plots

Create a readable checkpoint report:

```bash
python -m src.experiments.create_checkpoint_report \
  --input-path results/training_runs/<experiment_name>/checkpoint_evaluation.csv \
  --output-dir reports/<experiment_name> \
  --format both
```

Create an experiment summary report with rankings, deltas, traffic-light
statuses, main findings, tables and charts:

```bash
python -m src.experiments.create_experiment_summary \
  --input-path results/evaluation/generalization_evaluation.csv \
  --output-dir reports/generalization_summary \
  --format all \
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

Compare selected Q-tables:

```bash
python -m src.experiments.compare_selected_q_tables \
  --training-run-dir results/training_runs/<experiment_name>
```

Create a readable Q-table comparison report:

```bash
python -m src.experiments.create_q_table_report \
  --input-path results/evaluation/q_table_comparison_selected.json \
  --output-dir reports/q_table_comparison \
  --format both
```

## Experiment Validation

Run validation checks for checkpoint evaluation results:

```bash
python -m src.experiments.validate_checkpoint_evaluation \
  --input-path results/training_runs/<experiment_name>/checkpoint_evaluation.csv \
  --output-dir reports/<experiment_name>_validation \
  --validation-mode checkpoint \
  --format both
```

Run validation checks for direct head-to-head results:

```bash
python -m src.experiments.validate_checkpoint_evaluation \
  --input-path results/evaluation/head_to_head_vs_baselines.csv \
  --output-dir reports/head_to_head_validation \
  --validation-mode head-to-head \
  --format both
```

Validation checks produce `OK`, `WARNING`, `FAIL` or `SKIPPED` statuses. They
are used to detect unstable results, weak baselines, extreme BB/100 values,
large seed variance and suspicious dominance of simple sanity baselines.

## Metrics

The main evaluation metrics are:

- `total_profit_bb`: total profit or loss expressed in big blinds.
- `mean_profit_bb`: average game-level profit in big blinds.
- `bb_per_100`: profit normalised to 100 hands.
- `win_rate`: fraction of games won.
- `bust_rate`: fraction of games where the evaluated agent lost its stack.
- `standard_error`: uncertainty of the sample mean.
- `ci_95_lower` and `ci_95_upper`: approximate 95% confidence interval.
- `std_across_seeds`: stability of results across independently trained seeds.
- `classifier_accuracy`: correctness of classified non-unknown decisions.
- `classifier_coverage`: fraction of decisions where the classifier returned a
  known opponent type.
- `policy_switches`: number of times the adaptive player changed active policy.
- `delta_vs_rule_based`: difference in mean profit compared with the rule-based
  baseline.
- `delta_vs_oracle`: difference in mean profit compared with the oracle adaptive
  baseline.

`mean_profit_bb` is the main game-level profitability metric. `BB/100` is useful
for normalising by the number of hands, but it can become inflated when games end
after very few hands. For this reason, BB/100 should be interpreted together with
`mean_hands_played`, confidence intervals and seed stability.

## Current Experimental Status

The current pipeline supports the full flow from multi-seed training to
checkpoint evaluation, head-to-head evaluation, generalization evaluation,
validation and thesis-ready reports.

Current experiments suggest that the adaptive agent can perform well against
base opponents and some opponent families known from training. Direct
head-to-head and generalization tests are used to check whether this behaviour
extends beyond the exact scripted opponents used during training.

The always-raise baseline is included as a sanity check. Strong performance of
this baseline against some scripted opponents indicates that those opponents may
be too exploitable and should not be treated as sufficient evidence of robust
poker strategy.

## Testing and Quality

Run the project test suite:

```bash
PYTHONPATH=.:PyPokerEngine pytest tests -q
```

Run a selected test file:

```bash
PYTHONPATH=.:PyPokerEngine pytest tests/test_state_encoder.py -q
```

Running plain `pytest` from the repository root may also collect tests from the
vendored `PyPokerEngine` dependency. For project-level validation, use the
`tests/` directory explicitly.

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
- Results in poker have high variance, so meaningful conclusions require
  multiple seeds and enough evaluation games.
- Performance against predefined opponents or variants does not imply
  performance against human players.

## Research Hypothesis

An adaptive reinforcement learning agent that uses behavioural opponent
classification should achieve better long-term results than a non-adaptive
single-policy agent when opponents follow distinct and recognisable strategies.

The additional generalization experiments test whether this advantage extends to
unseen behavioural variants from the same broad opponent families.

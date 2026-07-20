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
non-adaptive agents and rule-based baselines.

The experiments are run in heads-up poker, where one evaluated agent plays
against one fixed opponent. This keeps the environment focused on opponent
modelling instead of multi-player table dynamics.

## Implemented Scope

- Heads-up Texas Hold'em simulation based on a local `PyPokerEngine` dependency.
- First-visit tabular Monte Carlo control agent.
- Single-policy Monte Carlo agent trained against mixed opponents.
- Specialist Monte Carlo policies trained against one opponent type.
- Adaptive player that uses behavioural classification to select a policy.
- Oracle adaptive baseline that knows the opponent type from the start.
- Rule-based baseline player.
- Fixed opponent strategies: fish, aggressive and calling.
- Discretised state representation with hand strength, poker context, pot
  information, stack-to-pot ratio and opponent type.
- Checkpoint training, checkpoint evaluation and readable HTML/Markdown reports.
- CSV/JSON result outputs and plots for analysis.
- Automated tests for agents, state encoding, metrics, reports and training
  helpers.

## Project Structure

```text
src/
├── agents/           # Reinforcement learning agents
├── cards/            # Card and hand-strength utilities
├── evaluation/       # Metrics, checkpoint evaluation, reports and plots
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

Most training scripts also expose CLI flags such as `--episodes`, `--seed`,
`--epsilon-schedule`, `--alpha-mode`, `--checkpoint-episodes`,
`--output-path` and `--checkpoint-directory`.

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

## Reports and Plots

Create a readable checkpoint report:

```bash
python -m src.experiments.create_checkpoint_report \
  --input-path results/training_runs/<experiment_name>/checkpoint_evaluation.csv \
  --output-dir reports/<experiment_name> \
  --format both
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

## Metrics

The main evaluation metrics are:

- `total_profit_bb`: total profit or loss expressed in big blinds.
- `mean_profit_bb`: average game-level profit in big blinds.
- `bb_per_100`: profit normalised to 100 hands.
- `win_rate`: fraction of games won.
- `bust_rate`: fraction of games where the evaluated agent lost its stack.
- `standard_error`: uncertainty of the sample mean.
- `ci_95_lower` and `ci_95_upper`: approximate 95% confidence interval.
- `classifier_accuracy`: correctness of classified non-unknown decisions.
- `classifier_coverage`: fraction of decisions where the classifier returned a
  known opponent type.
- `policy_switches`: number of times the adaptive player changed active policy.

`BB/100` is usually the most useful profitability metric because games can end
after different numbers of hands.

## Testing and Quality

Run the full test suite:

```bash
pytest
```

Run a selected test file:

```bash
pytest tests/test_state_encoder.py
```

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
- Results in poker have high variance, so meaningful conclusions require
  multiple seeds and enough evaluation games.
- Performance against predefined opponents does not imply performance against
  human players.

## Research Hypothesis

An adaptive reinforcement learning agent that uses behavioural opponent
classification should achieve better long-term results than a non-adaptive
single-policy agent when opponents follow distinct and recognisable strategies.

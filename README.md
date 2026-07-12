# Adaptive Opponent Modelling in Heads-Up Poker

A research project investigating dynamic strategy adaptation in heads-up Texas Hold'em using reinforcement learning and behavioural opponent classification.

The project is developed as part of a master's thesis:

> **Dynamic Strategy Adaptation in Imperfect-Information Games Using Reinforcement Learning and Behavioral Classification**

## Project Overview

The goal of the project is to evaluate whether a poker agent that identifies its opponent's playing style and adapts its policy can outperform non-adaptive agents and fixed rule-based strategies.

The experiments are conducted in heads-up Texas Hold'em, where two players compete against each other. This setup allows the agent to focus on learning and adapting to the behaviour of a single opponent.

## Implemented Features

The current implementation includes:

* heads-up Texas Hold'em simulation based on `PyPokerEngine`,
* Monte Carlo reinforcement learning agents,
* an adaptive agent using the estimated opponent type as part of the state,
* a single-policy reinforcement learning agent used as a non-adaptive baseline,
* predefined opponent strategies:

  * aggressive player,
  * calling player,
  * fish player,
* behavioral opponent classification based on observed actions,
* discretised state representation,
* configurable training and evaluation experiments,
* saving and loading trained models,
* game-level results exported to CSV,
* aggregated agent comparison,
* statistical evaluation using confidence intervals,
* automated tests for agents, state encoding and evaluation metrics.

## State Representation

The reinforcement learning agents use a discretised state representation containing information such as:

* pot size,
* amount required to call,
* current community-card stage,
* player stack size,
* estimated opponent type.

The adaptive agent includes the classified opponent type in its state, allowing it to learn different actions against different playing styles.

## Project Structure

```text
src/
├── agents/          # Reinforcement learning agents and opponent strategies
├── cards/           # Card and hand-strength evaluation
├── config/          # Game, training and evaluation configuration
├── evaluation/      # Metrics and result aggregation
└── experiments/     # Training and evaluation scripts

results/models/      # Saved trained models
tests/               # Automated tests
results/raw/         # Generated CSV experiment results
PyPokerEngine/       # Local poker engine dependency
```

## Requirements

* Python 3.11
* pip
* Git

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd poker-adaptive-opponent-model
```

If `PyPokerEngine` is included as a Git submodule, initialise it with:

```bash
git submodule update --init --recursive
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install the local `PyPokerEngine` package:

```bash
pip install -e ./PyPokerEngine
```

All commands should be executed from the project root directory.

## Running Training Experiments

### Train the adaptive agent

```bash
python -m src.experiments.train_adaptive_agent
```

The adaptive agent is trained against multiple opponent types. The estimated opponent category is included in its state representation.

### Train the single-policy agent

```bash
python -m src.experiments.train_single_policy_agent
```

The single-policy agent is trained against multiple opponents but does not explicitly include the opponent type in its state.

Training parameters such as the number of episodes, learning rate, exploration rate and exploration decay are defined in the project configuration.

## Running Evaluation Experiments

Run the configured evaluation experiments:

```bash
python -m src.experiments.evaluate_agents
```

The evaluation compares trained agents against the available opponent strategies and saves game-level results to CSV files.

To display the aggregated agent comparison, run:

```bash
python -m src.experiments.show_agent_comparison
```

The generated comparison includes profitability, variance, win rate, bust rate and the number of hands played.

## Running Tests

Run the complete test suite:

```bash
pytest
```

Run tests with more detailed output:

```bash
pytest -v
```

Run a selected test file:

```bash
pytest tests/path/to/test_file.py
```

## Evaluation Metrics

### Total profit in big blinds

The total profit or loss accumulated across all evaluated games, expressed in big blinds.

```text
total_profit_bb = sum(profit_bb)
```

A positive value indicates an overall profit, while a negative value indicates an overall loss.

### Mean profit in big blinds

The average profit obtained in one evaluated game.

```text
mean_profit_bb = total_profit_bb / number_of_games
```

This metric is useful for comparing average game-level performance.

### Standard deviation of profit

The standard deviation measures the variability of the agent's game results.

A high standard deviation indicates that the results are unstable or highly dependent on individual games. A lower value indicates more consistent performance.

### Big blinds per 100 hands

`BB/100` measures the number of big blinds won or lost per 100 played hands.

```text
bb_per_100 = total_profit_bb / total_hands_played * 100
```

This is the primary profitability metric because it normalises the result by the number of hands played.

A positive value indicates a profitable strategy. A negative value indicates a losing strategy.

### Win rate

The proportion of evaluated games won by the agent.

```text
win_rate = won_games / total_games
```

A game is considered won when the agent finishes with more chips than its opponent.

Win rate should not be interpreted independently from profit because an agent may win many small games while losing fewer but significantly larger games.

### Bust rate

The proportion of games in which the agent loses its entire stack.

```text
bust_rate = busted_games / total_games
```

A high bust rate may indicate excessive risk-taking or an inability to adapt before losing the available stack.

### Mean number of hands played

The average number of hands completed during one game.

This metric helps determine whether an agent tends to win or lose quickly and whether the game configuration gives it enough time to identify the opponent's strategy.

### Minimum and maximum hands played

These metrics show the shortest and longest evaluated games.

They help identify unusually short games, early bankruptcies and differences in game duration between opponent types.

### 95% confidence interval

The 95% confidence interval estimates the range in which the true mean performance is expected to lie.

It is calculated from the sample mean, standard deviation and standard error.

```text
standard_error = standard_deviation / sqrt(number_of_games)
```

A narrow confidence interval indicates a more precise estimate. A wide confidence interval suggests high variance or an insufficient number of evaluation games.

## Experimental Comparison

The project evaluates the following main approaches:

1. **Adaptive reinforcement learning agent**

   Uses the estimated opponent type as part of its state and may learn different strategies against different opponent categories.

2. **Single-policy reinforcement learning agent**

   Uses one shared policy against all opponent types without explicit opponent classification.

3. **Rule-based baseline**

   Uses predefined poker rules and does not learn from experience.

The agents are evaluated separately against aggressive, calling and fish opponents.

## Current Limitations

* The project uses a simplified poker environment.
* The state space is discretised and does not represent every detail of the game.
* Opponent classification requires several observed actions before it becomes reliable.
* An agent may lose a significant part of its stack before identifying an opponent.
* Hand-strength evaluation does not fully represent all strategic concepts, such as every draw type or advanced kicker comparison.
* The current implementation does not use deep reinforcement learning.
* Poker results have high variance and require a sufficiently large evaluation sample.
* Performance against predefined opponents does not guarantee equivalent performance against human players.

## Research Objective

The primary research objective is to determine whether behavioral opponent classification improves reinforcement learning performance in an imperfect-information game.

The main hypothesis is that an agent capable of identifying an opponent's playing style and adapting its policy should achieve better long-term results than an agent using one fixed policy against every opponent.

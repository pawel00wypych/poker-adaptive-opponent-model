# Poker Adaptive Opponent Model

Research code for a master's thesis on adaptive strategy selection in heads-up
Texas Hold'em. The project combines behavioural opponent classification with
tabular reinforcement learning to test whether switching between specialist
policies improves on using one fixed general policy.

## Sources of truth

This README is a short entry point. The authoritative experiment specification
is [final_experiment_guidelines.md](final_experiment_guidelines.md), including
the research questions, hypotheses, frozen configuration, comparison matrices,
aggregation rules, reports, and limitations.

The executable configuration is defined in:

- `src/experiment_protocol.py` - versioned `verification`, `final`, and
  `extended` presets, evaluation budgets, opponents, and provenance;
- `src/config.py` - shared game and training configuration;
- `src/experiments/run_thesis_pipeline.py` - the end-to-end CLI and stage graph.

Run-specific conclusions belong in generated reports, not in this README.

## Research scope

The project compares four tabular algorithms under the same state, action,
reward, training-budget, and seed protocol:

```text
Monte Carlo
Q-learning
SARSA
Double Q-learning
```

For every algorithm, training produces one general policy and specialists for
`tight`, `aggressive`, and `calling` opponents. Evaluations compare:

| Group | Role |
| --- | --- |
| Adaptive | Classifies the opponent and switches between specialists |
| General policy | Uses one fixed policy throughout the game |
| Family-informed Oracle | Uses the specialist assigned to the known family; it is diagnostic, not optimal |
| Baselines | `rule_based`, `always_call`, and `always_raise` |

Generalization is limited to `tight_extreme`, `aggressive_extreme`, and
`calling_extreme`. Stress tests, cross-play, learning curves, and baseline
head-to-head matches are diagnostic analyses.

The primary metric is `mean_profit_bb`. Comparisons of learned agents use paired
differences calculated per shared training seed; evaluation games are not
treated as independent training replicates.

## Requirements and installation

Python 3.11 is required and enforced at import time. Dependencies include a
pinned Git revision of PyPokerEngine.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alternatively:

```powershell
conda env create -f environment.yml
conda activate poker_env_311
```

Run commands from the repository root.

## End-to-end experiments

Start with the complete short rehearsal:

```powershell
python -m src.experiments.run_thesis_pipeline --config verification
```

Run the frozen thesis experiment:

```powershell
python -m src.experiments.run_thesis_pipeline --config final --workers 4
```

After a completed final run, reuse its models for the extended evaluation:

```powershell
python -m src.experiments.run_thesis_pipeline --config extended `
  --final-pipeline-dir results\pipelines\final `
  --workers 4
```

Use the same command with `--resume` after an interruption. Add `--dry-run` to
inspect the stage plan without executing it. `--workers` controls only runtime
parallelism and does not change the scientific protocol.

| Preset | Training | Evaluation |
| --- | --- | --- |
| `verification` | 500 episodes, 3 seeds | 200 games per matchup |
| `final` | 10,000 episodes, 5 seeds | 500 games per matchup |
| `extended` | Reuses final models | 1,000 games per matchup |

The pipeline is fail-fast, logs to the console and `pipeline.log`, supports
validated resume, and generates evaluations, validations, plots, and final
reports automatically. Outputs are stored under
`results\pipelines\<preset>\`, including:

```text
pipeline.log
pipeline_manifest.json
pipeline_summary.json
models/        # verification and final
evaluations/
reports/
```

Advanced training, evaluation, reporting, and validation CLIs are available
under `src/experiments/`; use each module's `--help`. Thesis runs should use the
end-to-end pipeline so that stages, provenance, and reports remain consistent.

## Reproducibility rules

- Do not combine results from different presets.
- Primary benchmarks load one `final.pkl` per algorithm and seed.
- Checkpoints are used only for learning-curve diagnostics.
- Uncertainty for learned agents is assessed across training seeds.
- Models and evaluation summaries record protocol hashes and source provenance.

The complete methodology and interpretation rules are defined in
[final_experiment_guidelines.md](final_experiment_guidelines.md).

## Repository structure

```text
src/
  agents/, rl/, training/       RL implementations and model training
  classifier/, features/        Opponent profiling and state encoding
  players/, poker/, cards/      Players and poker utilities
  evaluation/                   Runners, metrics, validation, and reports
  experiments/                  Command-line entry points and pipeline
  config.py                     Shared game and training configuration
  experiment_protocol.py        Frozen experiment presets
tests/                          Automated test suite
results/                        Generated experiment artefacts
final_experiment_guidelines.md  Authoritative experiment specification
```

## Development checks

```powershell
python -m pytest -q
python -m ruff check .
```

## Limitations

The conclusions apply only to the simplified heads-up environment, tabular
representation, stationary scripted opponents, and specified held-out variants.
They do not establish general poker ability, optimal play, or performance
against humans. Five training seeds are a computational compromise, so effects
and uncertainty must be reported alongside mean rankings.

### No postflop positional alternation

The pinned PyPokerEngine implementation makes the small blind act first on every
street. Real heads-up play reverses the postflop order, so positional conclusions
are valid only within this environment. Characterisation tests in
`tests\poker\test_round_state_utils.py` pin this behaviour.

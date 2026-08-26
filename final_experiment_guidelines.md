# Final Experiment Set

# 1. Context, Groups, and Notation

RL algorithms being compared:

```text
Monte Carlo
Q-learning
SARSA
Double Q-learning
```

The following are trained for each algorithm:

```text
general_policy
specialist_tight
specialist_aggressive
specialist_calling
```

## 1.1. Agent and Opponent Groups

### Agent Groups

| Symbol | Group | Agents |
| --- | --- | --- |
| `A` | Adaptive | `adaptive_mc`, `adaptive_q_learning`, `adaptive_sarsa`, `adaptive_double_q_learning` |
| `O` | Family-informed Oracle | `oracle_mc`, `oracle_q_learning`, `oracle_sarsa`, `oracle_double_q_learning` |
| `G` | General policy | `policy_general_mc`, `policy_general_q_learning`, `policy_general_sarsa`, `policy_general_double_q_learning` |
| `B` | Baseline | `rule_based`, `always_call`, `always_raise` |

### Opponent Groups

| Symbol | Group | Opponents |
| --- | --- | --- |
| `T` | Training opponents | `tight`, `aggressive`, `calling` |
| `H` | Held-out variants | `tight_extreme`, `aggressive_extreme`, `calling_extreme` |
| `S` | Stress-test opponents | `always_call`, `always_raise`, `rule_based` |

### Matrix Notation

The notation:

```text
A × T
```

means that every agent in group `A` plays against every opponent in group `T`.

For example:

```text
|A| = 4
|T| = 3

A × T = 4 × 3 = 12 matchups per model_seed
```

Unless stated otherwise, a matchup is directional:

```text
agent_A vs agent_B != agent_B vs agent_A
```

The Oracle is not an ordinary primary agent. It is a diagnostic `family-informed Oracle` benchmark: it knows the opponent's true family and uses the specialist policy assigned to that family from the outset. It does not empirically search for the best policy and is not a mathematically optimal agent; consequently, in some matchups it may be beaten by an adaptive agent or a general policy.

Fixed specialist policies are not evaluated as a separate primary agent group. For an opponent belonging to a particular family, the family-informed Oracle uses from the outset exactly the specialist policy assigned to that family. A separate run of the corresponding specialist against the same opponent would therefore duplicate the comparison performed by the Oracle.

Comparisons of specialists with opponents belonging to other families may be conducted as an optional ablation analysis. They make it possible to determine whether the specialist assigned to the nominal family is empirically the best fixed policy. However, this analysis is not required to evaluate the primary hypotheses H1–H3.

The absence of a full specialist–opponent matrix means that the Oracle must be interpreted solely as a benchmark using the specialist assigned to the known family, rather than as the best possible fixed policy.

The Oracle is meaningful only for algorithms that have specialist policies:

```text
specialist_tight
specialist_aggressive
specialist_calling
```

The Oracle is not meaningful for `policy_general_*`, `rule_based`, `always_call`, and `always_raise`, because these agents do not switch between specialists.


# 2. Research Questions and Hypotheses

## 2.1. Primary Objective

The primary objective of the experiments is to assess whether an adaptive agent that uses opponent classification and switches between specialist policies achieves better results than an agent that uses a single general policy throughout the game.

Supplementary objectives include:

```text
comparison of reinforcement learning algorithms
assessment of generalization to unseen opponent variants
assessment of opponent classifier quality
estimation of the cost of incorrect or delayed adaptation
assessment of result stability across seeds
testing robustness against stress-test strategies and sanity baselines
```

## 2.2. Primary Metric

The primary metric used to evaluate the hypotheses is:

```text
mean_profit_bb
```

The metric is first calculated separately for each training seed, agent, and opponent. Agent comparisons use paired differences between results obtained for the same seeds.

The metrics:

```text
bb_per_100
win_rate
bust_rate
```

are auxiliary metrics used to interpret the results, but they do not replace the primary metric when evaluating the hypotheses.

The primary scope of inference covers:

```text
training-opponent evaluation
generalization evaluation
```

Stress-test, cross-play, and baseline head-to-head are robustness and diagnostic analyses. They are not included in a single global result used to evaluate the primary hypotheses.

## 2.3. Research Questions

### RQ1 — Benefit of Adaptation against Training Opponents

Does adaptive policy selection based on opponent classification improve performance relative to the corresponding fixed general policy against the base training archetypes?

Comparison:

```text
adaptive_X vs policy_general_X
```

Opponent scope:

```text
tight
aggressive
calling
```

### RQ2 — Generalization of Adaptation

Does the benefit of adaptive policy selection persist against unseen variants of opponent families?

Comparison:

```text
adaptive_X vs policy_general_X
```

Opponent scope:

```text
tight_extreme
aggressive_extreme
calling_extreme
```

Conclusions from this question apply only to the specified variants. They do not automatically imply generalization to all possible poker opponents.

### RQ3 — Comparison of RL Algorithms

Which of the analyzed reinforcement learning algorithms achieves the best and most stable performance as an adaptive agent?

Algorithms being compared:

```text
Monte Carlo
Q-learning
SARSA
Double Q-learning
```

The comparison includes separate results per opponent and a macro-average in which every opponent receives equal weight.

The primary algorithm ranking includes exactly six opponents:

```text
tight
aggressive
calling
tight_extreme
aggressive_extreme
calling_extreme
```

Stress-test, cross-play, and baseline head-to-head are not included in the primary algorithm ranking.

For each algorithm and seed, the following macro-average is calculated:

```text
global_score_seed =
    mean(
        mean_profit_bb against tight,
        mean_profit_bb against aggressive,
        mean_profit_bb against calling,
        mean_profit_bb against tight_extreme,
        mean_profit_bb against aggressive_extreme,
        mean_profit_bb against calling_extreme
    )
```

Each opponent receives an equal weight of `1/6`. The result is not weighted by the number of hands, game length, or number of rows in the file.

The `global_score_seed` values are then aggregated across five seeds. The report should include:

```text
mean global_score
standard deviation across seeds
95% confidence interval
global_score for each seed
```

Algorithms are ranked in descending order by mean `global_score`. Other metrics, such as `bb_per_100`, `win_rate`, `bust_rate`, the number of matchups won, and average rank, are auxiliary and do not change the primary ranking order.

A global ranking may be produced only if every algorithm has a complete set of results for all six opponents and five seeds.

RQ3 is comparative and exploratory. It is not assumed in advance that all TD algorithms must outperform Monte Carlo. Any directional hypothesis for a particular algorithm should follow from the theoretical part of the thesis, not from the results obtained.

### RQ4 — Quality and Cost of Opponent Profiling

How accurately and how quickly does the classifier recognize the opponent's family, and what performance cost results from imperfect classification and delayed policy switching?

Comparison analyzed:

```text
family-informed Oracle_X vs adaptive_X
```

The Oracle agent used in the project knows the opponent's true family and selects the specialist policy assigned to it from the outset. It is therefore a family-informed Oracle, not a mathematically optimal agent.

The Oracle–adaptive gap is defined as:

```text
oracle_gap_bb =
    oracle_mean_profit_bb
    - adaptive_mean_profit_bb
```

A positive value means that the adaptive agent performs worse than its corresponding Oracle agent.

### RQ5 — Robustness and Stability

Do the most important conclusions remain stable across seeds, and do the learned policies maintain reasonable performance against sanity and stress-test strategies and other learned agents?

RQ5 is diagnostic in nature. It is used to identify the limitations of the solution, not to extend the primary conclusions to all possible poker strategies.

## 2.4. Research Hypotheses

### H1 — Benefit of Adaptation against Training Opponents

The adaptive agent achieves a higher `mean_profit_bb` than the corresponding fixed general policy against the base training archetypes.

For each seed, a macro-average across algorithms and opponents is calculated:

```text
adaptation_gain_training_seed =
    mean(
        mean_profit_bb(adaptive_X)
        - mean_profit_bb(policy_general_X)
    )
```

Statistical hypotheses:

```text
H1_0: mean(adaptation_gain_training_seed) <= 0
H1_A: mean(adaptation_gain_training_seed) > 0
```

### H2 — Benefit of Adaptation against Unseen Variants

The adaptive agent achieves a higher `mean_profit_bb` than the corresponding fixed general policy against unseen variants of opponent families.

For each seed, a macro-average across algorithms and variants is calculated:

```text
adaptation_gain_generalization_seed =
    mean(
        mean_profit_bb(adaptive_X)
        - mean_profit_bb(policy_general_X)
    )
```

Statistical hypotheses:

```text
H2_0: mean(adaptation_gain_generalization_seed) <= 0
H2_A: mean(adaptation_gain_generalization_seed) > 0
```

### H3 — Cost of Imperfect Opponent Profiling

An agent that knows the opponent's true family from the outset achieves a higher `mean_profit_bb` than the corresponding adaptive agent.

For each seed, the following gap is calculated:

```text
oracle_gap_seed =
    mean(
        mean_profit_bb(family-informed Oracle_X)
        - mean_profit_bb(adaptive_X)
    )
```

Statistical hypotheses:

```text
H3_0: mean(oracle_gap_seed) <= 0
H3_A: mean(oracle_gap_seed) > 0
```

H3 concerns the combined cost of the adaptation mechanism, including:

```text
time spent in the unknown state
incorrect opponent classification
delayed specialist selection
switching between policies
```

The entire gap should not be interpreted as the effect of a single classifier error alone.

## 2.5. Hypothesis Evaluation Rule

The hypotheses are evaluated on the basis of the mean paired difference between agents and a 95% confidence interval calculated across training seeds.

Interpretation:

```text
expected direction and confidence interval excluding zero
    -> the results support the hypothesis

expected direction, but confidence interval includes zero
    -> inconclusive result

absence of the expected direction
    -> the results do not support the hypothesis
```

Because five seeds are used, the report should always also present the results for each seed. A hypothesis should not be described as “proven.” Recommended formulations are:

```text
the results support the hypothesis
the results do not support the hypothesis
the results are inconclusive
```

# 3. Frozen Configuration

The final thesis results use the versioned protocol:

```text
protocol_id: thesis-final-v2
preset_name: final
```

The configuration is established before the final results are analyzed. Changing a
scientific parameter creates a configuration marked as `custom`, with its own
`experiment_config_hash`. Execution parameters, such as the number of workers,
file paths, and logging frequency, are not part of the hash.

## 3.1. Game Environment

| Parameter | Value |
| --- | ---: |
| Number of players | 2 |
| Maximum number of hands per game | 100 |
| Starting stack | 200 |
| Small blind | 5 |
| Big blind | 10 |
| Starting stack in BB | 20 BB |
| Early termination | player bankruptcy |

## 3.2. Training

| Parameter | Value |
| --- | --- |
| Training episodes per model | 10 000 |
| Seeds | 42, 123, 456, 789, 2026 |
| Alpha | 0.1 |
| Alpha mode | sqrt_visit (`alpha = 1 / sqrt(N(s,a))`) |
| Gamma | 1.0 |
| Epsilon start | 0.5 |
| Epsilon minimum | 0.05 |
| Epsilon schedule | linear |
| Checkpoints | 1000, 2500, 5000, 7500, 10000 |

One training episode means one PyPokerEngine game containing no more than
100 hands. The general policy is trained cyclically against:

```text
tight -> aggressive -> calling -> tight -> ...
```

A specialist policy is trained exclusively against its assigned family.

## 3.3. State Representation

```text
state_version: state_v2

street
hand_strength_bin
pair_strength_bin
pot_bucket
pot_odds_bin
spr_bin
is_small_blind
```

## 3.4. Action Space

```text
action_version: action_v1

0 -> fold
1 -> call/check
2 -> minimum raise
```

If raising is unavailable or has an invalid range, the choice is converted
to `call/check`. An attempt to fold when a free check is available is converted
to a check.

## 3.5. Reward

```text
reward_version: reward_bb_v1

reward_bb =
    (stack_after_hand - stack_before_hand)
    / big_blind
```

The reward is assigned after the hand ends. No intermediate reward shaping
is applied.

## 3.6. Execution Presets

| Preset | Episodes | Seeds | Games per matchup | Learning curve | Baseline replicates |
| --- | ---: | --- | ---: | ---: | ---: |
| `verification` | 500 | 42, 123, 456 | 200 | 200 | 3 |
| `final` | 10 000 | 42, 123, 456, 789, 2026 | 500 | 200 | 5 |
| `extended` | 10 000 | 42, 123, 456, 789, 2026 | 1000 | 200 | 5 |

The `extended` preset uses the same training as `final`; it differs only in
the primary evaluation budget.

## 3.7. Evaluation Seed Namespaces

`eval_seed_namespace` is an identifier of a randomness space, not a training
seed or number of games:

| Evaluation | Namespace |
| --- | ---: |
| Training-opponent | 1 |
| Learning-curve | 1 |
| Head-to-head and baseline-only | 2 |
| Generalization | 3 |
| Stress-test | 4 |
| Cross-play | 5 |

Training-opponent and learning-curve share namespace `1` so that final models
and checkpoints can receive corresponding random sequences. The other
evaluation types have separate namespaces.

Values `1–5` are conventional, human-readable identifiers. The seed of an
individual game is derived deterministically from the namespace, `model_seed`,
`model_episode`, and `matchup_game_index`.

## 3.8. Artifact Provenance

Each new model, training manifest, and evaluation `*.summary.json` records:

```text
protocol_id
preset_name
experiment_config_hash
training_config_hash
experiment_config
source_revision
source_dirty
```

`experiment_config.json` stores the full protocol snapshot for the training suite.
`training_config_hash` is the same for the `final` and `extended` presets, allowing
final models to be used in extended evaluation without
retraining.

# 4. Common Methodology

## 4.1. Seeds and Replicates

The seed set and budgets are specified by the frozen protocol in Section 3. Five training seeds represent a compromise between estimating variability and computational cost. Every algorithm uses the same seed set and the same training budget.

Results from the `verification`, `final`, and `extended` presets are not combined.

### Evaluation Seeds

The training seed and evaluation seed serve different purposes:

```text
model_seed      -> determines the course of model training
evaluation_seed -> determines the cards and other randomness of an individual game
```

A deterministic `evaluation_seed` is generated for each `model_seed`, opponent, and game index.

Corresponding comparisons use the same evaluation seeds. For the same training seed and game index, the agents being compared receive corresponding random sequences. This enables paired comparisons and reduces the effect of hand randomness on differences between agents.

Agent and opponent names are not components of the game seed. Common random
numbers are therefore preserved for corresponding matchups within a given
namespace.

### Evaluation Budget and Number of Games

The `games_per_matchup` parameter specifies the number of complete games played for a single agent–opponent combination.

One evaluation game is a complete heads-up match configured according to Section 3.1. The parameter does not mean the number of individual hands.

For evaluations using learned models, the budget is applied separately to each `model_seed`:

```text
number of games =
    games_per_matchup
    × number of model_seed
    × number of agent–opponent matchups
```

For baseline-only experiments:

```text
number of games =
    games_per_matchup
    × evaluation_replicates
    × number of baseline–baseline matchups
```

Preset values are specified exclusively in Section 3.6. The `extended` profile may be run only for the entire planned scope; the number of games must not be selectively increased on the basis of the results obtained.

The actual `games_per_matchup` value must be recorded in:

```text
the launch command
the summary.json file
report metadata
the experiment run name or identifier
```

### Baseline-only Replicates

The `rule_based`, `always_call`, and `always_raise` agents do not have training seeds.

Comparisons involving only such baselines use an `evaluation_replicate_id`, which denotes an independent simulation replicate. It must not be interpreted as a training seed or combined with variability resulting from model training.
If a baseline is evaluated as part of an evaluation containing learned models, it may be run separately in each `model_seed` block to use the same `evaluation_seed` values as the models being compared.

In that case, `model_seed` is not the baseline's training seed. It identifies only the comparison block and the shared evaluation-randomness set.

This makes it possible to calculate paired differences, for example:

```text
adaptive_X - rule_based
policy_general_X - rule_based
```

However, the baseline is not treated as a model trained with five seeds and is not included in the analysis of training variance.
## 4.2. Result Aggregation

Results are aggregated in three stages:

```text
1. Record the result of each individual game.
2. Aggregate games separately for each model_seed.
3. Aggregate the means obtained for individual training seeds.
```

Every final report concerning learned models should include at least:

```text
per-game results
per-model_seed aggregates
results for all five seeds
mean across seeds
standard deviation across seeds
standard error across seeds
95% Student-t confidence interval
minimum, maximum, and range across seeds
```

Baseline-only reports use a different aggregation scheme:

```text
per-game results
per-evaluation_replicate_id aggregates
mean across evaluation replicates
standard deviation across evaluation replicates
```

Baseline-only reports do not contain `model_seed`, because none of the compared agents is trained.

The unit used to assess uncertainty arising from training is the training seed, not an individual game. A large number of evaluation games reduces simulation noise but does not replace independent training runs.

Primary agent comparisons should use paired differences calculated separately for each shared seed:

```text
delta_seed = metric_agent_A_seed - metric_agent_B_seed
```

Only the `delta_seed` values are then aggregated across seeds. The final report should give the mean difference and its 95% confidence interval.

### Incomplete Runs

If training for one of the planned seeds ends with a technical error, training must be repeated with the same seed.

The following are prohibited:

```text
removing a seed on the basis of its achieved result
replacing a weak seed with another seed
selecting the best seed for the final report
comparing algorithms on different seed sets
treating baseline-only replicates as training seeds
```

The final dataset should contain a complete and identical set of five seeds for every learned algorithm being compared. If a complete set cannot be obtained, the missing data must be explicitly described and the comparison performed only on seeds shared by both agents.

## 4.3. Model Selection

Every primary benchmark (`training-opponent`, `generalization`, `stress-test`, `agent cross-play`, and learned-agent `head-to-head`) uses exactly one model per algorithm and seed: the `final.pkl` file. The `training_episode` field is read from `final.json` and describes completed training. It is not a checkpoint-selection parameter.

```text
for seed in model_seeds:
    load final.pkl for each evaluated policy
    verify completed_episodes and seed in final.json
    evaluate this final model once
```

Primary evaluations do not accept a list of `checkpoint_episodes`, do not select the “latest checkpoint,” and may not combine `final` models and intermediate checkpoints in one file.

Checkpoints are used exclusively in a separate diagnostic learning-analysis workflow:

```text
run_learning_curve_evaluation.py
learning_curve_evaluation.csv
learning_curve_report.md / learning_curve_report.html
```

In this workflow, the `model_source` field has the value `checkpoint`, and the learning axis is recorded as `checkpoint_episode`. These results are not used for the final algorithm ranking, hypothesis validation, or reporting final-model quality. This prevents selection of the best point on the curve based on evaluation data.

---

# 5. Experiments

## 5.1. Training-opponent evaluation

### Objective and Research Link

Performance against the base training archetypes.

```text
Link: RQ1, RQ3, RQ4, H1, H3
```

### Comparison Matrix

| Priority | Agent group | Opponent group | Number of matchups per seed | Objective |
| --- | --- | --- | ---: | --- |
| Primary | `A` | `T` | 12 | Evaluation of adaptive agents |
| Primary diagnostic | `O` | `T` | 12 | Estimation of adaptation cost |
| Control | `G` | `T` | 12 | Comparison with the general policy |
| Sanity | `B` | `T` | 9 | Check of simple strategies |

Total:

```text
45 matchups per model_seed
225 matchups for five model_seed
```

The family-informed Oracle knows the base opponent's true family and selects the specialist policy assigned to it from the outset. Comparison with the adaptive agent shows the difference between immediately using the family label and classifying the opponent during play. This is not a comparison with an optimal policy.

Oracle–adaptive comparisons are paired by algorithm:

```text
oracle_X - adaptive_X
```

Comparisons are not made between Oracle and adaptive agents from different algorithms.

Interpretation:

```text
small oracle-adaptive gap
    -> adaptive performs similarly to the specialist assigned to the family;
       the gap itself does not prove classification correctness

large positive oracle-adaptive gap
    -> the result is consistent with a possible cost of delayed or incorrect adaptation;
       the conclusion should be confirmed using classifier metrics

negative oracle-adaptive gap
    -> adaptive outperforms the specialist assigned to the family;
       the assigned specialist need not be the best policy in this matchup
```

### Auxiliary and Interpretive Questions

At the end of the training-opponent evaluation, you should be able to answer:

```text
1. Which algorithm performs best against opponents known from training?
2. Are adaptive agents better than policy_general?
3. How large is the gap between the adaptive agent and the Oracle?
4. Does weaker adaptive performance result from the RL algorithm or from incorrect/delayed opponent classification?
5. Does any algorithm have difficulty with a specific training archetype?
6. Do the sanity baselines avoid achieving absurdly good results?
7. Are the results stable across seeds?
```

### Expected Results File

```text
results/evaluation/training-opponent_final_algorithms_g500.csv
```

### Expected Report

```text
reports/training-opponent_final_algorithms_g500/
```

---

## 5.2. Generalization evaluation

### Objective and Research Link

Performance against held-out variants of opponent families.

```text
Link: RQ2, RQ3, RQ4, H2, H3
```

### Family Mapping

```text
calling_extreme -> calling family
aggressive_extreme -> aggressive family
tight_extreme -> tight family
```

This means:

```text
The Oracle uses specialist_calling against calling_extreme from the outset.

Adaptive should classify the opponent as calling and then switch to specialist_calling.

The same rule applies to aggressive_extreme and tight_extreme.
```

### Comparison Matrix

| Priority | Agent group | Opponent group | Number of matchups per seed | Objective |
| --- | --- | --- | ---: | --- |
| Primary | `A` | `H` | 12 | Evaluation of adaptation generalization |
| Primary diagnostic | `O` | `H` | 12 | Separation of classification from specialist generalization |
| Control | `G` | `H` | 12 | Comparison with the general policy |
| Sanity | `B` | `H` | 9 | Check of variant exploitability |

Total:

```text
45 matchups per model_seed
225 matchups for five model_seed
```

The family-informed Oracle knows the unseen variant's nominal family and uses the specialist assigned to that family from the outset. This does not mean that the selected specialist is empirically the best policy against the given variant.

Oracle–adaptive comparisons are paired by algorithm:

```text
oracle_X - adaptive_X
```

They make it possible to answer two questions:

```text
1. Does the specialist policy generalize to a variant of the same family?
2. Can the adaptive agent correctly recognize the opponent's family and select this policy?
```

Interpretation:

```text
adaptive good, oracle similar -> generalization and classification work well
adaptive weak, oracle good -> the problem is classification/adaptation, not the specialist policy itself
Oracle good against the base opponent but weak against the variant -> the specialist policy does not generalize well
adaptive better than oracle -> the unknown/mixed policy or an incorrect classification may happen to work better than the assigned family
```

### Auxiliary and Interpretive Questions

At the end of the generalization evaluation, you should be able to answer:

```text
1. Do the RL algorithms generalize to unseen opponent variants?
2. Which algorithm transfers its policy to calling_extreme best?
3. Which algorithm performs best against aggressive_extreme?
4. Which algorithm performs best against tight_extreme?
5. Does the adaptive agent actually use the correct specialist?
6. How large is the gap between the adaptive agent and the Oracle?
7. Does weak performance result from a lack of policy generalization or from incorrect opponent classification?
8. Is policy_general sometimes as good as or better than adaptive?
9. Is any opponent variant too easy for always_call or always_raise to exploit?
```

### Expected Results File

```text
results/evaluation/generalization_final_algorithms_g500.csv
```

### Expected Report

```text
reports/generalization_final_algorithms_g500/
```

---

## 5.3. Stress-test evaluation

### Objective and Research Link

Performance against extreme or manually written sanity strategies.

```text
Link: RQ5
```

### Comparison Matrix

| Priority | Agent group | Opponent group | Number of matchups per seed | Objective |
| --- | --- | --- | ---: | --- |
| Primary | `A` | `S` | 12 | Robustness of adaptive agents |
| Control | `G` | `S` | 12 | Robustness of general policies |

Total:

```text
24 matchups per model_seed
120 matchups for five model_seed
```

Baseline-vs-baseline comparisons are conducted separately in the `Baseline head-to-head sanity checks` section.

### Auxiliary and Interpretive Questions

At the end of the stress-test evaluation, you should be able to answer:

```text
1. Can the agent exploit always_call?
2. Does the agent avoid losing catastrophically to always_raise?
3. Can the agent beat a simple rule_based baseline?
4. Is a fixed policy sufficient, or does adaptation help?
5. Is any algorithm particularly vulnerable to extreme aggression?
6. Is any algorithm particularly weak against an opponent that never folds?
7. Do the results suggest that the agent learned only to exploit specific scripted archetypes?
```

### Expected Results File

```text
results/evaluation/stress_test_learned_models_g500.csv
```

### Expected Report

```text
reports/stress_test_learned_models_g500/
```

---

## 5.4. Agent cross-play

### Objective and Research Link

Comparison of learned agents with one another, for evaluation only, not during training.

```text
Link: RQ3, RQ5
```

### Comparison Matrix

| Priority | Matrix | Rule | Number of matchups per seed |
| --- | --- | --- | ---: |
| Primary | `A × A` | Directional, without self-play | 12 |
| Optional | `G × G` | Directional, without self-play | 12 |
| Optional | `A_X ↔ G_X` | Same-algorithm pairs only | 8 |

For the primary cross-play:

```text
adaptive_X vs adaptive_Y
X != Y
```

Both directions are separate matchups:

```text
adaptive_mc vs adaptive_sarsa
adaptive_sarsa vs adaptive_mc
```

The notation `A_X ↔ G_X` means both directions of comparison between adaptive and policy_general for the same algorithm. It is not the full `A × G`.

Self-play is disabled by default.

### Auxiliary and Interpretive Questions

At the end of agent cross-play, you should be able to answer:

```text
1. Which learned algorithm performs best against other learned algorithms?
2. Does an algorithm that wins against scripted opponents also win against learned policies?
3. Are adaptive agents stable against other adaptive agents?
4. Are the results symmetric when the positions A vs B and B vs A are reversed?
5. Did any algorithm learn a policy that exploited scripted opponents but performs poorly against other learned agents?
6. Does policy_general sometimes beat the adaptive agent from the same algorithm?
```

### Important Methodological Note

Cross-play must be interpreted cautiously.

It does not answer:

```text
Which algorithm plays poker best?
```

Rather, it answers:

```text
Which algorithm performs better in direct competition with other learned policies in this simplified environment?
```

### Expected Results File

```text
results/evaluation/learned_agent_cross_play_g500.csv
```

### Expected Report

```text
reports/learned_agent_cross_play_g500/
```

---

## 5.5. Baseline head-to-head sanity checks

### Objective and Research Link

Determine whether simple baselines reveal errors or extreme asymmetries in the environment.

```text
Link: RQ5
```

### Comparison Matrix

Full directional matrix:

```text
B × B
```

| Number of agents | Number of opponents | Number of matchups |
| ---: | ---: | ---: |
| 3 | 3 | 9 |

The matrix also includes three mirror matches:

```text
always_call vs always_call
always_raise vs always_raise
rule_based vs rule_based
```

### Auxiliary and Interpretive Questions

At the end of baseline head-to-head, you should be able to answer:

```text
1. Does always_raise avoid dominating the entire environment?
2. Does always_call avoid winning because of an accidental property of the simulation?
3. Does rule_based behave like a reasonable sanity baseline?
4. Are mirror-match results close to neutral?
5. Are there extreme results suggesting an error in the game, blind, or reward logic?
```

### Expected Results File

```text
results/evaluation/baseline_head_to_head_g500.csv
```

### Expected Report

```text
reports/baseline_head_to_head_g500/
```

---

# 6. Analyses and Reports

## 6.1. Algorithm comparison

### Objective

A dedicated comparison of RL algorithms.

This report does not run new games. It uses the combined results of `training-opponent evaluation` and `generalization evaluation`, filters group `A`, and then aggregates the results according to the `global_score_seed` definition from RQ3.

### Metrics Compared

The report uses the primary metric and auxiliary metrics from Section 2.2. It additionally reports:

```text
mean_profit_bb_std_across_seeds
delta_vs_monte_carlo
delta_vs_rule_based
oracle_gap_bb
non_negative_matchup_count
best_matchup_count
average_rank
```

`oracle_gap_bb` is interpreted according to the definition in RQ4.

### Auxiliary and Interpretive Questions

At the end of the algorithm comparison, you should be able to answer:

```text
1. Which RL algorithm has the best mean performance?
2. Which algorithm wins the most matchups?
3. Which algorithm has the most positive matchups?
4. Which algorithm is most stable across seeds?
5. How do Double Q-learning results against aggressive and aggressive_extreme compare with those of the other algorithms?
6. How does SARSA differ from the other algorithms in terms of mean_profit_bb, bust_rate, and variability across seeds?
7. Is Monte Carlo weaker globally, or only against specific opponents?
8. How large is the gap relative to the Oracle for each adaptive algorithm?
```

### Expected Report

```text
reports/algorithm_comparison_g500/
  algorithm_comparison.md
  algorithm_comparison.json
  algorithm_global_ranking.csv
  algorithm_by_opponent.csv
  algorithm_deltas.csv
  charts/
```

---

## 6.2. Experiment integrity checks and result diagnostics

### Objective

Verify the technical correctness of the evaluation data and identify results requiring cautious interpretation.

This stage does not run new games and does not assess whether an algorithm “should” achieve a particular result.

Checks are divided into two groups:

```text
integrity checks        -> correctness and completeness of the experiment
performance diagnostics -> properties and limitations of the results obtained
```

### Data Integrity Checks

Integrity checks determine whether the results can be used for scientific analysis.

They include:

```text
presence of required columns
correctness of model_source
absence of checkpoints in final-model results
completeness of required agents
completeness of required opponents and matchups
presence of the five required model_seed
presence of required evaluation_replicate_id for baseline-only
consistency of training_episode between compared agents
ability to pair results by model_seed
absence of duplicate game identifiers
absence of invalid NaN or infinity values
correct number of games per matchup
correct per-seed aggregation
correct aggregation across seeds
```

Failure to satisfy an integrity check may mean that the data are incomplete or were aggregated incorrectly.

### Result Diagnostics

Result diagnostics identify behavior requiring discussion, but do not determine the technical validity of the experiment.

They include:

```text
profitability diagnostics
delta vs rule_based
delta vs policy_general
classifier accuracy
classifier coverage
seed variance
extreme BB/100
always_raise dominance
always_call / always_raise exploitability
aggressive_extreme robustness
tight_extreme robustness
oracle-adaptive gap
```

For example:

```text
agent loses to always_raise
    -> weak result or policy limitation, not an experiment error

adaptive loses to policy_general
    -> no support for the hypothesis of an adaptation benefit

low classifier accuracy
    -> limitation of the opponent-profiling mechanism

high seed variance
    -> high uncertainty in the result

one of the required seeds is missing
    -> data-integrity error
```

### Meaning of Statuses

Each check should have a type:

```text
integrity
diagnostic
```

Integrity-check statuses:

```text
PASS
    The data satisfy the technical requirement.

FAIL
    The data are incomplete, inconsistent, or cannot serve as the basis for the given comparison.
```

Diagnostic statuses:

```text
PASS
    The result does not exceed the pre-established diagnostic threshold.

WARNING
    The result requires discussion as a possible weakness, instability, or limitation.

SKIPPED
    The diagnostic could not be performed or does not apply to the given agent type.
```

The `FAIL` status is reserved exclusively for technical and data-integrity problems.

Weak profitability, losing to a baseline, low classifier quality, or a large gap relative to the Oracle does not cause a `FAIL`. These are experimental results and should receive a `WARNING` or be taken into account when evaluating the hypotheses.

### Overall Report Status

The global technical status is determined solely from `integrity` checks:

```text
TECHNICAL PASS
    All required integrity checks have PASS status.

TECHNICAL FAIL
    At least one integrity check has FAIL status.
```

Diagnostic warnings do not change `TECHNICAL PASS` to `TECHNICAL FAIL`.

The report should present separately:

```text
technical_status
integrity_check_counts
diagnostic_warning_counts
skipped_check_counts
```

### Auxiliary and Interpretive Questions

At the end of the validation checks, you should be able to answer:

```text
1. Do the data pass all mandatory integrity checks?
2. Are all required algorithms, seeds, and matchups present?
3. Can comparisons be correctly paired by model_seed?
4. Which integrity checks have FAIL status?
5. Which results received a diagnostic WARNING status?
6. Could extreme BB/100 result from very short games?
7. Does high variance across seeds limit the strength of the conclusions?
8. Which warnings describe an algorithm weakness rather than a technical problem?
```

### Expected Report

```text
reports/final_algorithm_validation_g500/
  experiment_validation.md
  experiment_validation.json
```

The report should contain separate sections:

```text
Technical integrity
Performance diagnostics
```

### Final Run

The final run should enforce the complete experiment contract:

```text
--require-all-algorithms
--enforce-frozen-final-protocol
```

The `--enforce-frozen-final-protocol` option enforces the manifest, `protocol_id`,
hash consistency, namespace, five seeds, 500 games per matchup, and five
baseline-only replicates.

The companion `*.summary.json` is the evaluation manifest. The detailed JSON schema is part of the technical documentation, not the research protocol.

---

## 6.3. Per-seed stability analysis

### Objective

Assess result stability across different training seeds.

This stage is important because the final result should not depend on a single fortunate training run.

### Items Analyzed

```text
mean_profit_bb per seed
win_rate per seed
bust_rate per seed
bb_per_100 per seed
std_across_seeds
seed-level standard error
seed-level confidence interval
best seed
worst seed
seed spread
```

### Auxiliary and Interpretive Questions

At the end of the per-seed analysis, you should be able to answer:

```text
1. Does the algorithm perform consistently across seeds?
2. Does one seed substantially inflate the result?
3. Does one seed substantially depress the result?
4. Is the algorithm ranking stable across seeds?
5. Are differences between algorithms larger than variance across seeds?
```

### Planned Output

```text
reports/final_seed_stability_g500/
  seed_stability.md
  seed_stability.csv
  seed_stability.json
  charts/
```

---

## 6.4. Classifier quality analysis

### Objective

Determine whether the adaptive agent actually recognizes the opponent type correctly.

This is particularly important because the project concerns adaptation through opponent profiling.

### Items Analyzed

```text
classifier_accuracy
classifier_coverage
unknown_classification_rate
first_classification_hand
first_correct_classification_hand
policy_switches
final_predicted_type
confusion between opponent families
```

### Auxiliary and Interpretive Questions

At the end of the classifier quality analysis, you should be able to answer:

```text
1. What is classification quality per family, and what does the confusion matrix show?
2. How quickly does the agent begin to classify the opponent correctly?
3. How often does the agent remain in the unknown state?
4. Does incorrect classification explain weak results in a specific matchup?
5. Does the adaptive agent switch policies too often?
```

### Planned Output

```text
reports/final_classifier_quality_g500/
  classifier_quality.md
  classifier_quality.csv
  classifier_confusion_matrix.csv
  charts/
```

---

# 7. Final thesis-ready summary

## 7.1. Objective

Bring all results together in a single account ready for inclusion in the master's thesis.

This stage is not a new evaluation, but the final interpretation.

## 7.2. Summary Scope

```text
reference to the final configuration and methodology
global ranking and per-opponent results
stability across seeds
classifier quality
family-informed Oracle–adaptive gap
answers to RQ1–RQ5
statuses of H1–H3
limitations of interpretation
```

The final summary does not repeat the definitions of the questions and hypotheses from Section 2. For each hypothesis, it gives one of three statuses:

```text
H1: supported / unsupported / inconclusive result
H2: supported / unsupported / inconclusive result
H3: supported / unsupported / inconclusive result
```

Each answer should indicate:

```text
estimated effect value
95% confidence interval
individual seed results
scope of opponents covered by the conclusion
most important limitations of interpretation
```

## 7.3. Expected Output

```text
docs/thesis_results_notes.md
reports/final_thesis_summary_g500.md
```

---

# 8. Limitations

## 8.1. Environment Scope

The results concern a simplified heads-up Texas Hold’em environment, the defined state representation, three actions, and the established reward. They are not evidence of general poker-playing quality or performance against an optimal strategy.

## 8.2. Opponent Stationarity

Within an individual game, the opponent maintains the same behavioral archetype. Adaptation means recognizing an unknown but stationary opponent and selecting a specialist. The project does not cover detection of an actual strategy change or change points.

Opponent statistics are collected from the start of the game and reset before the next game.

## 8.3. Generalization Scope

Generalization is evaluated exclusively on the three variants in group `H`. The conclusions do not cover all possible opponents or all changes to strategy parameters.

## 8.4. Family-informed Oracle

The Oracle uses the specialist assigned to the nominal family but does not empirically search for the best policy. A negative Oracle–adaptive gap is not an experiment error.

## 8.5. Number of Seeds and Inferential Power

Five seeds are a computational compromise. Confidence intervals may remain wide, so the report must present effects, individual seed results, and uncertainty, rather than only a ranking of means.

## 8.6. Cross-play

Cross-play uses a matched-seed design: the compared models come from training runs labeled with the same seed. The analysis does not cover the full matrix of all seed pairs and is diagnostic in nature.

## 8.7. Optional Analyses

The full specialist–opponent matrix, non-stationary opponents, and extended 1000-game evaluation remain supplementary analyses or directions for future research. They are not required to evaluate H1–H3.

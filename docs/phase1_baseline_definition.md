# Frozen Phase 1 Baseline Definition

This document defines the implementation baseline to freeze before Phase 2 algorithm extensions. Q-learning, SARSA, DQN, function approximation, multi-agent RL, and proposed research extensions are outside this baseline.

## State Representation

A state is an immutable, sorted, hashable tuple of zero-based selected feature indices.

## Action Representation

An action is one feature index not already selected. Available actions are all unselected indices in the configured feature universe.

## Transition

A transition adds exactly one valid unselected feature and does not mutate the prior state.

## Reward

For a current and next state:

`reward = accuracy(next_state) - accuracy(current_state)`

Accuracy is produced by the configured SVM evaluator. Reward is not balanced accuracy.

## TD(0)

The state-value update is:

`V(S_t) <- V(S_t) + alpha * [reward + gamma * V(S_(t+1)) - V(S_t)]`

The frozen experimental reconstruction uses alpha `0.5` and gamma `0.7`, both labeled as candidates from FSRLearning rather than paper parameters.

## AOR

AOR maintains one selection count and running mean per feature. The Phase 1 experiments use the transition reward as the AOR contribution. This is a reconstruction choice because the paper's AOR notation also refers to state-value differences.

## Epsilon-Greedy Policy

Unseen states choose a random available feature. Visited states explore randomly with epsilon and otherwise exploit the available feature with maximum AOR, breaking ties by smallest feature index.

The experiment grid is epsilon `0.3, 0.4, 0.5, 0.6`, inferred from Figure 2.

## SVM Evaluation

- Kernel: RBF/Gaussian
- `C=1.0`
- `gamma="scale"`
- Scaling: none
- Strategy: stratified 80/20 holdout
- Metric: accuracy
- Split random state: run seed

These are project reconstruction settings, not complete paper specifications.

## Datasets and Preprocessing

- Australian: 14 numeric-encoded inspected input columns, binary target, no missing values observed in the supplied raw file.
- WPBC: 32 numerical variables only; identifier and time excluded; target N/R; mean imputation performed inside training splits through the evaluator pipeline. This is the explicit resolution used for the Phase 1 WPBC experiment, while the paper's 34-feature count remains a limitation.
- Sonar: 60 numeric features, R/M target, no missing-value treatment required.

## Hyperparameters and Seeds

- Episodes: 100
- Initial subset size: 1
- Seeds: 2021, 2022, 2023, 2024, 2025
- Maximum episode steps: feature count minus one initial feature
- AOR contribution mode: `reward`

Seeds and initial-state size are project choices. The paper specifies random initial states but not their complete distribution.

## Freeze Boundary

Future Q-learning/SARSA or other Phase 2 work must compare against this baseline without silently changing these definitions. Any change must be introduced as a named experimental variant with its own recorded configuration.

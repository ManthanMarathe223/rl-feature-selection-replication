# Average of Reward (AOR) Protocol

This document defines Phase 1E only: an independent, tested feature-scoring data structure. It does not implement action selection, epsilon-greedy exploration, episodes, TD(0) changes, Q-learning, SARSA, function approximation, or RL training.

## Purpose

AOR records the historical average contribution associated with selecting each feature. The tracker maintains one selection count and one running AOR value per feature, and can provide a deterministic ranking.

## Running-Average Formula

For the $k$-th observation of feature $f$:

$$
AOR_{new} = \frac{(k - 1)AOR_{old} + contribution}{k}
$$

For the first observation, $k=1$, so the AOR equals the contribution and the selection count becomes one.

## Contribution Semantics

The tracker accepts a generic finite numeric `contribution`. It intentionally does not decide whether that value is:

- the raw accuracy-difference reward;
- a state-value difference; or
- another quantity defined by a later protocol.

No evaluator, reward function, TD agent, environment, or action-selection policy is called by `AORTracker`.

## Interface

- `record(feature_index, contribution)` and `update(feature_index, contribution)` add one observation;
- `get_aor(feature_index)` returns the running score;
- `get_count(feature_index)` returns the selection count;
- `reset()` sets every count to zero and every score to `0.0`;
- `all_scores()` returns all scores by feature index; and
- `ranking()` orders features by descending AOR, breaking ties by ascending feature index.

Feature indices must be non-negative integers within the configured feature count. Contributions must be finite numeric values.

## Paper Evidence

The paper explicitly describes AOR as a feature-scoring criterion based on historical contribution/reward and gives the incremental average formula using the number of times a feature has been selected. The paper's notation connects AOR to state-value differences, while its earlier reward definition uses `accuracy(next_state) - accuracy(current_state)`.

That notation relationship is intentionally left unresolved here. This tracker stores the supplied contribution without interpreting it.

## Next Phase Boundary

The next phase may decide how a contribution is produced and how AOR scores participate in action selection. Those decisions are not part of this data structure and are not implemented here.
# State and Action Protocol

This document defines the Phase 1B feature-selection environment only. It does not define or implement learning, rewards, AOR, exploration, or episodes.

## State

A state is the current subset of selected, zero-based feature indices. The implementation stores the subset as a sorted tuple, so equivalent input orderings have identical equality and hashing.

Examples:

```text
empty state: {}
state with features 0 and 2: {0, 2}
```

`FeatureSubsetState` is immutable and hashable. It exposes selected features, the number selected, membership testing, and terminal testing against a total feature count.

## Action

An action is one feature index not already in the current state. Available actions are returned in ascending deterministic order.

For a total feature count of 5:

```text
state {}       -> actions {0, 1, 2, 3, 4}
state {0, 2}   -> actions {1, 3, 4}
```

Actions are rejected when they are negative, outside `0..total_features-1`, non-integer, or already selected.

## Transition

`transition(state, action)` returns a new state containing exactly one additional feature. It does not mutate the original state.

```text
{0, 2} + action 4 -> {0, 2, 4}
```

The `FeatureSelectionEnvironment.step(action)` method applies the same transition to its current state. It manages only state, action, and terminal status.

## Terminal Condition

A state is terminal when its selected feature count equals the environment's total feature count. An environment rejects further steps after reaching terminal state.

## Direct Paper Basis

The 2021 paper directly defines:

- state as a selected feature subset;
- action as selecting an unselected feature;
- transitions as sequentially adding features; and
- the terminal state as the state containing all features.

The deterministic tuple representation, zero-based indices, validation errors, and environment API are implementation choices for a reproducible software interface. They are not additional claims about the paper.

## Explicitly Out of Scope

This Phase 1B environment does not calculate accuracy, rewards, AOR, or values. It does not implement TD(0), Q-learning, SARSA, epsilon-greedy action selection, or episode execution.
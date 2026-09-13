# Exploration and Exploitation Policy

This document defines Phase 1F only: action selection from an already supplied state, AOR tracker, and visit history. It does not implement episodes, learning, rewards, evaluation, AOR updates, TD(0), Q-learning, SARSA, or policy training.

## Unseen States

If a state has not been explicitly marked as visited, the policy selects one available feature uniformly through its configured random number generator. AOR is not consulted for an unseen state.

The caller controls visit history with:

- `mark_visited(state)`;
- `has_visited(state)`; and
- `reset_visits()`.

Choosing an action does not automatically mark the state visited.

## Epsilon-Greedy Behavior

For a previously visited state:

- with probability `epsilon`, the policy explores by choosing an available feature randomly;
- otherwise, it exploits by selecting the available feature with the highest AOR.

`epsilon` is validated in the inclusive range `[0, 1]` and must be finite. A seeded or injected `random.Random` instance makes the random sequence reproducible.

The paper's Figure 2 displays experiments at `epsilon = 0.3, 0.4, 0.5, 0.6`. These are figure-inferred experiment values, not an unresolved project default and not a value inserted into configuration.

## Available Actions

Available actions are exactly the feature indices not already present in the state. If no available action exists, `choose_action` raises a terminal-state error.

## Exploitation and Ties

Exploitation reads AOR values through `AORTracker.get_aor`. It does not update AOR. If multiple available features have equal maximum AOR, the smallest feature index is selected deterministically.

## Direct Paper Support

Directly supported by the paper:

- unseen states use random action selection;
- previously seen states use an epsilon-greedy exploration/exploitation behavior;
- exploration selects an available feature randomly;
- exploitation uses the feature with the highest AOR; and
- actions are limited to unselected features.

The deterministic tie-break, explicit visit-management API, random-generator injection, and terminal error are implementation choices for reproducibility and testability.

## Policy Versus Learning

The policy only reads:

- the current immutable state;
- available actions;
- AOR scores; and
- visit information.

It does not calculate rewards, update AOR, update TD values, call the SVM evaluator, mutate the environment, or run episodes. The epsilon value remains a policy configuration parameter and is not inferred from FSRLearning defaults.

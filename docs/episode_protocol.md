# Phase 1G Episode Protocol

This document describes the paper-oriented episode runner only. It connects the existing state/action environment, SVM evaluator, reward calculator, TD(0) estimator, AOR tracker, and exploration/exploitation policy. It does not add a learning algorithm or change any component's mathematical behavior.

## Episode Sequence

For each episode:

1. Use an explicitly supplied initial state, or generate a random non-terminal feature subset.
2. Read available actions from the environment.
3. Ask the policy to choose an action. The policy handles unseen-state random selection and visited-state epsilon-greedy behavior.
4. Apply the action through the environment, adding one unselected feature.
5. Evaluate current and next states through the existing reward/evaluation layer.
6. Calculate `accuracy(next_state) - accuracy(current_state)`.
7. Pass the reward to the existing TD(0) estimator.
8. Record an explicitly configured contribution in the existing AOR tracker.
9. Mark the new state as visited.
10. Stop at terminal state or the configured maximum step count.

The runner does not select a policy, update AOR internally, calculate rewards itself, or call the SVM evaluator directly.

## Initial State

The paper specifies random initial states but does not fully specify the initial-state distribution.

The runner therefore exposes minimum and maximum initial subset sizes and uses an injected random generator. The generated subset is always non-terminal. An explicit initial state is respected, including a terminal state, which produces a zero-step terminal result.

FSRLearning's empty-state default is not used as the paper protocol.

## AOR Contribution Mode

The paper's notation connects AOR to state-value differences, while the earlier reward definition uses accuracy differences. The runner does not silently resolve this ambiguity. `aor_contribution_mode` must be selected explicitly:

- `reward`: record the transition reward;
- `td_state_difference`: record `old_value - next_state_value`, the state-value difference corresponding to the paper's AOR notation.

If the mode is `null`, the runner fails when an AOR update would be required.

## Termination

The runner stops when all features have been selected or when `maximum_episode_steps` is reached. It does not implement FSRLearning's worsen-patience rule.

## Reproducibility

The initial-state generator uses the injected runner random generator. The policy separately uses its configured random generator or seed. Reproducibility therefore requires the same dataset, component configurations, initial-state bounds, and random seeds/generator states.

## Episode Result

`EpisodeResult` records the initial and final states, state history, actions, rewards, accuracy observations, TD errors, selected features, step count, and termination reason.

## Paper-Supported Behavior

The paper supports the broad sequence of random initialization, available-feature actions, state transitions, accuracy-difference reward, TD state-value update, AOR update, and terminal completion. Exact initial-state distribution, evaluation settings, contribution interpretation, and reproducibility parameters remain reconstruction choices or unresolved experimental details.
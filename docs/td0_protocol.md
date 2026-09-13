# TD(0) Protocol

This document defines Phase 1D only: the tabular TD(0) state-value update. It does not implement Q-learning, SARSA, AOR, epsilon-greedy selection, policy selection, episode execution, or RL training.

## Equation

The paper directly specifies:

$$
V(S_t) \leftarrow V(S_t) + \alpha [r_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

The implementation computes the temporal-difference error first:

$$
\delta_t = r_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

and then updates:

$$
V(S_t) \leftarrow V(S_t) + \alpha \delta_t
$$

## Parameters

- **alpha:** learning rate controlling how strongly the current value moves toward the TD target. The implementation requires `0 < alpha < 1`.
- **gamma:** discount factor controlling the contribution of the next state's value. The implementation allows `0 <= gamma <= 1`.

The paper explains the roles and ranges of alpha and gamma but does **not** specify exact values. The project configuration leaves `td.alpha` and `td.gamma` as `null` reconstruction placeholders. FSRLearning's `alpha=0.5` and `gamma=0.70` are external candidate values, not paper settings.

## Numerical Example

Given:

```text
V(current) = 0.2
V(next) = 0.8
reward = 0.1
alpha = 0.5
gamma = 0.7
```

The TD error is:

```text
0.1 + 0.7 * 0.8 - 0.2 = 0.46
```

The updated current value is:

```text
0.2 + 0.5 * 0.46 = 0.43
```

Unseen states default to value `0.0` unless explicitly initialized with `set_value`.

## Update Result

Each update returns the current state, next state, reward, old current value, next-state value, TD error, new current value, alpha, and gamma. States are immutable and hashable, so their keys remain stable in the value dictionary.

## Direct Paper Support

Directly supported:

- TD(0) state-value learning is used;
- the equation above;
- alpha is a learning-rate parameter;
- gamma is a discount factor; and
- the update uses the reward and next-state value.

## Unresolved Details

The paper does not specify exact alpha or gamma values, initialization beyond the conceptual value table, random seeds, episode scheduling, stopping rules beyond the broader feature-selection description, or any complete training protocol. Those concerns remain outside this mathematical update layer.
# Frozen Phase 1 Baseline Definition

This document establishes the official, frozen baseline definition for Phase 1 replication of the paper *Reinforcement Learning for Feature Selection*.

All Phase 1 runs across **Australian Credit Approval**, **WPBC**, and **Sonar** datasets are complete. This definition serves as the immutable control against which future Phase 2 research (e.g., Q-Learning, SARSA, function approximation, DQN, or alternative cross-validation protocols) must be evaluated.

---

## 1. State Representation

- A state $S$ is an immutable, sorted, hashable tuple of zero-based selected feature indices:
  $$S = (f_{(1)}, f_{(2)}, \dots, f_{(k)}), \quad f_{(i)} \in \{0, 1, \dots, M-1\}$$
- Initial state: An arbitrary single-feature subset ($|S_0| = 1$), chosen uniformly at random using the run's random seed.
- Terminal state: The full feature universe ($|S_{\text{term}}| = M$), where all available features have been added.

---

## 2. Action Representation & Transitions

- **Action Space**: At state $S$, the set of available actions $A(S)$ consists of all unselected feature indices:
  $$A(S) = \{f \in \{0, 1, \dots, M-1\} \setminus S\}$$
- **Transition**: Choosing action $a \in A(S)$ transitions deterministically to next state $S' = S \cup \{a\}$, ordered and deduplicated.
- Transitions are strictly forward-additive; features are never removed within an episode.

---

## 3. Reward Formulation

For any state transition from $S_t$ to $S_{t+1}$:
$$\text{Reward}(S_t, S_{t+1}) = \text{Accuracy}(S_{t+1}) - \text{Accuracy}(S_t)$$

- Classification accuracy is computed by the configured SVM evaluator on the holdout test set.
- Reward is not balanced accuracy, F1-score, or AUC.

---

## 4. Temporal Difference Learning: TD(0)

State values $V(S)$ are updated online using one-step tabular Temporal Difference learning:
$$V(S_t) \leftarrow V(S_t) + \alpha \Big[ \text{Reward}(S_t, S_{t+1}) + \gamma V(S_{t+1}) - V(S_t) \Big]$$

- **Learning rate ($\alpha$)**: $0.5$ (reconstruction candidate derived from FSRLearning; not stated in paper).
- **Discount factor ($\gamma$)**: $0.7$ (reconstruction candidate derived from FSRLearning; not stated in paper).
- **Initial state values**: $V(S) = 0.0$ for all unvisited states.

---

## 5. Action-Outcome-Reward (AOR)

The AOR tracker maintains running frequencies and cumulative average rewards per individual feature across all transitions:
$$\text{AOR}(a) = \frac{1}{N_a} \sum_{i=1}^{N_a} r_{a, i}$$

- Contribution mode: Each transition adds the scalar transition reward ($\Delta \text{Acc}$) to the selected feature's AOR pool (`aor_contribution_mode="reward"`).
- Unselected features default to $\text{AOR}(f) = 0.0$.

---

## 6. Epsilon-Greedy Exploration Policy

At state $S$ with available actions $A(S)$:
1. **Unvisited State**: If $S$ has not been previously visited, select an action $a \in A(S)$ uniformly at random.
2. **Visited State**:
   - With probability $\epsilon$: select a random exploratory action $a \in A(S)$.
   - With probability $1 - \epsilon$: exploit by choosing the available feature that maximizes AOR:
     $$a^* = \arg\max_{a \in A(S)} \text{AOR}(a)$$
   - Ties in $\text{AOR}$ are broken deterministically by selecting the smallest feature index.
- **Exploration grid**: $\epsilon \in \{0.3, 0.4, 0.5, 0.6\}$ (inferred from Figure 2 in the paper).

---

## 7. SVM Evaluator Configuration

- **Estimator**: Scikit-Learn `SVC`
- **Kernel**: Radial Basis Function (RBF / Gaussian)
- **Regularization ($C$)**: $1.0$
- **Kernel Bandwidth ($\gamma$)**: `'scale'` ($1 / (n\_features \cdot \text{Var}(X))$)
- **Feature Scaling**: None (`scaling="none"`)
- **Evaluation Strategy**: Stratified 80% train / 20% test holdout split
- **Split Random State**: Fixed to the run's random seed (`random_state=seed`)
- **Scoring Metric**: Classification accuracy ($\frac{\text{correct}}{\text{total}}$)

---

## 8. Datasets and Preprocessing

| Dataset | Total Samples | Total Features ($M$) | Class Target | Missing Values Policy | Preprocessing Notes |
|---|---|---|---|---|---|
| **Australian** | 690 | 14 | Column 14 (binary 0/1) | None | 14 numeric-encoded attributes |
| **WPBC** | 198 | 32 | Column 1 (R/N) | Mean imputation | 32 continuous attributes; ID (col 0) and Time (col 2) excluded |
| **Sonar** | 208 | 60 | Column 60 (R/M) | None | 60 continuous sonar chirp attributes |

---

## 9. Experimental Execution Grid

- **Episodes per run**: 100
- **Epsilon values**: 4 ($\epsilon \in \{0.3, 0.4, 0.5, 0.6\}$)
- **Seeds**: 5 ($2021, 2022, 2023, 2024, 2025$)
- **Total runs per dataset**: $4 \times 5 = 20$ runs
- **Total Phase 1 runs**: $20 \times 3 = 60$ completed runs
- **Steps per episode**: $M - 1$ (until all features are added)

---

## 10. Summary of Replicated Baseline Metrics

The official aggregated results generated from the 60 completed runs are stored in:
- `results/tables/phase1_final_comparison.csv`
- `results/tables/phase1_epsilon_comparison.csv`

| Dataset | Paper Accuracy | Replicated Peak Mean Accuracy | Replicated Std | Best $\epsilon$ | Best State Feature Count | Mean Runtime (Best $\epsilon$) |
|---|---|---|---|---|---|---|
| **Australian** | $85.55 \pm 0.039\%$ | **$88.26\%$** | $\pm 0.71\%$ | $0.5$ | $5.00 \pm 0.89$ | $59.13\text{ s}$ |
| **WPBC** | $76.29 \pm 0.007\%$ | **$78.50\%$** | $\pm 1.22\%$ | $0.4$ ($^*$tie $0.5$) | $2.00 \pm 0.00$ | $51.68\text{ s}$ |
| **Sonar** | $73.69 \pm 0.108\%$ | **$95.71\%$** | $\pm 3.16\%$ | $0.5$ | $13.60 \pm 5.08$ | $56.90\text{ s}$ |

---

## 11. Freeze Boundary & Phase 2 Rules

This definition is strictly **frozen**.
1. **No Reruns or Tweaks**: Phase 1 code, parameters, and tables must not be modified or re-executed.
2. **Phase 2 Comparison Protocol**: Any algorithm extension proposed in Phase 2 (such as Q-learning, SARSA, or DQN) must be implemented as a separate module and compared directly against these documented baseline figures under identical evaluation conditions.

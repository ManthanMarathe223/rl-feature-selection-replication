# Phase 1 Replication Protocol

**Paper:** Sali Rasoul, Sodiq Adewole, and Alphonse Akakpo, *Feature Selection Using Reinforcement Learning*, arXiv:2101.09460 (2021)

**Phase:** Protocol preparation only. TD(0), Q-learning, SARSA, and all RL execution remain unimplemented.

## Purpose

This protocol converts the completed investigation into an auditable Phase 1 plan. It distinguishes:

1. values directly specified by the paper;
2. values inferred from paper figures;
3. values obtained only from the independent `blefo/FSRLearning` implementation; and
4. reconstruction assumptions introduced by this project.

An FSRLearning value is an implementation reference, not evidence that the 2021 paper used that value. An inferred figure value is also not an official paper setting.

The machine-readable version is [phase1_protocol.csv](../results/tables/phase1_protocol.csv). The broader evidence record is [replication_protocol.md](replication_protocol.md).

## Phase 1 Protocol Table

| Parameter | Value(s) | Evidence source | Confidence | Notes |
|---|---|---|---|---|
| State | Selected feature subset | Directly specified by paper | Exact | The state space is the power set of the feature set. |
| Action | Select one unselected feature | Directly specified by paper | Exact | The action space contains features not already in the current subset. |
| State transition | Add one selected feature sequentially | Directly specified by paper | Exact | No feature removal is described. |
| Reward | `accuracy(next state) - accuracy(current state)` | Directly specified by paper | Exact | The classifier/evaluation protocol behind accuracy remains unresolved. |
| TD state-value update | `V(S_t) <- V(S_t) + alpha * [r_(t+1) + gamma * V(S_(t+1)) - V(S_t)]` | Directly specified by paper | Exact | The update will not be implemented in this phase. |
| AOR feature scoring | Average feature reward/state-value contribution with incremental mean | Directly specified by paper | Partially specified | Equation and prose establish the concept; exact operational linkage to reward needs confirmation. |
| Exploration policy | Epsilon-greedy exploration/exploitation | Directly specified by paper | Exact | The paper describes random exploration and exploitation using AOR values. |
| Epsilon experiment values | `0.3, 0.4, 0.5, 0.6` | Inferred from Figure 2 | Exact | These are the displayed experiment values, not an official hidden default, and must not be replaced by FSRLearning's defaults. |
| Episode horizon shown | Approximately 100 episodes | Inferred from Figure 2 | Partially specified | The figure extends to approximately 100; exact loop count is not established by the paper text. |
| Alpha | Unresolved; candidate `0.5` | FSRLearning only for candidate reconstruction | Not specified by paper | `0.5` is not adopted as an official paper value. It is a sensitivity-analysis candidate only. |
| Gamma | Unresolved; candidate `0.70` | FSRLearning only for candidate reconstruction | Not specified by paper | `0.70` is not adopted as an official paper value. It is a sensitivity-analysis candidate only. |
| Number of episodes/iterations | Approximately 100 as a figure-based working horizon; exact value unresolved | Figure inference plus reconstruction planning | Partially specified | Do not treat 100 as a paper-confirmed parameter. |
| Initial state | Random feature subset per episode | Directly specified by paper | Partially specified | The paper does not specify the random-state distribution or seed. FSRLearning's empty default is not substituted. |
| Stopping criterion | All features included is terminal; additional stopping rule unresolved | Directly specified by paper plus unresolved detail | Partially specified | FSRLearning's worsen rule is not adopted as the paper rule. |
| SVM kernel | Gaussian/RBF SVM | Directly specified by paper | Partially specified | The Gaussian/RBF family is specified; library, implementation, and numeric kernel parameter are unresolved. |
| SVM C | Unresolved | No value in paper, figure, FSRLearning, or UCI docs | Not specified | Must remain a sensitivity-analysis parameter. |
| SVM gamma | Unresolved | No value in paper, figure, FSRLearning, or UCI docs | Not specified | “Gaussian” identifies the family, not a numeric gamma. |
| Feature scaling | Unresolved | Not specified by paper; no scaling documented in FSRLearning example | Not specified | No scaling is authorized by this protocol until evidence is found. |
| Train/test split | Unresolved | Not specified by paper; FSRLearning reward uses cross-validation instead | Not specified | The current SVM sanity-check split is not the Phase 1 paper protocol. |
| Random seed | Unresolved | Not specified by paper or FSRLearning | Not specified | A seed must be selected for reproducibility only after its sensitivity role is documented. |
| WPBC feature universe | Option A or B remains open | UCI docs plus paper's inconsistent count | Partially specified | See [wpbc_replication_options.md](wpbc_replication_options.md); do not select automatically. |
| WPBC missing values | Treatment unresolved | UCI documents four missing lymph-node values; raw inspection locates them | Partially specified | Do not automatically drop or impute in the replication protocol. |

## Reconstruction Assumptions Requiring Sensitivity Analysis

These are planning assumptions, not claims about the paper:

- **Exact alpha:** unresolved. Use FSRLearning's `0.5` only as one candidate in a sensitivity analysis.
- **Exact gamma:** unresolved. Use FSRLearning's `0.70` only as one candidate in a sensitivity analysis.
- **Train/test split:** unresolved. The existing baseline's stratified 80/20 split with seed 42 is an implementation sanity check only, not an official Phase 1 setting.
- **SVM C:** unresolved. No value is documented by the paper or FSRLearning.
- **SVM gamma:** unresolved. The paper's Gaussian/RBF description does not specify a numeric gamma; `gamma='scale'` is only a current baseline assumption.
- **Scaling:** unresolved. Compare no scaling against any evidence-supported alternative only after documenting the reason; do not silently scale.
- **Random seed:** unresolved. Record every seed in experiment outputs; do not claim that seed 42 came from the paper.
- **Stopping criterion:** unresolved beyond the terminal all-features condition. FSRLearning's worsen patience rule is an external implementation behavior, not a paper setting.
- **Episode count:** approximately 100 is a figure-based horizon, not a verified exact count. Sensitivity should include nearby horizons once the RL implementation exists.
- **SVM evaluation metric/protocol:** the paper says accuracy, while FSRLearning uses cross-validated balanced accuracy with a Random Forest by default. These must not be conflated.

## Protocol Boundaries

This document does not authorize:

- implementing TD(0) or any other RL algorithm;
- copying FSRLearning's Random Forest or balanced-accuracy evaluation into this project;
- selecting a WPBC interpretation;
- dropping or imputing WPBC rows;
- changing the current SVM evaluator;
- modifying raw datasets.

Before implementation begins, each unresolved item must either be supported by additional evidence or explicitly assigned a sensitivity-analysis grid and recorded as a project assumption.

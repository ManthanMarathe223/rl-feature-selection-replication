# Replication Protocol Investigation

**Paper:** Sali Rasoul, Sodiq Adewole, and Alphonse Akakpo, *Feature Selection Using Reinforcement Learning*, arXiv:2101.09460 (2021)

**Status:** Investigation only. This document does not implement TD(0), Q-learning, SARSA, or any other RL algorithm.

## Scope and Evidence

This investigation compares four evidence sources:

1. The 2021 paper, including its arXiv HTML version and reported equations/tables.
2. The independent public `blefo/FSRLearning` repository, version 1.0.7 as represented by its public `main` branch and example.
3. UCI dataset documentation and the raw files already present in this repository.
4. The current replication implementation.

No value is called a paper setting unless the paper actually documents it. The machine-readable parameter matrix is in [replication_parameters.csv](../results/tables/replication_parameters.csv).

## Confidence Labels

- **Exact:** The source states the behavior or value directly.
- **Partially specified:** The source states part of the behavior, gives alternatives/defaults, or leaves an implementation detail unresolved.
- **Not specified:** The source does not provide the requested parameter.

## Parameter Matrix

| Parameter | Paper | FSRLearning | UCI documentation | Our current implementation | Confidence |
|---|---|---|---|---|---|
| alpha | Constant in `(0, 1)`; no value | Default `0.5` | Not applicable | No RL; config `null` | Partially specified |
| gamma | Constant in `(0, 1)`; no value | Default `0.70` | Not applicable | No RL; config `null` | Partially specified |
| epsilon | Exploration is discussed and varied; values not stated in text | Default `0.1`; example `0.2` | Not applicable | No RL; config `null` | Partially specified |
| Episodes/iterations | Around 50th episode appears in a plot; total run count not stated | Default `100`; example `200` | Not applicable | No RL; config `null` | Partially specified |
| Initial state | Random state per episode | Default empty; optional random | Not applicable | Empty scaffold state; policy absent | Partially specified |
| State transition | Add an unseen feature | Add one unselected feature | Not applicable | `FeatureSubsetState.add` | Exact |
| Stopping condition | All features is terminal; other stopping detail unclear | All features or worsen-based stop | Not applicable | No RL termination policy | Partially specified |
| SVM kernel | Gaussian SVM | Default is Random Forest; caller supplies classifier | Not prescribed | RBF `SVC` baseline assumption | Partially specified |
| SVM C | Not specified | Not applicable to default RF | Not prescribed | `C=1.0` assumption | Not specified |
| SVM gamma | Not specified | Not applicable to default RF | Not prescribed | `gamma='scale'` assumption | Not specified |
| Feature scaling | Not specified | No scaling documented | Native types/ranges only | No scaling or imputation | Not specified |
| Train/test split | Not specified | Five-fold CV for reward by default | No split prescribed | Stratified 80/20, seed 42 | Not specified |
| Random seed | Not specified | No explicit seed | No seed prescribed | Baseline split seed 42; RL unresolved | Not specified |
| Reward | Accuracy difference: next state minus current state | Difference of state CV balanced-accuracy scores | Labels only | Conceptual accuracy difference helper | Exact |
| AOR | Mean feature reward; incremental equation given, notation is partly ambiguous | Two-row count/mean table; mean uses current state value | Not applicable | Global mean helper only | Partially specified |

## Paper Findings

The paper establishes the conceptual method but omits enough operational detail to reproduce numerical results directly:

- States are feature subsets; actions are unselected features.
- Episodes start from a random feature set in the paper's trajectory description.
- Features are added sequentially until all features are included, a stopping condition is reached, or the process returns to random exploration.
- The reward is the accuracy change between consecutive subsets.
- TD state values use alpha and gamma, but only their interpretation and range are given.
- AOR ranks features from accumulated reward/state-value differences.
- SVM is described as Gaussian, but no kernel library, `C`, SVM gamma, split, seed, scaling, or fixed evaluation protocol is reported.
- The dataset table reports Australian 14/690, WPBC 34/198, and Sonar 60/208. The WPBC count requires separate investigation below.

## FSRLearning Findings

Public repository: [blefo/FSRLearning](https://github.com/blefo/FSRLearning).

This repository is useful implementation evidence, but it is not proof of the original paper's hidden settings:

- `FeatureSelectorRL` defaults: `eps=0.1`, `alpha=0.5`, `gamma=0.70`, `nb_iter=100`, and `starting_state='empty'`.
- The example constructs the selector with `eps=0.2` and `nb_iter=100`; the README also shows `nb_iter=200` as an example.
- Its default reward classifier is `RandomForestClassifier(n_jobs=-1)`, not the paper's Gaussian SVM.
- State reward uses `cross_val_score`, normally `cv=5`, with `balanced_accuracy`; very small class counts alter the fold count.
- No train/test split, SVM `C`, SVM gamma, or explicit random seed was found in the inspected implementation.
- States add one feature not already present. The implementation tracks an explored-state graph and compares states by feature-set membership.
- It stops when all features are selected or when the value worsens for `round(sqrt(feature_number))` consecutive checks.
- AOR is a two-row table: selection count and running mean. The update is `((k - 1) * old + current_state.v_value) / k`.

The current project deliberately does not copy these RL settings yet.

## UCI Dataset Evidence

- Australian: 690 instances and 14 feature attributes, with mixed categorical, integer, and real-valued types. UCI documents missing values in the dataset family; the local raw inspection found none in the supplied file.
- WPBC: 198 instances. The UCI page reports 33 features in its summary while the accompanying `.names` file describes 34 non-target attributes: ID, time, and 32 numeric variables. The raw file therefore has 35 fields including the N/R outcome.
- Sonar: 208 instances and 60 continuous features; labels are `R` and `M`; UCI reports no missing values.

UCI documentation describes data structure and missingness, not the paper's RL parameters, classifier settings, split, or seed.

## WPBC Structure Investigation

The raw WPBC row layout is:

1. identifier
2. N/R outcome target
3. time
4-35. 32 numeric variables

The paper's reported “34 features” can therefore mean several things:

### Interpretation A: UCI non-target attribute count

The count is the 34 fields other than the outcome: identifier, time, and 32 numeric variables. This is supported by the `.names` documentation's statement that the dataset has 34 attributes when the outcome is treated as the class field. It would mean the paper used “features” in a broad raw-column sense, not necessarily model inputs.

### Interpretation B: All 34 non-target fields used as predictors

The count matches the raw non-target width exactly: identifier + time + 32 numeric variables. This is numerically possible, but the identifier is metadata and time may encode follow-up information, so there is no evidence yet that both were supplied to the SVM or RL feature space.

### Interpretation C: 32 numerical model inputs, with the paper using a loose count

The 32 numeric variables are the clearest candidate for predictive inputs after excluding identifier and time metadata. This is supported by the UCI `.names` file's explicit description of 32 real-valued variables, but it does not numerically match the paper's “34” statement.

A fourth arithmetic possibility is to include exactly one metadata field: 32 numeric variables plus time or identifier gives 33, not 34. Thus neither time-only nor identifier-only inclusion explains the paper count without also including the other metadata field.

The investigation does **not** select among A, B, and C.

## WPBC Missing Values

The UCI documentation identifies lymph node status as missing in four cases. The local raw inspection confirms literal `?` markers in raw column 35 (`feature_32`) on 1-based rows 7, 29, 86, and 197.

No treatment is selected here. The current baseline excludes those rows solely because scikit-learn `SVC` requires finite numeric inputs; this is an executable sanity-check assumption, not an established replication protocol. Candidate treatments to investigate later are:

- preserve rows and use a verified, documented imputation rule;
- preserve rows but use an estimator/evaluation path that supports missing values;
- exclude rows, only if the paper/source implementation or a justified protocol supports complete-case analysis.

The raw files remain unchanged.

## Open Decisions Before RL

1. Obtain or reconstruct the exact paper evaluation protocol, including SVM implementation and data split.
2. Decide whether the RL feature universe for WPBC is 32 numeric variables or includes metadata, with evidence.
3. Decide how the four lymph-node missing values are handled without silently changing the dataset.
4. Establish a seed policy and record whether the paper's random initial states can be replicated.
5. Reconcile the paper's reward wording with the AOR equation and FSRLearning's state-value update.

Until these are resolved, no RL implementation should be treated as a faithful replication.

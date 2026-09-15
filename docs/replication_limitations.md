# Replication Limitations & Methodological Deviations

The original paper (*Reinforcement Learning for Feature Selection*) establishes a novel heuristic framework for feature space exploration using TD(0) learning and Action-Outcome-Reward (AOR) statistics. However, it does not specify sufficient operational details to enable an exact numerical reproduction.

This document provides a comprehensive inventory of the limitations, ambiguities, reconstruction assumptions, and statistical deviations identified during Phase 1 replication.

---

## 1. Underspecified Algorithmic and Learning Hyperparameters

The paper does not report exact numerical values or procedures for key reinforcement learning components:

- **Learning Rate ($\alpha$)**: Not reported. Our replication uses $\alpha = 0.5$, adopted from the independent FSRLearning implementation.
- **Discount Factor ($\gamma$)**: Not reported. Our replication uses $\gamma = 0.7$, adopted from FSRLearning.
- **Initial State Distribution**: The paper states that episodes begin with a "random subset of features", but does not specify the probability distribution over subset sizes. Our replication fixes the initial subset size to 1 feature ($|S_0|=1$) to observe full forward-selection trajectories.
- **Episode Horizon and Termination**: The paper does not define an explicit stopping criterion or horizon per episode. Our replication terminates an episode when all features in the universe have been selected ($M-1$ transitions), recording intermediate evaluations at each step.
- **AOR Contribution Interpretation**: The paper's mathematical formulation of AOR introduces ambiguous notation that alternates between transition accuracy differences ($\Delta \text{Acc}$) and state-value differences ($\Delta V$). Our Phase 1 experiments strictly use transition reward ($\Delta \text{Acc}$) as the AOR contribution (`aor_contribution_mode="reward"`).

---

## 2. Evaluation Protocol Underspecification and Selection Bias (The Sonar Divergence)

The paper states that classification performance was measured using a Support Vector Machine (SVM) with a Gaussian/RBF kernel, but omits essential evaluation parameters:

- **Train/Test Splitting Protocol**: The paper does not specify whether accuracy was obtained via a fixed train/test split, repeated holdout, or $k$-fold cross-validation.
- **SVM Hyperparameters**: Neither the regularization parameter $C$ nor the RBF kernel width $\gamma$ (or bandwidth) is reported. Our replication uses Scikit-Learn defaults ($C=1.0$, $\gamma=\text{"scale"}$).
- **Feature Scaling**: Support Vector Machines with RBF kernels are highly sensitive to feature scaling. The paper does not state whether min-max scaling, standardization (z-score), or raw feature values were utilized. Our baseline uses unscaled raw inputs (`scaling="none"`).
- **Validation Overfitting / Optimism Bias**:
  - In our baseline holdout setup, the dataset is partitioned into an 80% train and 20% test split.
  - On the **Sonar** dataset ($N=208$ total instances), the test set contains only 42 samples.
  - Over 100 episodes, the RL agent explores between 5,600 and 5,900 unique state transitions, evaluating the SVM on that same 42-sample test set at every step.
  - Searching thousands of candidate subsets against a small static holdout partition inevitably causes extreme **multiple testing bias** (data snooping / validation leakage). The agent identifies combinations of features that happen to perfectly fit idiosyncratic noise in those 42 test points, resulting in an observed mean accuracy of **$95.71\%$** (and up to $100.0\%$ on seed 2025), compared to the paper's reported **$73.69 \pm 0.108\%$** ($+22.02\%$ gap).
  - The original paper almost certainly used 10-fold cross-validation or evaluated feature subsets on an independent, unseen test set that was completely isolated from the feature selection trajectory.

---

## 3. Dataset-Specific Structural Limitations

### 3.1. Australian Credit Approval
- **Input Dimension**: 14 features (6 continuous, 8 categorical numeric-encoded).
- **Missing Values**: None observed in the raw UCI `.dat` file.
- **Replication Outcome**: Achieved $88.26 \pm 0.71\%$ accuracy (best $\epsilon=0.5$), exceeding the paper's $85.55 \pm 0.039\%$.
- **Nuance**: The agent selects compact subsets averaging 5.0 features. Terminal 14-feature accuracy degrades to $\approx 65\% - 70\%$.

### 3.2. WPBC (Wisconsin Prognostic Breast Cancer)
- **Feature Count Discrepancy**:
  - The paper reports **34 features** for WPBC.
  - The standard UCI repository file contains 34 columns: Patient ID, Outcome (R/N), Time-to-recurrence, and 31 continuous cytological attributes (or 32 continuous attributes if including time).
  - Using Patient ID or Time-to-recurrence as input features is methodologically invalid (introducing spurious correlation or post-outcome target leakage).
  - Our replication explicitly isolates the **32 continuous clinical/cytological attributes** and excludes ID and Time. The 32 vs 34 feature discrepancy is a confirmed limitation of the published literature.
- **Missing Value Imputation**:
  - The raw dataset contains 4 missing values in the Lymph Node Status attribute.
  - Our pipeline applies mean imputation strictly inside the training fold of each split to prevent data leakage. The paper does not describe any missing value treatment.
- **Replication Outcome**: Achieved $78.50 \pm 1.22\%$ accuracy (best $\epsilon=0.4/0.5$), exceeding the paper's $76.29 \pm 0.007\%$. The best observed states require only 2 features.

### 3.3. Sonar (Connectionist Bench Sonar)
- **Input Dimension**: 60 continuous sonar chirp band energy features.
- **Missing Values**: None.
- **Replication Outcome**: Reached $95.71 \pm 3.16\%$ accuracy, diverging by $+22.02\%$ from the paper's $73.69 \pm 0.108\%$.
- **Runtime Anomaly**: Seeds 2021 and 2022 under $\epsilon=0.4$ recorded runtimes of ~35,000s each due to host system background suspension during execution, while other runs completed in ~55s to ~175s.

---

## 4. Divergence From the FSRLearning Reference Implementation

While the open-source FSRLearning package provides reference code for TD state updates and AOR tracking, it cannot be used as an authoritative paper proxy:
- **Evaluator**: FSRLearning defaults to a Random Forest classifier (`n_estimators=10`), whereas the paper explicitly specifies Gaussian/RBF SVM.
- **Metric**: FSRLearning optimizes 3-fold cross-validated **balanced accuracy**, whereas the paper specifies standard **classification accuracy**.
- **Conclusion**: FSRLearning served as evidence for candidate hyperparameters ($\alpha=0.5, \gamma=0.7$), but our project strictly adheres to the paper's SVM and classification accuracy specifications.

---

## 5. Epistemic Boundaries for Replication Assessment

1. **Numerical Proximity $\neq$ Protocol Verification**: Achieving $88.26\%$ on Australian (close to $85.55\%$) or $78.50\%$ on WPBC (close to $76.29\%$) does not prove that our hyperparameters match the original authors'. The agreement reflects similar inductive bias rather than identical experimental execution.
2. **Divergence $\neq$ Implementation Error**: The $+22.02\%$ divergence on Sonar is an expected statistical consequence of holdout evaluation on small sample sizes ($N=42$ test points) with massive multiple testing, rather than a bug in the TD(0) or AOR code.
3. **Phase 2 Baseline Requirement**: All Phase 2 algorithm comparisons (e.g., Q-learning, SARSA, DQN, nested CV) must benchmark against these documented Phase 1 results under identical evaluation conditions before introducing alternative protocol modifications.

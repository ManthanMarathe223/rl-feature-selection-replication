# Phase 1A Evaluation Protocol

This document covers the reusable feature-subset evaluation engine only. It does not implement TD(0), Q-learning, SARSA, AOR, epsilon-greedy selection, or RL episodes.

## Evidence Boundary

The 2021 paper directly specifies a Gaussian/RBF SVM as the classifier used to evaluate feature subsets and defines the later reward as:

`accuracy(next_state) - accuracy(current_state)`

The paper does **not** specify SVM `C`, SVM gamma, scaling, a train/test split, a random seed, or a complete evaluation protocol. Those settings must remain configurable and must not be described as paper settings.

The independent FSRLearning implementation is not copied as the paper protocol. It defaults to a Random Forest and cross-validated balanced accuracy, while this project uses accuracy only for Phase 1A. FSRLearning's behavior is relevant investigation evidence, not an authority for the paper's SVM settings.

## Configurable Engine

`EvaluationConfig` controls:

- SVM `kernel`, `C`, and `gamma`;
- `scaling`: `none` or `standard`;
- `random_state`;
- `evaluation_strategy`: `holdout` or `cross_validation`;
- `scoring_metric`, currently restricted to `accuracy`;
- holdout `test_size`;
- cross-validation `cv_folds`;
- `missing_value_policy`: `error`, `drop_rows`, or `mean_impute`.

The evaluator returns accuracy, sample count, selected-feature count, runtime, training time when available, strategy, and serialized model configuration.

### Holdout

The holdout strategy uses a stratified split with configurable `test_size` and `random_state`. The current runner's values (`test_size=0.2`, `random_state=42`, RBF, `C=1.0`, `gamma='scale'`, no scaling) are project baseline assumptions only.

### Cross-validation

The cross-validation strategy uses configurable `cv_folds` and the `accuracy` scoring metric. Balanced accuracy is intentionally not used.

## Feature Policies

The baseline runner supports:

- `numeric_only`: the currently inspected numeric input features;
- `all_non_target`: all non-target fields except the identifier by default.

Identifier inclusion requires the explicit `include_identifier=True` option. No policy automatically resolves the WPBC 32-versus-34 question. See [wpbc_replication_options.md](wpbc_replication_options.md).

## Missing Values

Missing-value handling is explicit:

- `error` refuses to evaluate data containing missing values;
- `drop_rows` removes affected rows only when that policy is explicitly selected and records the observed count in the baseline CSV;
- `mean_impute` fits a mean imputer inside the evaluation pipeline.

The current baseline runner explicitly selects `drop_rows` so the existing WPBC sanity check remains executable. This is not a replication conclusion and does not modify the raw dataset. No missing-value approach is claimed to come from the paper.

## Outputs

Run from the project root:

```powershell
python -m experiments.baseline.run_svm_evaluation
```

Results are written to `results/tables/svm_evaluation_baseline.csv`. The output records feature policy, identifier policy, observed missing rows, accuracy, runtime, training time, evaluation strategy, and model configuration.

## Phase 1A Boundary

This stage evaluates supplied feature subsets. It does not choose subsets, calculate RL rewards between states, update state values, rank features, or execute episodes.
# Reward Protocol

This document defines Phase 1C only. It does not implement TD(0), Q-learning, SARSA, AOR, epsilon-greedy selection, RL agents, or episode training.

## Formula

The 2021 paper defines the reward for selecting feature $f$ as:

$$
Reward_f = Accuracy(next\_state) - Accuracy(current\_state)
$$

The implementation exposes this formula as `calculate_reward(current_accuracy, next_accuracy)`.

## Meaning of the Reward

- **Positive reward:** the next feature subset has higher model accuracy.
- **Negative reward:** the next feature subset has lower model accuracy.
- **Zero reward:** both feature subsets have the same model accuracy.

Accuracy and reward are distinct quantities. Accuracy is the evaluator's score for one feature subset. Reward is the signed difference between two independently evaluated accuracies. Reward is not balanced accuracy and is not itself a classifier metric.

## Transition Evaluation

`evaluate_transition_reward` evaluates the current and next states independently using the existing SVM evaluator. It returns:

- current accuracy;
- next accuracy;
- accuracy-difference reward;
- current feature count; and
- next feature count.

No evaluation results are cached. No value function, AOR table, policy, or episode state is introduced here.

## Direct Paper Support

Directly supported by the paper:

- the accuracy-difference formula;
- evaluation of the current and next feature subsets; and
- the interpretation of reward as the effect of adding a feature.

## Unresolved Experimental Details

The paper does not fully specify the SVM implementation, train/test or cross-validation protocol, SVM `C`, SVM gamma, scaling, random seed, or missing-value handling. Those choices remain controlled by the existing configurable evaluation engine and are not resolved by this reward layer.

The reward layer therefore consumes evaluator accuracy without changing how accuracy is produced. It does not claim that any current evaluator setting originated in the paper.
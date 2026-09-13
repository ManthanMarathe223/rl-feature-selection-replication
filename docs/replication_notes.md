# Replication Notes

Working notes for reconstructing the methodology in *Feature Selection Using Reinforcement Learning* (Rasoul, Adewole, and Akakpo, 2021).

## Paper Methodology

_To be filled from the paper and supporting evidence._

## State Representation

A state represents a selected feature subset. Record the exact encoding, ordering, and terminal-state semantics here once verified.

## Action Representation

Actions represent features that are not yet selected. Record whether an action can revisit, remove, or only add features.

## Reward

The starter implementation uses the conceptual change in classifier accuracy:

`reward = accuracy(next_state) - accuracy(current_state)`

The evaluation split and accuracy protocol remain unresolved.

## TD(0)

The original Phase 1 agent will use a state-value update. Q-learning and SARSA are explicitly out of scope for this phase.

## Average of Reward (AOR)

Document the definition, initialization, and incremental update used by the paper and compare it with the implementation.

## Exploration and Exploitation

Record the exploration policy, epsilon schedule, tie-breaking behavior, and random-state handling.

## Episode Trajectory

Document initialization, state visits, action selection, transitions, rewards, updates, and termination in execution order.

## Evaluation Process

Document how selected subsets are evaluated, which metrics are reported, and how results are persisted under `results/`.

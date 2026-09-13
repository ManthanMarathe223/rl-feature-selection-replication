"""Exploration/exploitation action-selection policy for feature subsets."""

from __future__ import annotations

import random
from math import isfinite
from numbers import Real

from src.rl.actions import available_actions
from src.rl.aor import AORTracker
from src.rl.state import FeatureSubsetState


class FeatureSelectionPolicy:
    """Choose available features using unseen-state and epsilon-greedy rules.

    The policy reads AOR scores and visit information only. It does not update
    AOR, values, rewards, or environment state.
    """

    def __init__(
        self,
        total_features: int,
        aor_tracker: AORTracker,
        epsilon: Real,
        *,
        random_seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not isinstance(total_features, int) or isinstance(total_features, bool):
            raise TypeError("total_features must be a non-negative integer.")
        if total_features < 0:
            raise ValueError("total_features must be a non-negative integer.")
        if not isinstance(aor_tracker, AORTracker):
            raise TypeError("aor_tracker must be an AORTracker.")
        if aor_tracker.number_of_features != total_features:
            raise ValueError("AOR tracker size must match total_features.")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError("epsilon must be a numeric value.")
        epsilon_value = float(epsilon)
        if not isfinite(epsilon_value):
            raise ValueError("epsilon must be finite.")
        if not 0 <= epsilon_value <= 1:
            raise ValueError("epsilon must satisfy 0 <= epsilon <= 1.")
        if rng is not None and random_seed is not None:
            raise ValueError("Provide rng or random_seed, not both.")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random.")

        self.total_features = total_features
        self.aor_tracker = aor_tracker
        self.epsilon = epsilon_value
        self._rng = rng if rng is not None else random.Random(random_seed)
        self._visited_states: set[FeatureSubsetState] = set()

    def mark_visited(self, state: FeatureSubsetState) -> None:
        """Record a state as seen by the caller."""
        self._visited_states.add(state)

    def has_visited(self, state: FeatureSubsetState) -> bool:
        """Return whether the state has been explicitly marked as visited."""
        return state in self._visited_states

    def reset_visits(self) -> None:
        """Forget all recorded state visits without changing AOR scores."""
        self._visited_states.clear()

    def choose_action(self, state: FeatureSubsetState) -> int:
        """Choose one available feature according to the policy."""
        actions = available_actions(state, self.total_features)
        if not actions:
            raise RuntimeError("No available actions: the state is terminal.")

        if not self.has_visited(state):
            return self._rng.choice(actions)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        return min(
            actions,
            key=lambda feature_index: (
                -self.aor_tracker.get_aor(feature_index),
                feature_index,
            ),
        )

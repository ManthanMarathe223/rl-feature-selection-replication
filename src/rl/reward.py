"""Reward calculation for feature-selection transitions."""

from collections.abc import Callable

from src.rl.state import FeatureSubsetState


def accuracy_difference(
    current_state: FeatureSubsetState,
    next_state: FeatureSubsetState,
    accuracy: Callable[[FeatureSubsetState], float],
) -> float:
    """Return ``accuracy(next_state) - accuracy(current_state)``."""
    return float(accuracy(next_state) - accuracy(current_state))

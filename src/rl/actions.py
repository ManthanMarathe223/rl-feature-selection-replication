"""Action helpers for feature-subset states."""

from collections.abc import Iterable

from src.rl.state import FeatureSubsetState


def available_actions(
    state: FeatureSubsetState, number_of_features: int
) -> tuple[int, ...]:
    """Return unselected feature indices that can be added to ``state``."""
    if number_of_features < 0:
        raise ValueError("number_of_features must be non-negative.")
    return tuple(
        index
        for index in range(number_of_features)
        if index not in state.selected_features
    )

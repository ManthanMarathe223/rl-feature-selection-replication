"""Action validation and deterministic state transitions."""

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


def validate_action(
    state: FeatureSubsetState, action: int, number_of_features: int
) -> None:
    """Validate that an action is an unselected feature in the domain."""
    if number_of_features < 0:
        raise ValueError("number_of_features must be non-negative.")
    if not isinstance(action, int) or isinstance(action, bool):
        raise TypeError("Action must be an integer feature index.")
    if action < 0 or action >= number_of_features:
        raise ValueError(
            f"Action {action} is outside feature indices 0 through {number_of_features - 1}."
        )
    if state.contains(action):
        raise ValueError(f"Action {action} is already selected.")


def transition(
    state: FeatureSubsetState, action: int, number_of_features: int | None = None
) -> FeatureSubsetState:
    """Return a new state after adding exactly one valid feature."""
    if number_of_features is not None:
        validate_action(state, action, number_of_features)
    elif not isinstance(action, int) or isinstance(action, bool):
        raise TypeError("Action must be an integer feature index.")
    elif action < 0:
        raise ValueError("Feature index must be non-negative.")
    elif state.contains(action):
        raise ValueError(f"Action {action} is already selected.")
    return state.add(action)

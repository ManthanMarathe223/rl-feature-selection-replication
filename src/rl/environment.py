"""Minimal feature-selection state/action environment."""

from dataclasses import dataclass, field

from src.rl.actions import available_actions, transition
from src.rl.state import FeatureSubsetState


@dataclass
class FeatureSelectionEnvironment:
    """Manage feature-subset states and transitions without rewards or learning."""

    total_features: int
    current_state: FeatureSubsetState = field(default_factory=FeatureSubsetState)

    def __post_init__(self) -> None:
        if self.total_features < 0:
            raise ValueError("total_features must be non-negative.")
        self._validate_state(self.current_state)

    def _validate_state(self, state: FeatureSubsetState) -> None:
        state.is_terminal(self.total_features)

    @property
    def terminated(self) -> bool:
        """Return whether the current state contains every feature."""
        return self.current_state.is_terminal(self.total_features)

    def reset(self, state: FeatureSubsetState | None = None) -> FeatureSubsetState:
        """Reset to the empty state or a validated supplied state."""
        next_state = state if state is not None else FeatureSubsetState()
        self._validate_state(next_state)
        self.current_state = next_state
        return self.current_state

    def available_actions(self) -> tuple[int, ...]:
        """Return valid unselected feature indices."""
        return available_actions(self.current_state, self.total_features)

    def step(self, action: int) -> FeatureSubsetState:
        """Apply one action and return the new current state."""
        if self.terminated:
            raise RuntimeError("Cannot step from a terminal state.")
        self.current_state = transition(
            self.current_state, action, self.total_features
        )
        return self.current_state
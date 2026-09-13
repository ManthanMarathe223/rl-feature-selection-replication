"""Tabular TD(0) state-value update mechanism."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

from src.rl.state import FeatureSubsetState


@dataclass(frozen=True)
class TD0UpdateResult:
    """Inputs, intermediate values, and result of one TD(0) update."""

    current_state: FeatureSubsetState
    next_state: FeatureSubsetState
    reward: float
    old_value: float
    next_state_value: float
    td_error: float
    new_value: float
    alpha: float
    gamma: float


def _finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value.")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


@dataclass
class TD0ValueEstimator:
    """Deterministic tabular TD(0) state-value estimator."""

    alpha: float
    gamma: float
    values: dict[FeatureSubsetState, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.alpha = _finite_number(self.alpha, "alpha")
        self.gamma = _finite_number(self.gamma, "gamma")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must satisfy 0 < alpha < 1.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must satisfy 0 <= gamma <= 1.")
        self.values = {
            state: _finite_number(value, "state value")
            for state, value in self.values.items()
        }

    def get_value(self, state: FeatureSubsetState) -> float:
        """Return a state's value, defaulting to zero when unseen."""
        return self.values.get(state, 0.0)

    def value(self, state: FeatureSubsetState) -> float:
        """Compatibility alias for ``get_value``."""
        return self.get_value(state)

    def set_value(self, state: FeatureSubsetState, value: Real) -> None:
        """Explicitly initialize or replace a state's value."""
        self.values[state] = _finite_number(value, "state value")

    def update(
        self,
        current_state: FeatureSubsetState,
        reward: Real,
        next_state: FeatureSubsetState,
    ) -> TD0UpdateResult:
        """Apply exactly one TD(0) state-value update."""
        reward_value = _finite_number(reward, "reward")
        old_value = self.get_value(current_state)
        next_state_value = self.get_value(next_state)
        td_error = reward_value + self.gamma * next_state_value - old_value
        new_value = old_value + self.alpha * td_error
        self.values[current_state] = new_value
        return TD0UpdateResult(
            current_state=current_state,
            next_state=next_state,
            reward=reward_value,
            old_value=old_value,
            next_state_value=next_state_value,
            td_error=td_error,
            new_value=new_value,
            alpha=self.alpha,
            gamma=self.gamma,
        )


# Preserve the existing public class name while exposing the Phase 1D API.
TDAgent = TD0ValueEstimator

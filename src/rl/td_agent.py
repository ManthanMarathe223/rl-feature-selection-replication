"""TD(0) state-value agent skeleton."""

from dataclasses import dataclass, field

from src.rl.state import FeatureSubsetState


@dataclass
class TDAgent:
    """Tabular TD(0) agent for state-value estimates.

    Q-learning and SARSA are intentionally outside this Phase 1 interface.
    """

    alpha: float
    gamma: float
    values: dict[FeatureSubsetState, float] = field(default_factory=dict)

    def value(self, state: FeatureSubsetState) -> float:
        """Return the current estimate for ``state``."""
        return self.values.get(state, 0.0)

    def update(
        self,
        state: FeatureSubsetState,
        reward: float,
        next_state: FeatureSubsetState,
    ) -> float:
        """Apply TD(0) and return the updated value.

        V(S_t) <- V(S_t) + alpha * [r_(t+1) + gamma * V(S_(t+1)) - V(S_t)]
        """
        current_value = self.value(state)
        target = reward + self.gamma * self.value(next_state)
        updated_value = current_value + self.alpha * (target - current_value)
        self.values[state] = updated_value
        return updated_value

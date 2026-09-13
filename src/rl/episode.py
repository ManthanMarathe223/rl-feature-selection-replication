"""Episode orchestration skeleton for TD(0) feature selection."""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.rl.aor import AverageReward
from src.rl.reward import accuracy_difference
from src.rl.state import FeatureSubsetState
from src.rl.td_agent import TDAgent


@dataclass
class Episode:
    """Hold the state and update components needed for one episode."""

    agent: TDAgent
    average_reward: AverageReward = field(default_factory=AverageReward)
    state: FeatureSubsetState = field(default_factory=FeatureSubsetState)
    terminated: bool = False

    def initialize(self, state: FeatureSubsetState | None = None) -> None:
        """Initialize or reset the episode state."""
        self.state = state or FeatureSubsetState()
        self.terminated = False

    def visit_state(self) -> FeatureSubsetState:
        """Return the current state."""
        return self.state

    def select_action(self, actions: tuple[int, ...]) -> int:
        """Select an action; the paper's policy is pending verification."""
        raise NotImplementedError("The paper's action-selection policy is unresolved.")

    def transition(self, feature_index: int) -> FeatureSubsetState:
        """Apply an add-feature transition."""
        self.state = self.state.add(feature_index)
        return self.state

    def calculate_reward(
        self,
        previous_state: FeatureSubsetState,
        accuracy: Callable[[FeatureSubsetState], float],
    ) -> float:
        """Calculate reward for the current transition."""
        return accuracy_difference(previous_state, self.state, accuracy)

    def update_td(self, previous_state: FeatureSubsetState, reward: float) -> float:
        """Update the TD(0) value estimate for the latest transition."""
        return self.agent.update(previous_state, reward, self.state)

    def update_aor(self, reward: float) -> float:
        """Update the episode's average reward tracker."""
        return self.average_reward.update(reward)

    def terminate(self) -> None:
        """Mark the episode as terminated."""
        self.terminated = True

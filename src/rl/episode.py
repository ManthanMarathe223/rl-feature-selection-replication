"""Paper-oriented episode orchestration for feature selection."""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from src.rl.aor import AverageReward
from src.rl.reward import accuracy_difference
from src.rl.state import FeatureSubsetState
from src.rl.td_agent import TDAgent


AORContributionMode = Literal["reward", "td_state_difference"]


@dataclass(frozen=True)
class EpisodeResult:
    """Trace and termination metadata from one feature-selection episode."""

    initial_state: FeatureSubsetState
    final_state: FeatureSubsetState
    number_of_steps: int
    states: tuple[FeatureSubsetState, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    accuracies: tuple[float, ...]
    td_errors: tuple[float, ...]
    selected_features: tuple[int, ...]
    termination_reason: str


def random_initial_state(
    total_features: int,
    rng: random.Random,
    minimum_size: int = 1,
    maximum_size: int | None = None,
) -> FeatureSubsetState:
    """Generate a random non-terminal feature subset.

    The subset-size bounds are reconstruction choices; the paper does not
    specify their distribution.
    """
    if total_features <= 0:
        raise ValueError("A random initial state requires at least one feature.")
    maximum = total_features - 1 if maximum_size is None else maximum_size
    if not 1 <= minimum_size <= maximum <= total_features - 1:
        raise ValueError(
            "Initial subset bounds must satisfy 1 <= minimum <= maximum < total_features."
        )
    size = rng.randint(minimum_size, maximum)
    return FeatureSubsetState(tuple(rng.sample(range(total_features), size)))


class FeatureSelectionEpisodeRunner:
    """Connect the existing state, policy, evaluator, reward, TD, and AOR APIs."""

    def __init__(
        self,
        feature_count: int,
        environment: Any,
        evaluator: Any,
        reward_calculator: Callable[..., Any],
        td_estimator: Any,
        aor_tracker: Any,
        policy: Any,
        random_generator: random.Random,
        maximum_episode_steps: int,
        *,
        X: Any = None,
        y: Any = None,
        minimum_initial_subset_size: int = 1,
        maximum_initial_subset_size: int | None = None,
        aor_contribution_mode: AORContributionMode | None = None,
    ) -> None:
        if feature_count < 0:
            raise ValueError("feature_count must be non-negative.")
        if maximum_episode_steps < 0:
            raise ValueError("maximum_episode_steps must be non-negative.")
        if not isinstance(random_generator, random.Random):
            raise TypeError("random_generator must be an instance of random.Random.")
        if aor_contribution_mode not in {None, "reward", "td_state_difference"}:
            raise ValueError(
                "aor_contribution_mode must be None, 'reward', or 'td_state_difference'."
            )
        self.feature_count = feature_count
        self.environment = environment
        self.evaluator = evaluator
        self.reward_calculator = reward_calculator
        self.td_estimator = td_estimator
        self.aor_tracker = aor_tracker
        self.policy = policy
        self.random_generator = random_generator
        self.maximum_episode_steps = maximum_episode_steps
        self.X = X
        self.y = y
        self.minimum_initial_subset_size = minimum_initial_subset_size
        self.maximum_initial_subset_size = maximum_initial_subset_size
        self.aor_contribution_mode = aor_contribution_mode

    def _initial_state(self, initial_state: FeatureSubsetState | None) -> FeatureSubsetState:
        if initial_state is not None:
            initial_state.is_terminal(self.feature_count)
            return initial_state
        return random_initial_state(
            self.feature_count,
            self.random_generator,
            self.minimum_initial_subset_size,
            self.maximum_initial_subset_size,
        )

    def _aor_contribution(self, reward_result: Any, td_result: Any) -> float:
        if self.aor_contribution_mode == "reward":
            return float(reward_result.reward)
        if self.aor_contribution_mode == "td_state_difference":
            return float(td_result.old_value - td_result.next_state_value)
        raise ValueError(
            "aor_contribution_mode is unresolved; select 'reward' or "
            "'td_state_difference' before an AOR update."
        )

    def run_episode(
        self, initial_state: FeatureSubsetState | None = None
    ) -> EpisodeResult:
        """Run state/action transitions until terminal or the step limit."""
        state = self._initial_state(initial_state)
        self.environment.reset(state)
        states = [state]
        actions: list[int] = []
        rewards: list[float] = []
        accuracies: list[float] = []
        td_errors: list[float] = []

        if state.is_terminal(self.feature_count):
            return EpisodeResult(
                state,
                state,
                0,
                tuple(states),
                (),
                (),
                (),
                (),
                state.selected_features,
                "all_features_selected",
            )

        if self.aor_contribution_mode is None and self.maximum_episode_steps > 0:
            raise ValueError(
                "aor_contribution_mode is unresolved; select 'reward' or "
                "'td_state_difference' before an AOR update."
            )

        termination_reason = "maximum_episode_steps"
        for _ in range(self.maximum_episode_steps):
            available = self.environment.available_actions()
            if not available:
                termination_reason = "all_features_selected"
                break
            action = self.policy.choose_action(state)
            if action not in available:
                raise ValueError("Policy returned an unavailable action.")
            next_state = self.environment.step(action)
            reward_result = self.reward_calculator(
                self.evaluator,
                self.X,
                self.y,
                state,
                next_state,
            )
            td_result = self.td_estimator.update(
                state,
                reward_result.reward,
                next_state,
            )
            contribution = self._aor_contribution(reward_result, td_result)
            self.aor_tracker.record(action, contribution)
            self.policy.mark_visited(next_state)

            actions.append(action)
            rewards.append(float(reward_result.reward))
            accuracies.extend(
                [float(reward_result.current_accuracy), float(reward_result.next_accuracy)]
            )
            td_errors.append(float(td_result.td_error))
            states.append(next_state)
            state = next_state

            if state.is_terminal(self.feature_count):
                termination_reason = "all_features_selected"
                break

        return EpisodeResult(
            initial_state=states[0],
            final_state=state,
            number_of_steps=len(actions),
            states=tuple(states),
            actions=tuple(actions),
            rewards=tuple(rewards),
            accuracies=tuple(accuracies),
            td_errors=tuple(td_errors),
            selected_features=state.selected_features,
            termination_reason=termination_reason,
        )


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

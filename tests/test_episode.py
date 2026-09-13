"""Tests for the Phase 1G paper-oriented episode runner."""

import random
import unittest
from dataclasses import dataclass

from src.rl.aor import AORTracker
from src.rl.environment import FeatureSelectionEnvironment
from src.rl.episode import FeatureSelectionEpisodeRunner
from src.rl.policy import FeatureSelectionPolicy
from src.rl.reward import TransitionReward
from src.rl.state import FeatureSubsetState


@dataclass(frozen=True)
class FakeTDResult:
    old_value: float
    next_state_value: float
    td_error: float


class FakeTD:
    def __init__(self) -> None:
        self.calls: list[tuple[FeatureSubsetState, float, FeatureSubsetState]] = []

    def update(self, current_state, reward, next_state):
        self.calls.append((current_state, reward, next_state))
        return FakeTDResult(0.0, 0.0, reward)


class CountingAOR(AORTracker):
    def __init__(self, number_of_features: int) -> None:
        super().__init__(number_of_features)
        self.calls: list[tuple[int, float]] = []

    def record(self, feature_index, contribution):
        self.calls.append((feature_index, contribution))
        return super().record(feature_index, contribution)


class FakeRewardCalculator:
    def __init__(self) -> None:
        self.calls: list[tuple[FeatureSubsetState, FeatureSubsetState]] = []

    def __call__(self, evaluator, X, y, current_state, next_state):
        self.calls.append((current_state, next_state))
        current_accuracy = current_state.number_selected / 10
        next_accuracy = next_state.number_selected / 10
        return TransitionReward(
            current_accuracy=current_accuracy,
            next_accuracy=next_accuracy,
            reward=next_accuracy - current_accuracy,
            current_feature_count=current_state.number_selected,
            next_feature_count=next_state.number_selected,
        )


def make_runner(
    seed: int = 7,
    *,
    initial_state: FeatureSubsetState | None = None,
    maximum_steps: int = 3,
    aor_mode: str | None = "reward",
):
    del initial_state
    tracker = CountingAOR(4)
    policy = FeatureSelectionPolicy(4, tracker, epsilon=0.0, random_seed=seed)
    reward_calculator = FakeRewardCalculator()
    td = FakeTD()
    runner = FeatureSelectionEpisodeRunner(
        feature_count=4,
        environment=FeatureSelectionEnvironment(4),
        evaluator=object(),
        reward_calculator=reward_calculator,
        td_estimator=td,
        aor_tracker=tracker,
        policy=policy,
        random_generator=random.Random(seed),
        maximum_episode_steps=maximum_steps,
        minimum_initial_subset_size=1,
        maximum_initial_subset_size=1,
        aor_contribution_mode=aor_mode,
    )
    return runner, reward_calculator, td, tracker


class EpisodeRunnerTests(unittest.TestCase):
    def test_random_initial_state_is_valid_and_nonterminal(self) -> None:
        runner, _, _, _ = make_runner()
        result = runner.run_episode()
        self.assertGreater(result.initial_state.number_selected, 0)
        self.assertFalse(result.initial_state.is_terminal(4))

    def test_provided_initial_state_is_respected(self) -> None:
        runner, _, _, _ = make_runner(maximum_steps=1)
        initial = FeatureSubsetState((1, 3))
        result = runner.run_episode(initial)
        self.assertEqual(result.initial_state, initial)
        self.assertEqual(result.states[0], initial)

    def test_actions_are_valid_and_add_one_feature(self) -> None:
        runner, _, _, _ = make_runner(maximum_steps=2)
        result = runner.run_episode(FeatureSubsetState((0,)))
        for previous, action, following in zip(
            result.states, result.actions, result.states[1:]
        ):
            self.assertNotIn(action, previous.selected_features)
            self.assertEqual(following.number_selected, previous.number_selected + 1)

    def test_state_history_and_action_counts(self) -> None:
        runner, _, _, _ = make_runner(maximum_steps=2)
        result = runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(len(result.states), result.number_of_steps + 1)
        self.assertEqual(len(result.actions), len(result.states) - 1)

    def test_rewards_match_accuracy_difference(self) -> None:
        runner, reward_calculator, _, _ = make_runner(maximum_steps=2)
        result = runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(len(result.rewards), len(reward_calculator.calls))
        for reward in result.rewards:
            self.assertAlmostEqual(reward, 0.1)

    def test_td_and_aor_are_called_once_per_transition(self) -> None:
        runner, _, td, tracker = make_runner(maximum_steps=2)
        result = runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(len(td.calls), result.number_of_steps)
        self.assertEqual(len(tracker.calls), result.number_of_steps)

    def test_terminal_termination(self) -> None:
        runner, _, _, _ = make_runner(maximum_steps=5)
        result = runner.run_episode(FeatureSubsetState((0, 1, 2)))
        self.assertEqual(result.termination_reason, "all_features_selected")
        self.assertEqual(result.final_state.number_selected, 4)

    def test_maximum_step_termination(self) -> None:
        runner, _, _, _ = make_runner(maximum_steps=1)
        result = runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(result.number_of_steps, 1)
        self.assertEqual(result.termination_reason, "maximum_episode_steps")

    def test_same_seed_reproduces_trajectory(self) -> None:
        first, _, _, _ = make_runner(seed=11, maximum_steps=3)
        second, _, _, _ = make_runner(seed=11, maximum_steps=3)
        first_result = first.run_episode()
        second_result = second.run_episode()
        self.assertEqual(first_result.states, second_result.states)
        self.assertEqual(first_result.actions, second_result.actions)

    def test_different_seeds_can_change_trajectory(self) -> None:
        first, _, _, _ = make_runner(seed=11, maximum_steps=3)
        second, _, _, _ = make_runner(seed=12, maximum_steps=3)
        self.assertNotEqual(first.run_episode().states, second.run_episode().states)

    def test_invalid_aor_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_runner(aor_mode="unknown")

    def test_unresolved_aor_mode_fails_before_update(self) -> None:
        runner, _, td, tracker = make_runner(aor_mode=None)
        with self.assertRaisesRegex(ValueError, "aor_contribution_mode"):
            runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(td.calls, [])
        self.assertEqual(tracker.calls, [])

    def test_td_state_difference_mode_is_explicit(self) -> None:
        runner, _, _, tracker = make_runner(aor_mode="td_state_difference")
        runner.run_episode(FeatureSubsetState((0,)))
        self.assertEqual(tracker.calls[0][1], 0.0)

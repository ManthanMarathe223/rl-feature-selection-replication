"""Tests for the Phase 1F feature-selection policy."""

import unittest

from src.rl.aor import AORTracker
from src.rl.policy import FeatureSelectionPolicy
from src.rl.state import FeatureSubsetState


class FeatureSelectionPolicyTests(unittest.TestCase):
    def make_policy(
        self, epsilon: float = 0.0, seed: int | None = 7
    ) -> tuple[FeatureSelectionPolicy, AORTracker]:
        tracker = AORTracker(4)
        return (
            FeatureSelectionPolicy(
                total_features=4,
                aor_tracker=tracker,
                epsilon=epsilon,
                random_seed=seed,
            ),
            tracker,
        )

    def test_unseen_state_returns_valid_available_feature(self) -> None:
        policy, _ = self.make_policy()
        action = policy.choose_action(FeatureSubsetState((0, 2)))
        self.assertIn(action, (1, 3))

    def test_unseen_state_excludes_selected_features(self) -> None:
        policy, _ = self.make_policy()
        for _ in range(10):
            self.assertIn(policy.choose_action(FeatureSubsetState((0, 2))), (1, 3))

    def test_epsilon_one_always_explores_seen_state(self) -> None:
        policy, tracker = self.make_policy(epsilon=1.0, seed=3)
        tracker.record(1, 1.0)
        policy.mark_visited(FeatureSubsetState())
        actions = {policy.choose_action(FeatureSubsetState()) for _ in range(20)}
        self.assertEqual(actions, {0, 1, 2, 3})

    def test_epsilon_zero_always_exploits_seen_state(self) -> None:
        policy, tracker = self.make_policy(epsilon=0.0)
        tracker.record(3, 1.0)
        policy.mark_visited(FeatureSubsetState())
        self.assertEqual(policy.choose_action(FeatureSubsetState()), 3)

    def test_exploitation_chooses_highest_available_aor(self) -> None:
        policy, tracker = self.make_policy()
        tracker.record(1, 0.2)
        tracker.record(2, 0.8)
        policy.mark_visited(FeatureSubsetState())
        self.assertEqual(policy.choose_action(FeatureSubsetState()), 2)

    def test_selected_highest_aor_is_excluded(self) -> None:
        policy, tracker = self.make_policy()
        tracker.record(0, 1.0)
        tracker.record(2, 0.5)
        state = FeatureSubsetState((0,))
        policy.mark_visited(state)
        self.assertEqual(policy.choose_action(state), 2)

    def test_equal_aor_uses_smallest_feature_index(self) -> None:
        policy, tracker = self.make_policy()
        tracker.record(1, 0.4)
        tracker.record(2, 0.4)
        policy.mark_visited(FeatureSubsetState())
        self.assertEqual(policy.choose_action(FeatureSubsetState()), 1)

    def test_same_seed_produces_same_sequence(self) -> None:
        first, _ = self.make_policy(epsilon=1.0, seed=19)
        second, _ = self.make_policy(epsilon=1.0, seed=19)
        state = FeatureSubsetState((0,))
        first.mark_visited(state)
        second.mark_visited(state)
        self.assertEqual(
            [first.choose_action(state) for _ in range(8)],
            [second.choose_action(state) for _ in range(8)],
        )

    def test_different_seeds_produce_valid_sequences(self) -> None:
        first, _ = self.make_policy(seed=1)
        second, _ = self.make_policy(seed=2)
        state = FeatureSubsetState((0,))
        first_actions = [first.choose_action(state) for _ in range(8)]
        second_actions = [second.choose_action(state) for _ in range(8)]
        self.assertTrue(all(action in (1, 2, 3) for action in first_actions))
        self.assertTrue(all(action in (1, 2, 3) for action in second_actions))
        self.assertNotEqual(first_actions, second_actions)

    def test_invalid_epsilon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_policy(epsilon=-0.1)
        with self.assertRaises(ValueError):
            self.make_policy(epsilon=1.1)
        with self.assertRaises(ValueError):
            self.make_policy(epsilon=float("nan"))
        with self.assertRaises(ValueError):
            self.make_policy(epsilon=float("inf"))

    def test_no_available_actions_is_terminal_error(self) -> None:
        policy, _ = self.make_policy()
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            policy.choose_action(FeatureSubsetState((0, 1, 2, 3)))

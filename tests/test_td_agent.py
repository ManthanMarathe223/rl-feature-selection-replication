"""Tests for the Phase 1D TD(0) value update."""

import unittest

from src.rl.state import FeatureSubsetState
from src.rl.td_agent import TD0ValueEstimator


class TD0ValueEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = FeatureSubsetState((0,))
        self.next_state = FeatureSubsetState((0, 1))

    def test_unseen_values_start_at_zero(self) -> None:
        estimator = TD0ValueEstimator(alpha=0.5, gamma=0.7)
        self.assertEqual(estimator.get_value(self.current), 0.0)
        self.assertEqual(estimator.get_value(self.next_state), 0.0)

    def test_simple_update(self) -> None:
        result = TD0ValueEstimator(alpha=0.5, gamma=0.7).update(
            self.current, 1.0, self.next_state
        )
        self.assertEqual(result.old_value, 0.0)
        self.assertEqual(result.next_state_value, 0.0)
        self.assertEqual(result.td_error, 1.0)
        self.assertEqual(result.new_value, 0.5)

    def test_nonzero_next_state_value(self) -> None:
        estimator = TD0ValueEstimator(alpha=0.5, gamma=0.7)
        estimator.set_value(self.current, 0.2)
        estimator.set_value(self.next_state, 0.8)
        result = estimator.update(self.current, 0.1, self.next_state)
        expected_error = 0.1 + 0.7 * 0.8 - 0.2
        self.assertAlmostEqual(result.td_error, expected_error)
        self.assertAlmostEqual(result.new_value, 0.2 + 0.5 * expected_error)

    def test_negative_reward(self) -> None:
        result = TD0ValueEstimator(alpha=0.5, gamma=0.7).update(
            self.current, -1.0, self.next_state
        )
        self.assertEqual(result.new_value, -0.5)

    def test_zero_reward(self) -> None:
        estimator = TD0ValueEstimator(alpha=0.5, gamma=0.7)
        estimator.set_value(self.current, 0.4)
        result = estimator.update(self.current, 0.0, self.next_state)
        self.assertEqual(result.new_value, 0.2)

    def test_repeated_update(self) -> None:
        estimator = TD0ValueEstimator(alpha=0.5, gamma=0.7)
        first = estimator.update(self.current, 1.0, self.next_state)
        second = estimator.update(self.current, 1.0, self.next_state)
        self.assertEqual(first.new_value, 0.5)
        self.assertEqual(second.old_value, 0.5)
        self.assertEqual(second.new_value, 0.75)

    def test_alpha_must_be_strictly_between_zero_and_one(self) -> None:
        with self.assertRaises(ValueError):
            TD0ValueEstimator(alpha=0.0, gamma=0.7)
        with self.assertRaises(ValueError):
            TD0ValueEstimator(alpha=1.0, gamma=0.7)

    def test_gamma_must_be_between_zero_and_one_inclusive(self) -> None:
        TD0ValueEstimator(alpha=0.5, gamma=0.0)
        TD0ValueEstimator(alpha=0.5, gamma=1.0)
        with self.assertRaises(ValueError):
            TD0ValueEstimator(alpha=0.5, gamma=-0.1)
        with self.assertRaises(ValueError):
            TD0ValueEstimator(alpha=0.5, gamma=1.1)

    def test_same_inputs_are_deterministic(self) -> None:
        first = TD0ValueEstimator(alpha=0.5, gamma=0.7).update(
            self.current, 0.3, self.next_state
        )
        second = TD0ValueEstimator(alpha=0.5, gamma=0.7).update(
            self.current, 0.3, self.next_state
        )
        self.assertEqual(first, second)

    def test_state_keys_are_not_mutated(self) -> None:
        estimator = TD0ValueEstimator(alpha=0.5, gamma=0.7)
        original = self.current.selected_features
        estimator.update(self.current, 1.0, self.next_state)
        self.assertEqual(self.current.selected_features, original)
        self.assertIn(self.current, estimator.values)
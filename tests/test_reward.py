"""Tests for pure and transition-level reward calculation."""

import unittest
from dataclasses import dataclass

import numpy as np

from src.rl.reward import (
    calculate_reward,
    evaluate_transition_reward,
    accuracy_difference,
)
from src.rl.state import FeatureSubsetState


@dataclass(frozen=True)
class FakeEvaluation:
    accuracy: float


class FakeEvaluator:
    def __init__(self, accuracies: dict[tuple[int, ...], float]) -> None:
        self.accuracies = accuracies
        self.calls: list[tuple[int, ...]] = []

    def evaluate(self, X: object, y: object, selected_features: tuple[int, ...]) -> FakeEvaluation:
        self.calls.append(selected_features)
        return FakeEvaluation(self.accuracies[selected_features])


class RewardTests(unittest.TestCase):
    def test_positive_reward(self) -> None:
        self.assertAlmostEqual(calculate_reward(0.70, 0.80), 0.10)

    def test_negative_reward(self) -> None:
        self.assertAlmostEqual(calculate_reward(0.80, 0.70), -0.10)

    def test_equal_accuracy(self) -> None:
        self.assertEqual(calculate_reward(0.70, 0.70), 0.0)

    def test_nan_and_infinity_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_reward(np.nan, 0.8)
        with self.assertRaises(ValueError):
            calculate_reward(0.7, np.inf)

    def test_return_value_is_float(self) -> None:
        self.assertIsInstance(calculate_reward(0.7, 0.8), float)

    def test_transition_evaluation_calculates_reward(self) -> None:
        current = FeatureSubsetState((0,))
        next_state = FeatureSubsetState((0, 1))
        evaluator = FakeEvaluator({(0,): 0.70, (0, 1): 0.80})
        result = evaluate_transition_reward(evaluator, "X", "y", current, next_state)
        self.assertEqual(result.current_accuracy, 0.70)
        self.assertEqual(result.next_accuracy, 0.80)
        self.assertAlmostEqual(result.reward, 0.10)
        self.assertEqual(result.current_feature_count, 1)
        self.assertEqual(result.next_feature_count, 2)

    def test_transition_evaluates_states_independently(self) -> None:
        current = FeatureSubsetState((0,))
        next_state = FeatureSubsetState((0, 1))
        evaluator = FakeEvaluator({(0,): 0.80, (0, 1): 0.70})
        result = evaluate_transition_reward(evaluator, None, None, current, next_state)
        self.assertEqual(evaluator.calls, [(0,), (0, 1)])
        self.assertLess(result.reward, 0.0)

    def test_legacy_callback_wrapper(self) -> None:
        current = FeatureSubsetState((0,))
        next_state = FeatureSubsetState((0, 1))
        scores = {current: 0.60, next_state: 0.75}
        self.assertAlmostEqual(
            accuracy_difference(current, next_state, scores.__getitem__), 0.15
        )

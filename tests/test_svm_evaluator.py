"""Unit tests for the Phase 1A evaluation engine."""

import unittest

import numpy as np

from src.evaluation.evaluation_config import EvaluationConfig
from src.evaluation.svm_evaluator import SVMEvaluator


class SVMEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [
                [0.0, 0.1, 1.0],
                [0.1, 0.2, 1.0],
                [1.0, 0.9, 0.0],
                [0.9, 1.0, 0.0],
                [0.2, 0.1, 1.0],
                [0.8, 0.9, 0.0],
            ]
        )
        self.targets = np.array([0, 0, 1, 1, 0, 1])

    def test_valid_feature_subset(self) -> None:
        result = SVMEvaluator(
            EvaluationConfig(test_size=0.5, random_state=7)
        ).evaluate(self.features, self.targets, [0, 2])
        self.assertEqual(result.number_of_selected_features, 2)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)

    def test_invalid_feature_index(self) -> None:
        with self.assertRaises(IndexError):
            SVMEvaluator().evaluate(self.features, self.targets, [3])

    def test_empty_feature_subset(self) -> None:
        with self.assertRaises(ValueError):
            SVMEvaluator().evaluate(self.features, self.targets, [])

    def test_fixed_seed_is_deterministic(self) -> None:
        config = EvaluationConfig(test_size=0.5, random_state=11)
        first = SVMEvaluator(config).evaluate(self.features, self.targets, [0, 1])
        second = SVMEvaluator(config).evaluate(self.features, self.targets, [0, 1])
        self.assertEqual(first.accuracy, second.accuracy)

    def test_cross_validation_uses_accuracy(self) -> None:
        config = EvaluationConfig(
            evaluation_strategy="cross_validation",
            cv_folds=3,
        )
        result = SVMEvaluator(config).evaluate(self.features, self.targets, [0, 1])
        self.assertEqual(result.evaluation_strategy, "cross_validation")
        self.assertEqual(result.model_configuration["scoring_metric"], "accuracy")


class EvaluationConfigTests(unittest.TestCase):
    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationConfig(C=0)
        with self.assertRaises(ValueError):
            EvaluationConfig(scoring_metric="balanced_accuracy")
        with self.assertRaises(ValueError):
            EvaluationConfig(evaluation_strategy="unknown")  # type: ignore[arg-type]

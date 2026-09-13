"""Tests for the minimal feature-selection environment."""

import unittest

from src.rl.environment import FeatureSelectionEnvironment
from src.rl.state import FeatureSubsetState


class FeatureSelectionEnvironmentTests(unittest.TestCase):
    def test_reset_and_step(self) -> None:
        environment = FeatureSelectionEnvironment(5)
        self.assertEqual(environment.available_actions(), (0, 1, 2, 3, 4))
        state = environment.reset(FeatureSubsetState((0, 2)))
        self.assertEqual(state.selected_features, (0, 2))
        self.assertEqual(environment.step(4).selected_features, (0, 2, 4))

    def test_terminal_state_and_no_step_after_terminal(self) -> None:
        environment = FeatureSelectionEnvironment(2)
        environment.step(1)
        environment.step(0)
        self.assertTrue(environment.terminated)
        self.assertEqual(environment.available_actions(), ())
        with self.assertRaises(RuntimeError):
            environment.step(0)

    def test_invalid_index_and_duplicate_are_rejected(self) -> None:
        environment = FeatureSelectionEnvironment(3)
        with self.assertRaises(ValueError):
            environment.step(3)
        environment.step(1)
        with self.assertRaisesRegex(ValueError, "already selected"):
            environment.step(1)

    def test_identical_environments_behave_identically(self) -> None:
        first = FeatureSelectionEnvironment(4, FeatureSubsetState((2, 0)))
        second = FeatureSelectionEnvironment(4, FeatureSubsetState((0, 2)))
        self.assertEqual(first.available_actions(), second.available_actions())
        self.assertEqual(first.step(3), second.step(3))
"""Tests for feature-selection actions and transitions."""

import unittest

from src.rl.actions import available_actions, transition, validate_action
from src.rl.state import FeatureSubsetState


class ActionTests(unittest.TestCase):
    def test_empty_state_actions(self) -> None:
        self.assertEqual(available_actions(FeatureSubsetState(), 4), (0, 1, 2, 3))

    def test_selected_features_are_excluded(self) -> None:
        self.assertEqual(available_actions(FeatureSubsetState((0, 2)), 5), (1, 3, 4))

    def test_transition_adds_one_feature(self) -> None:
        self.assertEqual(
            transition(FeatureSubsetState((0, 2)), 4, 5).selected_features,
            (0, 2, 4),
        )

    def test_duplicate_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "already selected"):
            validate_action(FeatureSubsetState((0, 2)), 2, 5)

    def test_invalid_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_action(FeatureSubsetState(), 5, 5)
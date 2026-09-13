"""Tests for immutable feature-subset states."""

import unittest

from src.rl.state import FeatureSubsetState


class FeatureSubsetStateTests(unittest.TestCase):
    def test_empty_state_and_properties(self) -> None:
        state = FeatureSubsetState()
        self.assertEqual(state.selected_features, ())
        self.assertEqual(state.number_selected, 0)
        self.assertFalse(state.contains(0))

    def test_add_preserves_order_and_original(self) -> None:
        state = FeatureSubsetState((2, 0))
        next_state = state.add(4)
        self.assertEqual(state.selected_features, (0, 2))
        self.assertEqual(next_state.selected_features, (0, 2, 4))

    def test_duplicate_constructor_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FeatureSubsetState((0, 0))

    def test_terminal_state(self) -> None:
        self.assertFalse(FeatureSubsetState((0, 2)).is_terminal(3))
        self.assertTrue(FeatureSubsetState((0, 1, 2)).is_terminal(3))

    def test_hashing_and_equality_are_deterministic(self) -> None:
        first = FeatureSubsetState((2, 0))
        second = FeatureSubsetState((0, 2))
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual({first: "state"}[second], "state")
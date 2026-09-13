"""Tests for the conceptual reward."""

import unittest

from src.rl.reward import accuracy_difference
from src.rl.state import FeatureSubsetState


class RewardTests(unittest.TestCase):
    def test_accuracy_difference(self) -> None:
        current = FeatureSubsetState((0,))
        next_state = FeatureSubsetState((0, 1))
        scores = {current: 0.60, next_state: 0.75}
        self.assertAlmostEqual(
            accuracy_difference(current, next_state, scores.__getitem__), 0.15
        )

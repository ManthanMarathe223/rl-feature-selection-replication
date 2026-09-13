"""Tests for incremental Average of Reward updates."""

import unittest

from src.rl.aor import AverageReward


class AverageRewardTests(unittest.TestCase):
    def test_incremental_average(self) -> None:
        average = AverageReward()
        self.assertEqual(average.value, 0.0)
        average.update(0.2)
        self.assertAlmostEqual(average.update(0.8), 0.5)
        self.assertEqual(average.count, 2)

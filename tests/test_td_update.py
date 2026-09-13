"""Tests for the TD(0) update."""

import unittest

from src.rl.state import FeatureSubsetState
from src.rl.td_agent import TDAgent


class TDUpdateTests(unittest.TestCase):
    def test_update_equation(self) -> None:
        state = FeatureSubsetState((0,))
        next_state = FeatureSubsetState((0, 1))
        agent = TDAgent(alpha=0.5, gamma=0.9, values={next_state: 0.8})
        self.assertAlmostEqual(agent.update(state, 0.2, next_state).new_value, 0.46)

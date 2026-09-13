"""Tests for independent per-feature AOR tracking."""

import unittest

from src.rl.aor import AORTracker


class AORTrackerTests(unittest.TestCase):
    def test_first_observation(self) -> None:
        tracker = AORTracker(5)
        tracker.record(2, 0.04)
        self.assertEqual(tracker.get_count(2), 1)
        self.assertAlmostEqual(tracker.get_aor(2), 0.04)

    def test_two_observations(self) -> None:
        tracker = AORTracker(5)
        tracker.record(2, 0.04)
        tracker.record(2, 0.02)
        self.assertAlmostEqual(tracker.get_aor(2), 0.03)

    def test_three_observations(self) -> None:
        tracker = AORTracker(5)
        for contribution in (0.04, 0.02, -0.01):
            tracker.record(2, contribution)
        self.assertAlmostEqual(tracker.get_aor(2), 0.016666666666666666)

    def test_repeated_observations_equal_arithmetic_mean(self) -> None:
        tracker = AORTracker(1)
        contributions = (0.1, 0.3, 0.2, 0.8)
        for contribution in contributions:
            tracker.update(0, contribution)
        self.assertAlmostEqual(tracker.get_aor(0), sum(contributions) / len(contributions))
        self.assertEqual(tracker.get_count(0), len(contributions))

    def test_negative_and_zero_contributions(self) -> None:
        tracker = AORTracker(2)
        tracker.record(0, -0.5)
        tracker.record(1, 0.0)
        self.assertEqual(tracker.get_aor(0), -0.5)
        self.assertEqual(tracker.get_aor(1), 0.0)

    def test_invalid_feature_index(self) -> None:
        tracker = AORTracker(2)
        with self.assertRaises(ValueError):
            tracker.record(2, 0.1)
        with self.assertRaises(ValueError):
            tracker.record(-1, 0.1)

    def test_non_integer_feature_index(self) -> None:
        with self.assertRaises(TypeError):
            AORTracker(2).record(1.0, 0.1)  # type: ignore[arg-type]

    def test_nonfinite_contribution(self) -> None:
        tracker = AORTracker(2)
        with self.assertRaises(ValueError):
            tracker.record(0, float("nan"))
        with self.assertRaises(ValueError):
            tracker.record(0, float("inf"))

    def test_reset_clears_all_scores(self) -> None:
        tracker = AORTracker(3)
        tracker.record(0, 0.2)
        tracker.record(2, -0.4)
        tracker.reset()
        self.assertEqual(tracker.all_scores(), {0: 0.0, 1: 0.0, 2: 0.0})
        self.assertEqual(tracker.get_count(0), 0)
        self.assertEqual(tracker.get_count(2), 0)

    def test_ranking_is_deterministic(self) -> None:
        tracker = AORTracker(3)
        tracker.record(0, 0.1)
        tracker.record(1, 0.3)
        tracker.record(2, 0.2)
        self.assertEqual(tracker.ranking(), [1, 2, 0])

    def test_equal_scores_use_feature_index_tiebreak(self) -> None:
        tracker = AORTracker(4)
        tracker.record(3, 0.5)
        tracker.record(1, 0.5)
        tracker.record(2, 0.5)
        self.assertEqual(tracker.ranking(), [1, 2, 3, 0])

"""Integration tests for the Australian single-episode smoke run."""

import unittest

from experiments.baseline.run_single_episode import run_single_episode


class SingleEpisodeIntegrationTests(unittest.TestCase):
    def test_episode_trace_is_complete_and_consistent(self) -> None:
        trace = run_single_episode(write_outputs=False)
        self.assertGreater(trace["number_of_steps"], 0)
        self.assertEqual(len(trace["steps"]), trace["number_of_steps"])
        for step in trace["steps"]:
            self.assertIn(step["action"], range(trace["number_of_features"]))
            self.assertEqual(
                len(step["next_state"]), len(step["current_state"]) + 1
            )
            self.assertAlmostEqual(
                step["reward"],
                step["next_accuracy"] - step["current_accuracy"],
            )
            self.assertIn(step["selection_mode"], {"exploration", "exploitation"})
            self.assertEqual(step["selected_feature_count"], 1)
        self.assertEqual(
            trace["final_state"], trace["steps"][-1]["next_state"]
        )
        self.assertEqual(
            trace["final_state"],
            sorted(set(trace["initial_state"] + [step["action"] for step in trace["steps"]])),
        )

    def test_td_and_aor_update_once_per_transition(self) -> None:
        trace = run_single_episode(write_outputs=False)
        self.assertEqual(len(trace["steps"]), trace["number_of_steps"])
        for step in trace["steps"]:
            self.assertIsInstance(step["td_error"], float)
            self.assertIsInstance(step["selected_feature_aor"], float)

    def test_same_seed_reproduces_trajectory(self) -> None:
        first = run_single_episode(seed=2021, write_outputs=False)
        second = run_single_episode(seed=2021, write_outputs=False)
        self.assertEqual(first, second)

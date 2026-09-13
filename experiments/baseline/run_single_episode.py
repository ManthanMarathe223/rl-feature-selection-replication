"""Run one small Australian-only end-to-end smoke episode."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.data.inspect_datasets import PROJECT_ROOT, dataset_specs, load_dataset
from src.evaluation.evaluation_config import EvaluationConfig
from src.evaluation.svm_evaluator import SVMEvaluator
from src.rl.aor import AORTracker
from src.rl.environment import FeatureSelectionEnvironment
from src.rl.episode import EpisodeResult, FeatureSelectionEpisodeRunner
from src.rl.policy import FeatureSelectionPolicy
from src.rl.reward import evaluate_transition_reward
from src.rl.td_agent import TD0ValueEstimator, TD0UpdateResult
from src.rl.state import FeatureSubsetState


SEED = 2021
SMOKE_FEATURE_COUNT = 4
MAXIMUM_EPISODE_STEPS = 3
ALPHA = 0.5
GAMMA = 0.7
EPSILON = 0.0

JSON_OUTPUT = PROJECT_ROOT / "results" / "raw" / "single_episode_trace.json"
TEXT_OUTPUT = PROJECT_ROOT / "results" / "single_episode_trace.txt"


class RecordingTD0ValueEstimator(TD0ValueEstimator):
    """TD estimator that exposes update details for the smoke trace."""

    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__(alpha=alpha, gamma=gamma)
        self.update_results: list[TD0UpdateResult] = []

    def update(self, current_state, reward, next_state):
        result = super().update(current_state, reward, next_state)
        self.update_results.append(result)
        return result


class RecordingAORTracker(AORTracker):
    """AOR tracker that exposes selected-feature updates for the smoke trace."""

    def __init__(self, number_of_features: int) -> None:
        super().__init__(number_of_features)
        self.recorded_updates: list[tuple[int, float, float, int]] = []

    def record(self, feature_index: int, contribution: float) -> float:
        score = super().record(feature_index, contribution)
        self.recorded_updates.append(
            (feature_index, float(contribution), score, self.get_count(feature_index))
        )
        return score


def _build_runner(seed: int) -> tuple[FeatureSelectionEpisodeRunner, RecordingTD0ValueEstimator, RecordingAORTracker]:
    inspection = load_dataset(dataset_specs()[0])
    feature_columns = inspection.spec.feature_columns[:SMOKE_FEATURE_COUNT]
    X = inspection.frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y = inspection.frame.loc[:, inspection.spec.target_column].to_numpy()

    evaluator = SVMEvaluator(
        EvaluationConfig(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            scaling="none",
            random_state=seed,
            evaluation_strategy="holdout",
            scoring_metric="accuracy",
            test_size=0.2,
            missing_value_policy="error",
        )
    )
    environment = FeatureSelectionEnvironment(SMOKE_FEATURE_COUNT)
    aor_tracker = RecordingAORTracker(SMOKE_FEATURE_COUNT)
    policy = FeatureSelectionPolicy(
        SMOKE_FEATURE_COUNT,
        aor_tracker,
        epsilon=EPSILON,
        random_seed=seed,
    )
    td_estimator = RecordingTD0ValueEstimator(ALPHA, GAMMA)
    runner = FeatureSelectionEpisodeRunner(
        feature_count=SMOKE_FEATURE_COUNT,
        environment=environment,
        evaluator=evaluator,
        reward_calculator=evaluate_transition_reward,
        td_estimator=td_estimator,
        aor_tracker=aor_tracker,
        policy=policy,
        random_generator=random.Random(seed),
        maximum_episode_steps=MAXIMUM_EPISODE_STEPS,
        X=X,
        y=y,
        minimum_initial_subset_size=1,
        maximum_initial_subset_size=1,
        aor_contribution_mode="reward",
    )
    return runner, td_estimator, aor_tracker


def _state_list(state: FeatureSubsetState) -> list[int]:
    return list(state.selected_features)


def build_trace(seed: int = SEED) -> dict[str, Any]:
    """Run the smoke episode and return a JSON-serializable full trace."""
    runner, td_estimator, aor_tracker = _build_runner(seed)
    result: EpisodeResult = runner.run_episode()
    steps: list[dict[str, Any]] = []
    for index, (current_state, action, next_state) in enumerate(
        zip(result.states, result.actions, result.states[1:])
    ):
        td_result = td_estimator.update_results[index]
        aor_feature, _, aor_score, aor_count = aor_tracker.recorded_updates[index]
        current_accuracy = result.accuracies[index * 2]
        next_accuracy = result.accuracies[index * 2 + 1]
        steps.append(
            {
                "step_number": index + 1,
                "current_state": _state_list(current_state),
                "action": action,
                "next_state": _state_list(next_state),
                "current_accuracy": current_accuracy,
                "next_accuracy": next_accuracy,
                "reward": result.rewards[index],
                "old_td_value": td_result.old_value,
                "next_state_td_value": td_result.next_state_value,
                "td_error": td_result.td_error,
                "new_td_value": td_result.new_value,
                "selected_feature_aor": aor_score,
                "selected_feature_count": aor_count,
                "selection_mode": "exploration" if index == 0 else "exploitation",
                "aor_feature": aor_feature,
            }
        )
    return {
        "dataset": "Australian Credit Approval",
        "number_of_features": SMOKE_FEATURE_COUNT,
        "feature_columns": [f"feature_{index}" for index in range(1, SMOKE_FEATURE_COUNT + 1)],
        "random_seed": seed,
        "reconstruction_parameters": {
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "svm_kernel": "rbf",
            "svm_C": 1.0,
            "svm_gamma": "scale",
            "scaling": "none",
            "evaluation_strategy": "holdout",
            "test_size": 0.2,
            "initial_subset_size": 1,
            "maximum_episode_steps": MAXIMUM_EPISODE_STEPS,
            "aor_contribution_mode": "reward",
        },
        "initial_state": _state_list(result.initial_state),
        "steps": steps,
        "final_state": _state_list(result.final_state),
        "number_of_steps": result.number_of_steps,
        "termination_reason": result.termination_reason,
    }


def _write_outputs(trace: dict[str, Any]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    lines = [
        "SINGLE EPISODE TRACE",
        f"Dataset: {trace['dataset']}",
        f"Number of features: {trace['number_of_features']}",
        f"Random seed: {trace['random_seed']}",
        f"Initial state: {trace['initial_state']}",
        "",
    ]
    for step in trace["steps"]:
        lines.extend(
            [
                f"Step {step['step_number']} ({step['selection_mode']}):",
                f"  Current state: {step['current_state']}",
                f"  Action: {step['action']}",
                f"  Next state: {step['next_state']}",
                f"  Current accuracy: {step['current_accuracy']}",
                f"  Next accuracy: {step['next_accuracy']}",
                f"  Reward: {step['reward']}",
                f"  Old TD value: {step['old_td_value']}",
                f"  Next-state TD value: {step['next_state_td_value']}",
                f"  TD error: {step['td_error']}",
                f"  New TD value: {step['new_td_value']}",
                f"  Selected feature AOR: {step['selected_feature_aor']}",
                f"  Selected feature count: {step['selected_feature_count']}",
                "",
            ]
        )
    lines.extend(
        [
            f"Final state: {trace['final_state']}",
            f"Number of steps: {trace['number_of_steps']}",
            f"Termination reason: {trace['termination_reason']}",
        ]
    )
    TEXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_episode(seed: int = SEED, write_outputs: bool = True) -> dict[str, Any]:
    """Run one deterministic smoke episode and optionally save its trace."""
    trace = build_trace(seed)
    if write_outputs:
        _write_outputs(trace)
    return trace


def main() -> None:
    trace = run_single_episode()
    print(f"Dataset: {trace['dataset']}")
    print(f"Number of features: {trace['number_of_features']}")
    print(f"Random seed: {trace['random_seed']}")
    print(f"Initial state: {trace['initial_state']}")
    for step in trace["steps"]:
        print(f"\nStep {step['step_number']} ({step['selection_mode']})")
        for key, value in step.items():
            if key not in {"step_number", "selection_mode", "aor_feature"}:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    print(f"\nFinal state: {trace['final_state']}")
    print(f"Number of steps: {trace['number_of_steps']}")
    print(f"Termination reason: {trace['termination_reason']}")
    print(f"Saved JSON trace: {JSON_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Saved text trace: {TEXT_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

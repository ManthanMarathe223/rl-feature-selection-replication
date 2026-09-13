"""Run the Phase 1 replication protocol for WPBC and Sonar."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.data.inspect_datasets import PROJECT_ROOT, dataset_specs, load_dataset
from src.evaluation.evaluation_config import EvaluationConfig
from src.evaluation.svm_evaluator import SVMEvaluator
from src.rl.aor import AORTracker
from src.rl.environment import FeatureSelectionEnvironment
from src.rl.episode import FeatureSelectionEpisodeRunner
from src.rl.policy import FeatureSelectionPolicy
from src.rl.reward import evaluate_transition_reward
from src.rl.state import FeatureSubsetState
from src.rl.td_agent import TD0ValueEstimator


EPSILON_VALUES = (0.3, 0.4, 0.5, 0.6)
SEEDS = (2021, 2022, 2023, 2024, 2025)
ALPHA = 0.5
GAMMA = 0.7
EPISODES = 100
INITIAL_SUBSET_SIZE = 1
DATASET_SETTINGS = {
    "WPBC": {
        "spec_index": 1,
        "paper_accuracy": 0.7629,
        "paper_std": 0.00007,
        "feature_policy": "32 numeric variables; identifier/time excluded",
        "missing_value_policy": "mean_impute",
    },
    "Sonar": {
        "spec_index": 2,
        "paper_accuracy": 0.7369,
        "paper_std": 0.00108,
        "feature_policy": "60 numeric variables",
        "missing_value_policy": "error",
    },
}


class RecordingTD0ValueEstimator(TD0ValueEstimator):
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__(alpha=alpha, gamma=gamma)
        self.update_results = []

    def update(self, current_state, reward, next_state):
        result = super().update(current_state, reward, next_state)
        self.update_results.append(result)
        return result


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _state_text(state: FeatureSubsetState | tuple[int, ...]) -> str:
    selected = state.selected_features if isinstance(state, FeatureSubsetState) else state
    return json.dumps(list(selected), separators=(",", ":"))


def _paths(dataset_key: str) -> tuple[Path, Path, Path, Path]:
    name = dataset_key.lower()
    return (
        PROJECT_ROOT / "results" / "raw" / f"{name}_episode_results.csv",
        PROJECT_ROOT / "results" / "tables" / f"{name}_replication_results.csv",
        PROJECT_ROOT / "results" / "tables" / f"{name}_paper_comparison.csv",
        PROJECT_ROOT / "results" / "figures",
    )


def _build_runner(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    seed: int,
    missing_value_policy: str,
) -> tuple[FeatureSelectionEpisodeRunner, RecordingTD0ValueEstimator]:
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
            missing_value_policy=missing_value_policy,
        )
    )
    aor_tracker = AORTracker(X.shape[1])
    policy = FeatureSelectionPolicy(X.shape[1], aor_tracker, epsilon, random_seed=seed)
    td_estimator = RecordingTD0ValueEstimator(ALPHA, GAMMA)
    runner = FeatureSelectionEpisodeRunner(
        feature_count=X.shape[1],
        environment=FeatureSelectionEnvironment(X.shape[1]),
        evaluator=evaluator,
        reward_calculator=evaluate_transition_reward,
        td_estimator=td_estimator,
        aor_tracker=aor_tracker,
        policy=policy,
        random_generator=random.Random(seed),
        maximum_episode_steps=X.shape[1] - INITIAL_SUBSET_SIZE,
        X=X,
        y=y,
        minimum_initial_subset_size=INITIAL_SUBSET_SIZE,
        maximum_initial_subset_size=INITIAL_SUBSET_SIZE,
        aor_contribution_mode="reward",
    )
    return runner, td_estimator


def _run_configuration(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    seed: int,
    missing_value_policy: str,
    code_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    runner, td_estimator = _build_runner(X, y, epsilon, seed, missing_value_policy)
    rows: list[dict[str, Any]] = []
    visited_states: set[FeatureSubsetState] = set()
    rewards: list[float] = []
    best_accuracy = float("-inf")
    best_state: tuple[int, ...] = ()
    best_episode = 0
    final_accuracy = float("nan")
    timestamp = datetime.now(timezone.utc).isoformat()

    for episode_number in range(1, EPISODES + 1):
        td_start = len(td_estimator.update_results)
        result = runner.run_episode()
        visited_states.update(result.states)
        rewards.extend(result.rewards)
        final_accuracy = result.accuracies[-1]
        for step_index, (current_state, action, next_state) in enumerate(
            zip(result.states, result.actions, result.states[1:])
        ):
            td_result = td_estimator.update_results[td_start + step_index]
            current_accuracy = result.accuracies[step_index * 2]
            next_accuracy = result.accuracies[step_index * 2 + 1]
            for state, accuracy, state_value, reward in (
                (current_state, current_accuracy, td_result.old_value, result.rewards[step_index]),
                (next_state, next_accuracy, td_result.new_value, ""),
            ):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "epsilon": epsilon,
                        "alpha": ALPHA,
                        "gamma": GAMMA,
                        "episode_number": episode_number,
                        "state": _state_text(state),
                        "selected_features": _state_text(state),
                        "accuracy": accuracy,
                        "reward": reward,
                        "state_value": state_value,
                        "action": action,
                        "episode_states_visited": len(visited_states),
                        "code_version": code_version,
                        "timestamp_utc": timestamp,
                    }
                )
            if next_accuracy > best_accuracy:
                best_accuracy = next_accuracy
                best_state = next_state.selected_features
                best_episode = episode_number

    runtime = time.perf_counter() - started
    summary = {
        "dataset": dataset_name,
        "seed": seed,
        "epsilon": epsilon,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "episode_count": EPISODES,
        "initial_subset_size": INITIAL_SUBSET_SIZE,
        "final_selected_features": _state_text(result.final_state),
        "final_number_of_selected_features": len(result.final_state.selected_features),
        "best_observed_state": _state_text(best_state),
        "best_observed_feature_count": len(best_state),
        "best_observed_accuracy": best_accuracy,
        "episode_of_best_state": best_episode,
        "final_state_accuracy": final_accuracy,
        "runtime_seconds": runtime,
        "states_visited": len(visited_states),
        "total_transitions": sum(1 for row in rows if row["reward"] != ""),
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "maximum_state_value": max((u.new_value for u in td_estimator.update_results), default=0.0),
        "termination_reason": result.termination_reason,
        "code_version": code_version,
        "timestamp_utc": timestamp,
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(dataset_key: str, rows: list[dict[str, Any]]) -> None:
    _, _, _, figures_path = _paths(dataset_key)
    figures_path.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[float(row["epsilon"])][int(row["seed"])].append(row)

    def series(epsilon: float, field: str, reducer=max) -> list[float]:
        values = []
        for episode in range(1, EPISODES + 1):
            per_seed = []
            for seed_rows in grouped[epsilon].values():
                selected = [float(row[field]) for row in seed_rows if int(row["episode_number"]) == episode]
                if selected:
                    per_seed.append(reducer(selected))
            values.append(float(np.mean(per_seed)) if per_seed else float("nan"))
        return values

    prefix = dataset_key.lower()
    for epsilon in EPSILON_VALUES:
        plt.figure()
        plt.plot(range(1, EPISODES + 1), series(epsilon, "state_value"))
        plt.xlabel("Episode")
        plt.ylabel("Maximum state value")
        plt.title(f"{dataset_key}: Maximum State Value (epsilon={epsilon})")
        plt.tight_layout()
        plt.savefig(figures_path / f"{prefix}_max_state_value_epsilon_{epsilon}.png")
        plt.close()

    for field, title, filename, reducer in (
        ("episode_states_visited", "Number of Visited States", "visited_states_vs_episode", max),
        ("accuracy", "Best Accuracy", "best_accuracy_vs_episode", max),
    ):
        plt.figure()
        for epsilon in EPSILON_VALUES:
            plt.plot(range(1, EPISODES + 1), series(epsilon, field, reducer), label=f"epsilon={epsilon}")
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.title(f"{dataset_key}: {title} vs Episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_path / f"{prefix}_{filename}.png")
        plt.close()

    plt.figure()
    for epsilon in EPSILON_VALUES:
        selected = [row for row in rows if float(row["epsilon"]) == epsilon and row["reward"] != ""]
        plt.scatter(
            [len(json.loads(row["selected_features"])) for row in selected],
            [float(row["accuracy"]) for row in selected],
            label=f"epsilon={epsilon}",
            alpha=0.5,
        )
    plt.xlabel("Number of selected features")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_key}: Accuracy vs Number of Selected Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_path / f"{prefix}_accuracy_vs_features.png")
    plt.close()

    distribution: list[list[float]] = [[] for _ in EPSILON_VALUES]
    for index, epsilon in enumerate(EPSILON_VALUES):
        for seed_rows in grouped[epsilon].values():
            final = [row for row in seed_rows if int(row["episode_number"]) == EPISODES]
            if final:
                distribution[index].append(float(final[-1]["accuracy"]))
    plt.figure()
    plt.boxplot(distribution, tick_labels=[str(epsilon) for epsilon in EPSILON_VALUES])
    plt.xlabel("Epsilon")
    plt.ylabel("Final accuracy")
    plt.title(f"{dataset_key}: Final Accuracy Distribution")
    plt.tight_layout()
    plt.savefig(figures_path / f"{prefix}_final_accuracy_distribution.png")
    plt.close()


def run_dataset(dataset_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one configured remaining dataset across 20 experiments."""
    settings = DATASET_SETTINGS[dataset_key]
    inspection = load_dataset(dataset_specs()[settings["spec_index"]])
    feature_columns = inspection.spec.feature_columns
    X = inspection.frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y = inspection.frame.loc[:, inspection.spec.target_column].to_numpy()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    code_version = _code_version()
    for epsilon in EPSILON_VALUES:
        for seed in SEEDS:
            summary, episode_rows = _run_configuration(
                dataset_key,
                X,
                y,
                epsilon,
                seed,
                settings["missing_value_policy"],
                code_version,
            )
            summaries.append(summary)
            rows.extend(episode_rows)
            print(f"completed dataset={dataset_key}, epsilon={epsilon}, seed={seed}")

    raw_path, summary_path, comparison_path, _ = _paths(dataset_key)
    _write_csv(raw_path, rows)
    comparison_rows = []
    for epsilon in EPSILON_VALUES:
        runs = [row for row in summaries if row["epsilon"] == epsilon]
        accuracies = np.array([row["best_observed_accuracy"] for row in runs])
        counts = np.array([row["best_observed_feature_count"] for row in runs])
        runtimes = np.array([row["runtime_seconds"] for row in runs])
        best_run = max(runs, key=lambda row: row["best_observed_accuracy"])
        worst_run = min(runs, key=lambda row: row["best_observed_accuracy"])
        for row in runs:
            row.update(
                {
                    "mean_best_accuracy": float(np.mean(accuracies)),
                    "std_best_accuracy": float(np.std(accuracies)),
                    "mean_selected_features": float(np.mean(counts)),
                    "std_selected_features": float(np.std(counts)),
                    "mean_runtime_seconds": float(np.mean(runtimes)),
                    "std_runtime_seconds": float(np.std(runtimes)),
                    "best_run_seed": best_run["seed"],
                    "worst_run_seed": worst_run["seed"],
                }
            )
        comparison_rows.append(
            {
                "epsilon": epsilon,
                "our_mean_accuracy": float(np.mean(accuracies)),
                "our_std_accuracy": float(np.std(accuracies)),
                "paper_accuracy": settings["paper_accuracy"],
                "paper_std_accuracy": settings["paper_std"],
                "accuracy_difference": float(np.mean(accuracies) - settings["paper_accuracy"]),
                "mean_selected_features": float(np.mean(counts)),
                "notes": f"{settings['feature_policy']}; paper comparison is not used for tuning.",
            }
        )
    _write_csv(summary_path, summaries)
    _write_csv(comparison_path, comparison_rows)
    _write_plots(dataset_key, rows)
    return summaries, rows


if __name__ == "__main__":
    run_dataset("WPBC")
    run_dataset("Sonar")

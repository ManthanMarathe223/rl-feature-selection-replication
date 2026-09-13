"""Run the first Australian-only replication experiment."""

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


DATASET_NAME = "Australian Credit Approval"
EPSILON_VALUES = (0.3, 0.4, 0.5, 0.6)
SEEDS = (2021, 2022, 2023, 2024, 2025)
ALPHA = 0.5
GAMMA = 0.7
EPISODES = 100
INITIAL_SUBSET_SIZE = 1
PAPER_ACCURACY = 0.8555
PAPER_STD = 0.00039

RAW_RESULTS_PATH = PROJECT_ROOT / "results" / "raw" / "australian_episode_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "australian_replication_results.csv"
COMPARISON_PATH = PROJECT_ROOT / "results" / "tables" / "australian_paper_comparison.csv"
FIGURES_PATH = PROJECT_ROOT / "results" / "figures"


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


class RecordingTD0ValueEstimator(TD0ValueEstimator):
    """Expose TD update values without changing the TD implementation."""

    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__(alpha=alpha, gamma=gamma)
        self.update_results = []

    def update(self, current_state, reward, next_state):
        result = super().update(current_state, reward, next_state)
        self.update_results.append(result)
        return result


def _state_text(state: FeatureSubsetState | tuple[int, ...]) -> str:
    selected_features = (
        state.selected_features if isinstance(state, FeatureSubsetState) else state
    )
    return json.dumps(list(selected_features), separators=(",", ":"))


def _build_components(X: np.ndarray, y: np.ndarray, epsilon: float, seed: int):
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
    aor_tracker = AORTracker(X.shape[1])
    policy = FeatureSelectionPolicy(
        X.shape[1],
        aor_tracker,
        epsilon=epsilon,
        random_seed=seed,
    )
    td_estimator = RecordingTD0ValueEstimator(alpha=ALPHA, gamma=GAMMA)
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
    return runner, td_estimator, aor_tracker


def _run_configuration(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    seed: int,
    code_version: str,
    timestamp: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    runner, td_estimator, aor_tracker = _build_components(X, y, epsilon, seed)
    episode_rows: list[dict[str, Any]] = []
    best_accuracy = float("-inf")
    best_state: tuple[int, ...] = ()
    best_episode = 0
    final_accuracy = float("nan")
    all_rewards: list[float] = []
    max_state_value = 0.0
    visited_states: set[FeatureSubsetState] = set()

    for episode_number in range(1, EPISODES + 1):
        td_start = len(td_estimator.update_results)
        result = runner.run_episode()
        visited_states.update(result.states)
        all_rewards.extend(result.rewards)
        final_accuracy = result.accuracies[-1]
        for step_index, (current_state, action, next_state) in enumerate(
            zip(result.states, result.actions, result.states[1:])
        ):
            td_result = td_estimator.update_results[td_start + step_index]
            current_accuracy = result.accuracies[step_index * 2]
            next_accuracy = result.accuracies[step_index * 2 + 1]
            current_value = td_result.new_value
            next_value = td_result.next_state_value
            observations = (
                (current_state, current_accuracy, current_value, result.rewards[step_index]),
                (next_state, next_accuracy, next_value, ""),
            )
            for state, accuracy, state_value, reward in observations:
                episode_rows.append(
                    {
                        "dataset": DATASET_NAME,
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

        max_state_value = max(
            max_state_value,
            max(
                (update.new_value for update in td_estimator.update_results[td_start:]),
                default=0.0,
            ),
        )

    runtime = time.perf_counter() - started
    summary = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "epsilon": epsilon,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "episode_count": EPISODES,
        "initial_subset_size": INITIAL_SUBSET_SIZE,
        "final_selected_features": _state_text(result.final_state),
        "final_number_of_selected_features": len(result.final_state.selected_features),
        "best_observed_state": _state_text(best_state),
        "best_observed_accuracy": best_accuracy,
        "episode_of_best_state": best_episode,
        "final_state_accuracy": final_accuracy,
        "runtime_seconds": runtime,
        "states_visited": len(visited_states),
        "total_transitions": sum(
            1 for row in episode_rows if row["reward"] != ""
        ),
        "average_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "maximum_state_value": max_state_value,
        "termination_reason": result.termination_reason,
        "code_version": code_version,
        "timestamp_utc": timestamp,
    }
    return summary, episode_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(episode_rows: list[dict[str, Any]]) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in episode_rows:
        grouped[float(row["epsilon"])][int(row["seed"])].append(row)

    def averaged_series(epsilon: float, value_name: str, maximum: bool = False) -> list[float]:
        by_seed = grouped[epsilon]
        values: list[float] = []
        for episode in range(1, EPISODES + 1):
            episode_values = []
            for rows in by_seed.values():
                selected = [float(row[value_name]) for row in rows if int(row["episode_number"]) == episode]
                if selected:
                    episode_values.append(max(selected) if maximum else selected[-1])
            values.append(float(np.mean(episode_values)) if episode_values else float("nan"))
        return values

    for epsilon in EPSILON_VALUES:
        plt.figure()
        plt.plot(range(1, EPISODES + 1), averaged_series(epsilon, "state_value", maximum=True))
        plt.xlabel("Episode")
        plt.ylabel("Maximum state value")
        plt.title(f"Maximum State Value vs Episode (epsilon={epsilon})")
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / f"australian_max_state_value_epsilon_{epsilon}.png")
        plt.close()

    plt.figure()
    for epsilon in EPSILON_VALUES:
        plt.plot(
            range(1, EPISODES + 1),
            averaged_series(epsilon, "episode_states_visited"),
            label=f"epsilon={epsilon}",
        )
    plt.xlabel("Episode")
    plt.ylabel("Visited states proxy")
    plt.title("Number of Visited States vs Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "australian_visited_states_vs_episode.png")
    plt.close()

    plt.figure()
    for epsilon in EPSILON_VALUES:
        plt.plot(range(1, EPISODES + 1), averaged_series(epsilon, "accuracy", maximum=True), label=f"epsilon={epsilon}")
    plt.xlabel("Episode")
    plt.ylabel("Best accuracy")
    plt.title("Best Accuracy vs Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "australian_best_accuracy_vs_episode.png")
    plt.close()

    plt.figure()
    for epsilon in EPSILON_VALUES:
        rows = [row for row in episode_rows if float(row["epsilon"]) == epsilon and row["reward"] != ""]
        plt.scatter(
            [len(json.loads(row["selected_features"])) for row in rows],
            [float(row["accuracy"]) for row in rows],
            label=f"epsilon={epsilon}",
            alpha=0.5,
        )
    plt.xlabel("Number of selected features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Selected Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "australian_accuracy_vs_features.png")
    plt.close()

    plt.figure()
    distribution: list[list[float]] = [[] for _ in EPSILON_VALUES]
    # Final accuracies are the last accuracy observation of each run.
    for epsilon_index, epsilon in enumerate(EPSILON_VALUES):
        for rows in grouped[epsilon].values():
            final_episode_rows = [
                row for row in rows if int(row["episode_number"]) == EPISODES
            ]
            if final_episode_rows:
                distribution[epsilon_index].append(
                    float(final_episode_rows[-1]["accuracy"])
                )
    plt.boxplot(distribution, tick_labels=[str(epsilon) for epsilon in EPSILON_VALUES])
    plt.xlabel("Epsilon")
    plt.ylabel("Final accuracy")
    plt.title("Final Accuracy Distribution Across Seeds")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "australian_final_accuracy_distribution.png")
    plt.close()


def run_experiment() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all four epsilon values across the five project-selected seeds."""
    inspection = load_dataset(dataset_specs()[0])
    feature_columns = inspection.spec.feature_columns
    X = inspection.frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y = inspection.frame.loc[:, inspection.spec.target_column].to_numpy()
    code_version = _code_version()
    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for epsilon in EPSILON_VALUES:
        for seed in SEEDS:
            timestamp = datetime.now(timezone.utc).isoformat()
            summary, rows = _run_configuration(X, y, epsilon, seed, code_version, timestamp)
            summaries.append(summary)
            episode_rows.extend(rows)
            print(f"completed epsilon={epsilon}, seed={seed}")

    _write_csv(RAW_RESULTS_PATH, episode_rows)
    _write_csv(SUMMARY_PATH, summaries)
    comparison_rows: list[dict[str, Any]] = []
    for epsilon in EPSILON_VALUES:
        runs = [row for row in summaries if row["epsilon"] == epsilon]
        accuracies = np.array([row["best_observed_accuracy"] for row in runs], dtype=float)
        selected_counts = np.array([row["final_number_of_selected_features"] for row in runs], dtype=float)
        runtimes = np.array([row["runtime_seconds"] for row in runs], dtype=float)
        best_run = max(runs, key=lambda row: row["best_observed_accuracy"])
        worst_run = min(runs, key=lambda row: row["best_observed_accuracy"])
        for row in runs:
            row.update(
                {
                    "mean_best_accuracy": float(np.mean(accuracies)),
                    "std_best_accuracy": float(np.std(accuracies)),
                    "mean_selected_features": float(np.mean(selected_counts)),
                    "std_selected_features": float(np.std(selected_counts)),
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
                "paper_accuracy": PAPER_ACCURACY,
                "paper_std_accuracy": PAPER_STD,
                "accuracy_difference": float(np.mean(accuracies) - PAPER_ACCURACY),
                "mean_selected_features": float(np.mean(selected_counts)),
                "notes": "Paper target converted from 85.55 +/- 0.039 percent; not used for tuning.",
            }
        )
    _write_csv(SUMMARY_PATH, summaries)
    _write_csv(COMPARISON_PATH, comparison_rows)
    _write_plots(episode_rows)
    return summaries, episode_rows


if __name__ == "__main__":
    run_experiment()

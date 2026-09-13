"""Accuracy-difference rewards for feature-selection transitions."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from src.rl.state import FeatureSubsetState


@dataclass(frozen=True)
class TransitionReward:
    """Evaluation results for two states and their accuracy difference."""

    current_accuracy: float
    next_accuracy: float
    reward: float
    current_feature_count: int
    next_feature_count: int


def _validate_accuracy(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric accuracy value.")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def calculate_reward(current_accuracy: Real, next_accuracy: Real) -> float:
    """Return the paper's reward: next accuracy minus current accuracy."""
    current = _validate_accuracy(current_accuracy, "current_accuracy")
    next_value = _validate_accuracy(next_accuracy, "next_accuracy")
    return float(next_value - current)


def evaluate_transition_reward(
    evaluator: Any,
    X: Any,
    y: Any,
    current_state: FeatureSubsetState,
    next_state: FeatureSubsetState,
) -> TransitionReward:
    """Evaluate two states independently and calculate their accuracy difference."""
    current_result = evaluator.evaluate(X, y, current_state.selected_features)
    next_result = evaluator.evaluate(X, y, next_state.selected_features)
    current_accuracy = _validate_accuracy(
        current_result.accuracy, "current_accuracy"
    )
    next_accuracy = _validate_accuracy(next_result.accuracy, "next_accuracy")
    return TransitionReward(
        current_accuracy=current_accuracy,
        next_accuracy=next_accuracy,
        reward=calculate_reward(current_accuracy, next_accuracy),
        current_feature_count=current_state.number_selected,
        next_feature_count=next_state.number_selected,
    )


def accuracy_difference(
    current_state: FeatureSubsetState,
    next_state: FeatureSubsetState,
    accuracy: Callable[[FeatureSubsetState], float],
) -> float:
    """Compatibility wrapper for the pre-Phase-1C callback interface."""
    return calculate_reward(accuracy(current_state), accuracy(next_state))

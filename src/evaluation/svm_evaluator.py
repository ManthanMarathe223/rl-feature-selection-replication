"""Reusable, configurable SVM evaluation for feature subsets."""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.evaluation_config import EvaluationConfig


@dataclass(frozen=True)
class EvaluationResult:
    """Accuracy and metadata returned by one evaluation."""

    accuracy: float
    number_of_samples: int
    number_of_selected_features: int
    runtime_seconds: float
    training_time_seconds: float | None
    evaluation_strategy: str
    model_configuration: dict[str, Any]


class SVMEvaluator:
    """Evaluate selected numeric columns with an explicitly configured SVM."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    def _estimator(self) -> Pipeline | SVC:
        classifier = SVC(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=self.config.gamma,
        )
        steps: list[tuple[str, Any]] = []
        if self.config.missing_value_policy == "mean_impute":
            steps.append(("imputer", SimpleImputer(strategy="mean")))
        if self.config.scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        if steps:
            steps.append(("classifier", classifier))
            return Pipeline(steps)
        return classifier

    def evaluate(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        selected_features: Sequence[int],
    ) -> EvaluationResult:
        """Evaluate only ``selected_features`` using the configured strategy."""
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array.")
        if targets.ndim != 1 or len(features) != len(targets):
            raise ValueError("targets must be one-dimensional and match features.")
        if not selected_features:
            raise ValueError("At least one feature must be selected.")
        indices = list(selected_features)
        if any(index < 0 or index >= features.shape[1] for index in indices):
            raise IndexError("selected feature index is outside the input feature matrix.")
        subset = features[:, indices]
        finite_rows = np.isfinite(subset).all(axis=1)
        if self.config.missing_value_policy == "error" and not finite_rows.all():
            raise ValueError("Selected features contain missing or non-finite values.")
        if self.config.missing_value_policy == "drop_rows":
            subset = subset[finite_rows]
            targets = targets[finite_rows]
            if len(subset) == 0:
                raise ValueError("No rows remain after the explicit drop_rows policy.")

        estimator = self._estimator()
        start = perf_counter()
        if self.config.evaluation_strategy == "holdout":
            x_train, x_test, y_train, y_test = train_test_split(
                subset,
                targets,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=targets,
            )
            training_start = perf_counter()
            estimator.fit(x_train, y_train)
            training_time = perf_counter() - training_start
            accuracy = float(estimator.score(x_test, y_test))
        else:
            scores = cross_val_score(
                estimator,
                subset,
                targets,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
            )
            training_time = None
            accuracy = float(np.mean(scores))
        runtime_seconds = perf_counter() - start
        return EvaluationResult(
            accuracy=accuracy,
            number_of_samples=len(subset),
            number_of_selected_features=len(indices),
            runtime_seconds=runtime_seconds,
            training_time_seconds=training_time,
            evaluation_strategy=self.config.evaluation_strategy,
            model_configuration={
                "kernel": self.config.kernel,
                "C": self.config.C,
                "gamma": self.config.gamma,
                "scaling": self.config.scaling,
                "random_state": self.config.random_state,
                "scoring_metric": self.config.scoring_metric,
                "test_size": self.config.test_size,
                "cv_folds": self.config.cv_folds,
                "missing_value_policy": self.config.missing_value_policy,
            },
        )

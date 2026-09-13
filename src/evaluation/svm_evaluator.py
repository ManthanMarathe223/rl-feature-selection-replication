"""SVM evaluation for a chosen feature subset."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.svm import SVC

from src.evaluation.metrics import classification_accuracy


class SVMEvaluator:
    """Evaluate a feature subset with a configurable support vector classifier."""

    def __init__(self, **svm_parameters: Any) -> None:
        self.svm_parameters = dict(svm_parameters)

    def evaluate(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        selected_features: Sequence[int],
    ) -> float:
        """Fit and score an SVM on the supplied data and selected columns.

        The train/evaluation protocol is intentionally left to the caller until
        the original paper's evaluation procedure has been verified.
        """
        if not selected_features:
            raise ValueError("At least one feature must be selected.")
        subset = features[:, list(selected_features)]
        predictions = SVC(**self.svm_parameters).fit(subset, targets).predict(subset)
        return classification_accuracy(targets, predictions)

"""Evaluation metric helpers."""

from sklearn.metrics import accuracy_score


def classification_accuracy(y_true: object, y_pred: object) -> float:
    """Return classification accuracy as a floating-point value."""
    return float(accuracy_score(y_true, y_pred))

"""Configuration for reproducible feature-subset evaluation."""

from dataclasses import dataclass
from typing import Literal


EvaluationStrategy = Literal["holdout", "cross_validation"]
ScalingOption = Literal["none", "standard"]
MissingValuePolicy = Literal["error", "drop_rows", "mean_impute"]


@dataclass(frozen=True)
class EvaluationConfig:
    """Validated SVM and evaluation settings.

    These defaults are project choices for an executable baseline, not claims
    about the 2021 paper.
    """

    kernel: str = "rbf"
    C: float = 1.0
    gamma: float | str = "scale"
    scaling: ScalingOption = "none"
    random_state: int = 42
    evaluation_strategy: EvaluationStrategy = "holdout"
    scoring_metric: str = "accuracy"
    test_size: float = 0.2
    cv_folds: int = 5
    missing_value_policy: MissingValuePolicy = "error"

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be greater than zero.")
        if isinstance(self.gamma, (int, float)) and self.gamma <= 0:
            raise ValueError("Numeric gamma must be greater than zero.")
        if self.scaling not in {"none", "standard"}:
            raise ValueError("scaling must be 'none' or 'standard'.")
        if self.evaluation_strategy not in {"holdout", "cross_validation"}:
            raise ValueError(
                "evaluation_strategy must be 'holdout' or 'cross_validation'."
            )
        if self.scoring_metric != "accuracy":
            raise ValueError("Only the accuracy scoring metric is supported.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between zero and one.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least two.")
        if self.missing_value_policy not in {"error", "drop_rows", "mean_impute"}:
            raise ValueError(
                "missing_value_policy must be 'error', 'drop_rows', or 'mean_impute'."
            )

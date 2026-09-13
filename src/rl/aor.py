"""Independent Average of Reward (AOR) feature scoring."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


def _validate_feature_index(feature_index: int, number_of_features: int) -> None:
    if not isinstance(feature_index, int) or isinstance(feature_index, bool):
        raise TypeError("feature_index must be a non-negative integer.")
    if feature_index < 0 or feature_index >= number_of_features:
        raise ValueError(
            f"feature_index must be between 0 and {number_of_features - 1}."
        )


def _validate_contribution(contribution: Real) -> float:
    if isinstance(contribution, bool) or not isinstance(contribution, Real):
        raise TypeError("contribution must be a numeric value.")
    value = float(contribution)
    if not isfinite(value):
        raise ValueError("contribution must be finite.")
    return value


@dataclass
class AORTracker:
    """Track a running AOR score independently for every feature.

    ``contribution`` is intentionally generic. This class does not decide
    whether it represents raw reward, a state-value difference, or another
    quantity from a later protocol phase.
    """

    number_of_features: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.number_of_features, int)
            or isinstance(self.number_of_features, bool)
            or self.number_of_features < 0
        ):
            raise ValueError("number_of_features must be a non-negative integer.")
        self._counts = [0] * self.number_of_features
        self._scores = [0.0] * self.number_of_features

    def record(self, feature_index: int, contribution: Real) -> float:
        """Record one contribution and return the updated feature AOR."""
        return self.update(feature_index, contribution)

    def update(self, feature_index: int, contribution: Real) -> float:
        """Apply the incremental AOR update for one feature observation."""
        _validate_feature_index(feature_index, self.number_of_features)
        value = _validate_contribution(contribution)
        count = self._counts[feature_index] + 1
        old_score = self._scores[feature_index]
        new_score = ((count - 1) * old_score + value) / count
        self._counts[feature_index] = count
        self._scores[feature_index] = new_score
        return new_score

    def get_aor(self, feature_index: int) -> float:
        """Return a feature's current running AOR score."""
        _validate_feature_index(feature_index, self.number_of_features)
        return self._scores[feature_index]

    def get_count(self, feature_index: int) -> int:
        """Return how many contributions have been recorded for a feature."""
        _validate_feature_index(feature_index, self.number_of_features)
        return self._counts[feature_index]

    def reset(self) -> None:
        """Reset every feature count and AOR score to zero."""
        self._counts = [0] * self.number_of_features
        self._scores = [0.0] * self.number_of_features

    def all_scores(self) -> dict[int, float]:
        """Return all feature scores in ascending feature-index order."""
        return {
            feature_index: self._scores[feature_index]
            for feature_index in range(self.number_of_features)
        }

    def ranking(self) -> list[int]:
        """Return feature indices by descending AOR, then ascending index."""
        return sorted(
            range(self.number_of_features),
            key=lambda feature_index: (-self._scores[feature_index], feature_index),
        )


@dataclass
class AverageReward:
    """Legacy global-average helper retained for the existing episode scaffold."""

    total: float = 0.0
    count: int = 0

    @property
    def value(self) -> float:
        """Return the current average, or zero before any observations."""
        return self.total / self.count if self.count else 0.0

    def update(self, reward: float) -> float:
        """Add one reward and return the updated average."""
        self.total += float(reward)
        self.count += 1
        return self.value


@dataclass
class AverageReward:
    """Maintain an incremental arithmetic mean of observed rewards."""

    total: float = 0.0
    count: int = 0

    @property
    def value(self) -> float:
        """Return the current average, or zero before any observations."""
        return self.total / self.count if self.count else 0.0

    def update(self, reward: float) -> float:
        """Add one reward and return the updated average."""
        self.total += float(reward)
        self.count += 1
        return self.value

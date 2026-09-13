"""State representation for selected feature subsets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSubsetState:
    """Immutable state represented by sorted, zero-based feature indices."""

    selected_features: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.selected_features)))
        if any(index < 0 for index in normalized):
            raise ValueError("Feature indices must be non-negative.")
        object.__setattr__(self, "selected_features", normalized)

    def add(self, feature_index: int) -> "FeatureSubsetState":
        """Return a new state with one feature selected."""
        return FeatureSubsetState(self.selected_features + (feature_index,))

"""Immutable state representation for selected feature subsets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSubsetState:
    """Immutable state represented by sorted, zero-based feature indices."""

    selected_features: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted(self.selected_features))
        if len(normalized) != len(set(normalized)):
            raise ValueError("selected_features cannot contain duplicates.")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in normalized):
            raise TypeError("Feature indices must be integers.")
        if any(index < 0 for index in normalized):
            raise ValueError("Feature indices must be non-negative.")
        object.__setattr__(self, "selected_features", normalized)

    @property
    def number_selected(self) -> int:
        """Return the number of selected features."""
        return len(self.selected_features)

    def contains(self, feature_index: int) -> bool:
        """Return whether ``feature_index`` is already selected."""
        return feature_index in self.selected_features

    def is_terminal(self, total_features: int) -> bool:
        """Return whether all features in a domain have been selected."""
        if total_features < 0:
            raise ValueError("total_features must be non-negative.")
        if any(index >= total_features for index in self.selected_features):
            raise ValueError("State contains an index outside the feature domain.")
        return self.number_selected == total_features

    def add(self, feature_index: int) -> "FeatureSubsetState":
        """Return a new state with one feature selected."""
        if not isinstance(feature_index, int) or isinstance(feature_index, bool):
            raise TypeError("Feature index must be an integer.")
        if feature_index < 0:
            raise ValueError("Feature index must be non-negative.")
        if self.contains(feature_index):
            raise ValueError(f"Feature {feature_index} is already selected.")
        return FeatureSubsetState(self.selected_features + (feature_index,))

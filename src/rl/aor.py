"""Average of Reward (AOR) tracking."""

from dataclasses import dataclass


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

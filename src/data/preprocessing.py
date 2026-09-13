"""Explicit preprocessing interfaces.

No preprocessing is applied implicitly. The exact protocol must be established
from the replication evidence before these functions are implemented.
"""

from typing import Any


def encode_features(features: Any) -> Any:
    """Encode features according to the verified replication protocol."""
    raise NotImplementedError("Feature encoding protocol has not been established.")


def handle_missing_values(features: Any) -> Any:
    """Handle missing values according to the verified replication protocol."""
    raise NotImplementedError("Missing-value protocol has not been established.")


def split_data(features: Any, targets: Any, **kwargs: Any) -> Any:
    """Split data according to an explicitly configured replication protocol."""
    raise NotImplementedError("Train/test split protocol has not been established.")


def scale_features(features: Any) -> Any:
    """Scale features according to the verified replication protocol."""
    raise NotImplementedError("Feature-scaling protocol has not been established.")

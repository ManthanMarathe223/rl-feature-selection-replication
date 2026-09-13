"""Dataset loading interfaces.

Dataset acquisition is intentionally not automatic. Implementations should read
files already placed under ``data/raw`` and return feature and target arrays.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
TargetArray = NDArray[np.generic]


def _not_implemented(dataset: str) -> None:
    raise NotImplementedError(
        f"The {dataset} loader is a Phase 1 placeholder; place the raw dataset "
        "locally and document its format before implementing it."
    )


def load_australian(path: Path | str | None = None) -> Tuple[Array, TargetArray]:
    """Load the Australian Credit Approval dataset from a local raw file."""
    _not_implemented("Australian Credit Approval")


def load_wpbc(path: Path | str | None = None) -> Tuple[Array, TargetArray]:
    """Load the Wisconsin Breast Cancer Prognostic dataset from a local raw file."""
    _not_implemented("WPBC")


def load_sonar(path: Path | str | None = None) -> Tuple[Array, TargetArray]:
    """Load the Connectionist Bench Sonar dataset from a local raw file."""
    _not_implemented("Sonar")

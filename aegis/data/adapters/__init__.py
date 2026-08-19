"""
Empirical dataset adapters for AEGIS.
"""

from .empirical import (
    EmpiricalSample,
    ImageStatus,
)

from .fakeddit import (
    FAKEDDIT_NATIVE_LABEL_COLUMNS,
    FAKEDDIT_REQUIRED_COLUMNS,
    FAKEDDIT_SPLIT_FILES,
    FakedditAdapter,
)

__all__ = [
    "EmpiricalSample",
    "ImageStatus",
    "FAKEDDIT_NATIVE_LABEL_COLUMNS",
    "FAKEDDIT_REQUIRED_COLUMNS",
    "FAKEDDIT_SPLIT_FILES",
    "FakedditAdapter",
]
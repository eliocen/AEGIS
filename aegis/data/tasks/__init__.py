"""
AEGIS empirical dataset task contracts.
"""

from .fakeddit import (
    AEGISIntegrityTarget,
    FAKEDDIT_BINARY_TASK_NAME,
    FakedditBinaryLabel,
    FakedditBinaryTarget,
    build_fakeddit_binary_target,
    build_fakeddit_binary_target_from_labels,
    map_fakeddit_binary_to_integrity,
    normalize_fakeddit_binary_label,
)

__all__ = [
    "AEGISIntegrityTarget",
    "FAKEDDIT_BINARY_TASK_NAME",
    "FakedditBinaryLabel",
    "FakedditBinaryTarget",
    "normalize_fakeddit_binary_label",
    "map_fakeddit_binary_to_integrity",
    "build_fakeddit_binary_target",
    "build_fakeddit_binary_target_from_labels",
]
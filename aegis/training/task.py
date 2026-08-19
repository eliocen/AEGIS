"""
AEGIS Training Task Definitions.

Version: 0.24.0

Defines explicit empirical training modes so dataset-specific
supervision cannot accidentally be interpreted as a richer AEGIS
information-integrity ontology.
"""

from enum import Enum


class TrainingTask(str, Enum):
    """
    Supported AEGIS training objectives.
    """

    BINARY_INTEGRITY = "binary_integrity"
    HIERARCHICAL = "hierarchical"

    @classmethod
    def from_value(cls, value) -> "TrainingTask":
        if isinstance(value, cls):
            return value

        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(task.value for task in cls)
            raise ValueError(
                f"Unknown AEGIS training task {value!r}. "
                f"Expected one of: {valid}."
            ) from exc
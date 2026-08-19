"""
Common empirical dataset structures for AEGIS.

These structures preserve dataset-native information before
AEGIS label harmonization is applied.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ImageStatus(str, Enum):
    """
    Availability state of a local image associated with
    an empirical multimodal sample.
    """

    AVAILABLE = "available"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class EmpiricalSample:
    """
    Dataset-independent representation of an empirical sample.

    Native labels are deliberately preserved. They must not be
    interpreted as AEGIS labels until an explicit harmonization
    policy has been applied.
    """

    sample_id: str
    dataset: str
    split: str

    text: str

    image_path: Optional[Path]
    image_status: ImageStatus

    native_labels: Dict[str, Any]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    provenance: Dict[str, Any] = field(
        default_factory=dict
    )

    harmonized_label: Optional[str] = None
    harmonization_status: str = "not_harmonized"

    def __post_init__(self) -> None:
        self.sample_id = str(
            self.sample_id
        ).strip()

        if not self.sample_id:
            raise ValueError(
                "EmpiricalSample requires a non-empty sample_id."
            )

        if not isinstance(
            self.text,
            str,
        ):
            raise TypeError(
                "EmpiricalSample.text must be a string."
            )

        self.text = self.text.strip()

        if not self.text:
            raise ValueError(
                "EmpiricalSample requires non-empty text."
            )

        if self.image_path is not None:
            self.image_path = Path(
                self.image_path
            )

        if not isinstance(
            self.image_status,
            ImageStatus,
        ):
            raise TypeError(
                "image_status must be an ImageStatus."
            )

        if not isinstance(
            self.native_labels,
            dict,
        ):
            raise TypeError(
                "native_labels must be a dictionary."
            )
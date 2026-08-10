"""
Validation utilities for AEGIS acquisition records.

Version: 0.5.0
"""

from pathlib import Path

from .record import AcquisitionRecord


class AcquisitionValidationError(ValueError):
    """Raised when an acquisition record is invalid."""


def validate_record(record: AcquisitionRecord) -> None:
    """
    Validate the minimum requirements for an AEGIS acquisition record.
    """

    if not record.sample_id or not str(record.sample_id).strip():
        raise AcquisitionValidationError(
            "AcquisitionRecord must contain a valid sample_id."
        )

    if not record.has_text() and not record.has_image():
        raise AcquisitionValidationError(
            "AcquisitionRecord must contain at least text or an image."
        )

    if record.image_path:
        image_path = Path(record.image_path)

        if image_path.suffix == "":
            raise AcquisitionValidationError(
                f"Image path has no file extension: {record.image_path}"
            )
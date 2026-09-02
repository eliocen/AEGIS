"""
AEGIS Empirical Image Validation
================================

Version: 0.24.0

Provides reusable validation for locally resolved empirical images.

A local path being present is not sufficient evidence that an image
is usable by the empirical representation pipeline. The underlying
file must also be decodable by the configured image library.

This module performs read-only validation. It never modifies,
repairs, deletes, or replaces source dataset files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import (
    Image,
    UnidentifiedImageError,
)


@dataclass(frozen=True)
class ImageValidationResult:
    """
    Result of validating one local empirical image.
    """

    path: str

    exists: bool
    is_file: bool
    decodable: bool

    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None

    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def valid(self) -> bool:
        """
        Return True only when the path exists, is a regular file,
        and can be decoded successfully.
        """

        return (
            self.exists
            and self.is_file
            and self.decodable
        )

    def as_dict(self) -> dict:
        """
        Return a machine-readable validation record.
        """

        return {
            "path": self.path,
            "exists": self.exists,
            "is_file": self.is_file,
            "decodable": self.decodable,
            "valid": self.valid,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def validate_decodable_image(
    image_path: Path | str,
) -> ImageValidationResult:
    """
    Validate that an empirical image exists and can be decoded.

    The function first uses Pillow ``verify()`` for structural
    verification, then reopens the image and calls ``load()`` to
    ensure that actual pixel decoding succeeds.

    No source file is modified.
    """

    path = Path(
        image_path
    )

    if not path.exists():

        return ImageValidationResult(
            path=str(path),
            exists=False,
            is_file=False,
            decodable=False,
            error_type="file_not_found",
            error_message=(
                "Image path does not exist."
            ),
        )

    if not path.is_file():

        return ImageValidationResult(
            path=str(path),
            exists=True,
            is_file=False,
            decodable=False,
            error_type="not_a_file",
            error_message=(
                "Image path is not a regular file."
            ),
        )

    try:

        with Image.open(
            path
        ) as image:

            image_format = (
                image.format
            )

            width, height = (
                image.size
            )

            mode = (
                image.mode
            )

            image.verify()

        # Pillow requires the file to be reopened after verify().
        with Image.open(
            path
        ) as image:

            image.load()

        return ImageValidationResult(
            path=str(path),
            exists=True,
            is_file=True,
            decodable=True,
            format=image_format,
            width=int(
                width
            ),
            height=int(
                height
            ),
            mode=mode,
        )

    except UnidentifiedImageError as exc:

        return ImageValidationResult(
            path=str(path),
            exists=True,
            is_file=True,
            decodable=False,
            error_type="unidentified_image",
            error_message=str(
                exc
            ),
        )

    except OSError as exc:

        return ImageValidationResult(
            path=str(path),
            exists=True,
            is_file=True,
            decodable=False,
            error_type="image_decode_error",
            error_message=str(
                exc
            ),
        )

    except Exception as exc:

        return ImageValidationResult(
            path=str(path),
            exists=True,
            is_file=True,
            decodable=False,
            error_type=(
                type(
                    exc
                ).__name__
            ),
            error_message=str(
                exc
            ),
        )
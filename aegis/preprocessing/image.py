"""
Image preprocessing utilities for AEGIS.

Version: 0.6.0
"""

from pathlib import Path
from typing import Optional


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def normalize_image_path(
    image_path: Optional[str],
) -> Optional[str]:
    """
    Normalize an image path without requiring the file
    to already exist locally.
    """

    if not image_path:
        return None

    image_path = str(image_path).strip()

    if not image_path:
        return None

    path = Path(image_path)

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None

    return str(path)


def image_exists(
    image_path: Optional[str],
) -> bool:
    """
    Check whether the referenced image currently exists.
    """

    if not image_path:
        return False

    return Path(image_path).is_file()
"""
Metadata normalization for AEGIS.

Version: 0.6.0
"""

from typing import Any, Dict, Optional


def normalize_metadata(
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Normalize metadata keys while preserving values.
    """

    if not metadata:
        return {}

    normalized = {}

    for key, value in metadata.items():

        clean_key = (
            str(key)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if not clean_key:
            continue

        normalized[clean_key] = value

    return normalized
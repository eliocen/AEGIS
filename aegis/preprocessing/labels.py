"""
Information-integrity label normalization.

Version: 0.6.0
"""

from typing import Optional

from .exceptions import UnsupportedLabelError


CANONICAL_LABELS = {
    "true": "true",
    "true information": "true",
    "real": "true",
    "verified": "true",

    "misinformation": "misinformation",
    "misinfo": "misinformation",

    "disinformation": "disinformation",
    "disinfo": "disinformation",

    "malinformation": "malinformation",
    "malinfo": "malinformation",

    "hate speech": "hate_speech",
    "hate_speech": "hate_speech",
    "hatespeech": "hate_speech",
}


def normalize_label(
    label: Optional[str],
    strict: bool = False,
) -> Optional[str]:
    """
    Normalize dataset-specific labels into the AEGIS
    information-integrity taxonomy.

    Canonical classes:
    - true
    - misinformation
    - disinformation
    - malinformation
    - hate_speech
    """

    if label is None:
        return None

    normalized = (
        str(label)
        .strip()
        .lower()
    )

    if not normalized:
        return None

    canonical = CANONICAL_LABELS.get(
        normalized
    )

    if canonical:
        return canonical

    if strict:
        raise UnsupportedLabelError(
            f"Unsupported AEGIS label: {label}"
        )

    return normalized
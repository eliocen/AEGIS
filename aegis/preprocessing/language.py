"""
Language normalization utilities for AEGIS.

Version: 0.6.0
"""

from typing import Optional


LANGUAGE_ALIASES = {
    "english": "en",
    "en-us": "en",
    "en-gb": "en",

    "chinese": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "mandarin": "zh",

    "french": "fr",
    "fr-fr": "fr",

    "arabic": "ar",

    "swahili": "sw",
    "kiswahili": "sw",

    "spanish": "es",

    "russian": "ru",

    "luganda": "lg",

    "lango": "laj",
}


def normalize_language(
    language: Optional[str],
) -> str:
    """
    Normalize a language description into a compact language code.

    Unknown or missing languages are represented as 'und'.
    """

    if not language:
        return "und"

    normalized = (
        str(language)
        .strip()
        .lower()
        .replace("_", "-")
    )

    if not normalized:
        return "und"

    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]

    if len(normalized) in {2, 3}:
        return normalized

    return "und"
"""
Multilingual-safe text preprocessing for AEGIS.

Version: 0.6.0
"""

import html
import re
import unicodedata
from typing import Optional


_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Normalize multilingual text while preserving semantic content.

    Operations:
    - HTML entity decoding
    - Unicode normalization
    - whitespace normalization
    - leading/trailing whitespace removal

    This intentionally does NOT:
    - lowercase text
    - remove punctuation
    - remove hashtags
    - remove mentions
    - remove emojis
    - remove non-Latin characters

    These may contain information relevant to cognitive threat detection.
    """

    if text is None:
        return None

    text = str(text)

    text = html.unescape(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = _WHITESPACE_PATTERN.sub(
        " ",
        text,
    )

    text = text.strip()

    return text or None
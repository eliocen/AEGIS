"""
AEGIS Language Utilities.

Version: 0.22.0
"""

from dataclasses import dataclass


LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",

    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
}


def normalize_language(
    language,
):
    """
    Normalize language identifiers used by
    multilingual AEGIS evaluation.

    Unknown but non-empty ISO-like language codes
    are preserved after normalization.
    """

    if language is None:
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
        return LANGUAGE_ALIASES[
            normalized
        ]

    return normalized


@dataclass(frozen=True)
class LanguagePair:
    """
    Defines one train-language/test-language pair.
    """

    train_language: str
    test_language: str

    def __post_init__(self):

        object.__setattr__(
            self,
            "train_language",
            normalize_language(
                self.train_language
            ),
        )

        object.__setattr__(
            self,
            "test_language",
            normalize_language(
                self.test_language
            ),
        )

    @property
    def is_in_language(self):

        return (
            self.train_language
            == self.test_language
        )

    @property
    def is_cross_lingual(self):

        return not (
            self.is_in_language
        )

    @property
    def key(self):

        return (
            f"{self.train_language}"
            f"->{self.test_language}"
        )
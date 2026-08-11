"""
AEGIS Controlled Perturbation Utilities.

Version: 0.23.0
"""

import random
import string


def whitespace_perturbation(
    text: str,
):
    """
    Introduce harmless whitespace variation.
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string."
        )

    return (
        "  "
        .join(
            text.split()
        )
    )


def lowercase_perturbation(
    text: str,
):
    """
    Convert text to lowercase.
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string."
        )

    return text.lower()


def punctuation_perturbation(
    text: str,
):
    """
    Remove punctuation while preserving text.
    """

    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "text must be a string."
        )

    return "".join(
        character
        for character in text
        if character
        not in string.punctuation
    )


def word_deletion_perturbation(
    text: str,
    probability: float = 0.10,
    seed: int = 42,
):
    """
    Randomly remove words with a controlled seed.
    """

    if not (
        0.0 <= probability <= 1.0
    ):
        raise ValueError(
            "probability must lie "
            "within [0, 1]."
        )

    words = text.split()

    if len(words) <= 1:
        return text

    rng = random.Random(
        seed
    )

    retained = [
        word
        for word in words
        if rng.random()
        > probability
    ]

    if not retained:

        retained = [
            words[
                rng.randrange(
                    len(words)
                )
            ]
        ]

    return " ".join(
        retained
    )


def character_noise_perturbation(
    text: str,
    probability: float = 0.05,
    seed: int = 42,
):
    """
    Introduce minor character substitutions.
    """

    if not (
        0.0 <= probability <= 1.0
    ):
        raise ValueError(
            "probability must lie "
            "within [0, 1]."
        )

    rng = random.Random(
        seed
    )

    characters = list(
        text
    )

    alphabet = (
        string.ascii_lowercase
    )

    for index, character in enumerate(
        characters
    ):

        if (
            character.isalpha()
            and rng.random()
            < probability
        ):

            replacement = (
                rng.choice(
                    alphabet
                )
            )

            if character.isupper():

                replacement = (
                    replacement.upper()
                )

            characters[
                index
            ] = replacement

    return "".join(
        characters
    )
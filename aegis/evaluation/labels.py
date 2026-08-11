"""
AEGIS Evaluation Label Utilities.

Version: 0.20.0
"""


INTEGRITY_LABELS = [
    "true",
    "harmful",
]


THREAT_LABELS = [
    "misinformation",
    "disinformation",
    "malinformation",
    "hate_speech",
]


FIVE_CLASS_LABELS = [
    "true",
    "misinformation",
    "disinformation",
    "malinformation",
    "hate_speech",
]


FIVE_CLASS_LABEL_TO_INDEX = {
    label: index
    for index, label
    in enumerate(
        FIVE_CLASS_LABELS
    )
}


def hierarchical_label(
    integrity_target: int,
    threat_target: int,
):
    """
    Convert hierarchical targets into a canonical
    AEGIS five-class label.

    Integrity:
        0 = TRUE
        1 = HARMFUL

    Threat:
        0 = misinformation
        1 = disinformation
        2 = malinformation
        3 = hate speech
    """

    if integrity_target == 0:
        return "true"

    if integrity_target != 1:
        raise ValueError(
            "integrity_target must be 0 or 1."
        )

    if (
        threat_target < 0
        or threat_target
        >= len(THREAT_LABELS)
    ):
        raise ValueError(
            "Harmful samples require "
            "threat_target in the range 0..3."
        )

    return THREAT_LABELS[
        threat_target
    ]


def five_class_index(
    label: str,
) -> int:
    """
    Convert a canonical five-class string label
    into its numeric evaluation index.
    """

    normalized = (
        str(label)
        .strip()
        .lower()
    )

    if (
        normalized
        not in FIVE_CLASS_LABEL_TO_INDEX
    ):
        raise ValueError(
            "Unsupported five-class label: "
            f"{label}"
        )

    return FIVE_CLASS_LABEL_TO_INDEX[
        normalized
    ]
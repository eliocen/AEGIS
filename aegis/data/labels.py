"""
AEGIS Research Dataset Label Mapping.

Version: 0.17.0
"""

from enum import Enum


class DatasetLabel(Enum):
    TRUE = "true"
    MISINFORMATION = "misinformation"
    DISINFORMATION = "disinformation"
    MALINFORMATION = "malinformation"
    HATE_SPEECH = "hate_speech"


LABEL_ALIASES = {
    "true": DatasetLabel.TRUE,
    "real": DatasetLabel.TRUE,
    "verified": DatasetLabel.TRUE,
    "true_information": DatasetLabel.TRUE,

    "misinformation": DatasetLabel.MISINFORMATION,
    "misinfo": DatasetLabel.MISINFORMATION,

    "disinformation": DatasetLabel.DISINFORMATION,
    "disinfo": DatasetLabel.DISINFORMATION,

    "malinformation": DatasetLabel.MALINFORMATION,
    "malinfo": DatasetLabel.MALINFORMATION,

    "hate_speech": DatasetLabel.HATE_SPEECH,
    "hate speech": DatasetLabel.HATE_SPEECH,
    "hatespeech": DatasetLabel.HATE_SPEECH,
}


THREAT_TARGETS = {
    DatasetLabel.MISINFORMATION: 0,
    DatasetLabel.DISINFORMATION: 1,
    DatasetLabel.MALINFORMATION: 2,
    DatasetLabel.HATE_SPEECH: 3,
}


def normalize_dataset_label(
    value,
) -> DatasetLabel:
    """
    Normalize an external dataset label into
    the canonical AEGIS five-class taxonomy.
    """

    if isinstance(
        value,
        DatasetLabel,
    ):
        return value

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
    )

    if normalized not in LABEL_ALIASES:
        raise ValueError(
            f"Unsupported AEGIS dataset label: {value}"
        )

    return LABEL_ALIASES[
        normalized
    ]


def label_to_hierarchical_targets(
    label,
):
    """
    Convert the canonical five-class label into:

    integrity_target:
        0 = TRUE
        1 = HARMFUL

    threat_target:
       -1 = not applicable for TRUE
        0 = misinformation
        1 = disinformation
        2 = malinformation
        3 = hate speech
    """

    label = normalize_dataset_label(
        label
    )

    if label == DatasetLabel.TRUE:
        return 0, -1

    return (
        1,
        THREAT_TARGETS[label],
    )
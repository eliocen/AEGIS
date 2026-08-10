"""
AEGIS hierarchical Information Integrity taxonomy.

Version: 0.10.0
"""

from enum import Enum


class IntegrityStatus(Enum):
    """
    Stage-1 Information Integrity Assessment.
    """

    TRUE = "true"
    HARMFUL = "harmful"


class CognitiveThreatType(Enum):
    """
    Stage-2 Cognitive Cyber Threat taxonomy.

    Applied only when information is classified as harmful.
    """

    MISINFORMATION = "misinformation"
    DISINFORMATION = "disinformation"
    MALINFORMATION = "malinformation"
    HATE_SPEECH = "hate_speech"


INTEGRITY_INDEX_TO_LABEL = {
    0: IntegrityStatus.TRUE,
    1: IntegrityStatus.HARMFUL,
}


INTEGRITY_LABEL_TO_INDEX = {
    value: key
    for key, value
    in INTEGRITY_INDEX_TO_LABEL.items()
}


THREAT_INDEX_TO_LABEL = {
    0: CognitiveThreatType.MISINFORMATION,
    1: CognitiveThreatType.DISINFORMATION,
    2: CognitiveThreatType.MALINFORMATION,
    3: CognitiveThreatType.HATE_SPEECH,
}


THREAT_LABEL_TO_INDEX = {
    value: key
    for key, value
    in THREAT_INDEX_TO_LABEL.items()
}
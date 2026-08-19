"""
AEGIS Fakeddit Empirical Task Contracts
=======================================

Version: 0.24.0

Defines scientifically defensible mappings between Fakeddit native
labels and AEGIS empirical training targets.

Important
---------
Fakeddit native labels are preserved exactly.

The binary Fakeddit task may support AEGIS Stage-1 integrity
classification:

    TRUE vs HARMFUL / NON-AUTHENTIC

It does NOT provide sufficient annotation evidence to infer the
intent-sensitive AEGIS threat subtypes:

    Misinformation
    Disinformation
    Malinformation
    Hate Speech

No subtype target is therefore manufactured from the binary label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Optional


FAKEDDIT_BINARY_TASK_NAME = "fakeddit_2way_integrity"


class FakedditBinaryLabel(IntEnum):
    """
    Native Fakeddit 2-way labels.

    Native encoding:
        0 -> Fake
        1 -> True
    """

    FAKE = 0
    TRUE = 1


class AEGISIntegrityTarget(IntEnum):
    """
    AEGIS Stage-1 integrity targets.

    AEGIS encoding:
        0 -> True information
        1 -> Harmful / non-authentic information
    """

    TRUE = 0
    HARMFUL = 1


@dataclass(frozen=True)
class FakedditBinaryTarget:
    """
    Dataset-native and AEGIS-compatible target for one Fakeddit sample.

    threat_target deliberately remains None because the Fakeddit
    binary task does not identify misinformation, disinformation,
    malinformation, or hate speech.
    """

    sample_id: str

    native_label: int
    native_name: str

    integrity_target: int
    integrity_name: str

    threat_target: Optional[int] = None

    task_name: str = FAKEDDIT_BINARY_TASK_NAME

    source_dataset: str = "Fakeddit"

    def as_dict(self) -> dict[str, Any]:
        """
        Return a machine-readable target representation.
        """

        return {
            "sample_id": self.sample_id,
            "source_dataset": self.source_dataset,
            "task_name": self.task_name,
            "native_label": self.native_label,
            "native_name": self.native_name,
            "integrity_target": self.integrity_target,
            "integrity_name": self.integrity_name,
            "threat_target": self.threat_target,
        }


def normalize_fakeddit_binary_label(
    value: Any,
) -> FakedditBinaryLabel:
    """
    Normalize one Fakeddit 2-way label.

    Raises
    ------
    ValueError
        If the value is not a supported Fakeddit binary label.
    """

    try:
        label = int(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Fakeddit 2-way label: {value!r}"
        ) from exc

    try:
        return FakedditBinaryLabel(label)

    except ValueError as exc:
        raise ValueError(
            "Fakeddit 2-way labels must be 0 or 1; "
            f"received {label}."
        ) from exc


def map_fakeddit_binary_to_integrity(
    value: Any,
) -> AEGISIntegrityTarget:
    """
    Map the native Fakeddit binary label to AEGIS Stage 1.

    Native:
        0 = Fake
        1 = True

    AEGIS:
        0 = True
        1 = Harmful / non-authentic
    """

    native = normalize_fakeddit_binary_label(
        value
    )

    if native is FakedditBinaryLabel.TRUE:
        return AEGISIntegrityTarget.TRUE

    return AEGISIntegrityTarget.HARMFUL


def build_fakeddit_binary_target(
    sample_id: str,
    native_label: Any,
) -> FakedditBinaryTarget:
    """
    Construct the complete empirical task target for one sample.
    """

    sample_id = str(sample_id).strip()

    if not sample_id:
        raise ValueError(
            "sample_id must be non-empty."
        )

    native = normalize_fakeddit_binary_label(
        native_label
    )

    integrity = map_fakeddit_binary_to_integrity(
        native
    )

    return FakedditBinaryTarget(
        sample_id=sample_id,

        native_label=int(native),

        native_name=(
            "fake"
            if native is FakedditBinaryLabel.FAKE
            else "true"
        ),

        integrity_target=int(integrity),

        integrity_name=(
            "harmful"
            if integrity is AEGISIntegrityTarget.HARMFUL
            else "true"
        ),

        threat_target=None,
    )


def build_fakeddit_binary_target_from_labels(
    sample_id: str,
    native_labels: Mapping[str, Any],
) -> FakedditBinaryTarget:
    """
    Construct a target from an adapter-native label dictionary.

    The adapter must preserve the original Fakeddit column
    `2_way_label`.
    """

    if not isinstance(
        native_labels,
        Mapping,
    ):
        raise TypeError(
            "native_labels must be a mapping."
        )

    if "2_way_label" not in native_labels:
        raise KeyError(
            "Fakeddit native labels do not contain "
            "'2_way_label'."
        )

    return build_fakeddit_binary_target(
        sample_id=sample_id,
        native_label=native_labels[
            "2_way_label"
        ],
    )
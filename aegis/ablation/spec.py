"""
AEGIS Ablation Study Specification.

Version: 0.21.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from typing import Any, Dict


@dataclass(frozen=True)
class AblationSpec:
    """
    Defines one controlled AEGIS ablation variant.

    An ablation should differ from the baseline by
    a small, explicitly documented set of changes.
    """

    name: str

    description: str = ""

    use_text: bool = True

    use_vision: bool = True

    alignment_loss_weight: float = 1.0

    classification_loss_weight: float = 1.0

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def __post_init__(self):

        if not self.name.strip():
            raise ValueError(
                "Ablation name cannot be empty."
            )

        if not (
            self.use_text
            or self.use_vision
        ):
            raise ValueError(
                "At least one modality must remain enabled."
            )

        if self.alignment_loss_weight < 0:
            raise ValueError(
                "alignment_loss_weight "
                "must be non-negative."
            )

        if (
            self.classification_loss_weight
            < 0
        ):
            raise ValueError(
                "classification_loss_weight "
                "must be non-negative."
            )

        if (
            self.alignment_loss_weight == 0
            and self.classification_loss_weight == 0
        ):
            raise ValueError(
                "At least one training objective "
                "must remain active."
            )

    @property
    def is_text_only(self):

        return (
            self.use_text
            and not self.use_vision
        )

    @property
    def is_vision_only(self):

        return (
            self.use_vision
            and not self.use_text
        )

    @property
    def is_multimodal(self):

        return (
            self.use_text
            and self.use_vision
        )

    def as_dict(self):

        return asdict(
            self
        )
"""
AEGIS Research Dataset Record.

Version: 0.17.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .labels import (
    DatasetLabel,
    label_to_hierarchical_targets,
    normalize_dataset_label,
)


@dataclass
class ResearchSample:
    """
    One model-ready research sample.

    v0.17.0 assumes that text and visual embeddings
    have already been generated.
    """

    sample_id: str

    text_embedding: torch.Tensor
    vision_embedding: torch.Tensor

    label: DatasetLabel

    language: str = "und"

    domain: str = "general"

    source_dataset: str = "unknown"

    group_id: Optional[str] = None

    source_id: Optional[str] = None

    event_id: Optional[str] = None

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def __post_init__(self):

        self.label = (
            normalize_dataset_label(
                self.label
            )
        )

    @property
    def integrity_target(self):
        return (
            label_to_hierarchical_targets(
                self.label
            )[0]
        )

    @property
    def threat_target(self):
        return (
            label_to_hierarchical_targets(
                self.label
            )[1]
        )
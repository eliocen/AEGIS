"""
AEGIS Hierarchical Information Integrity Classifier.

Version: 0.10.0
"""

import torch
from torch import nn


class HierarchicalInformationIntegrityClassifier(
    nn.Module
):
    """
    Two-stage neural classifier for AEGIS.

    Stage 1:
        True vs Harmful

    Stage 2:
        Misinformation
        Disinformation
        Malinformation
        Hate Speech
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.shared_backbone = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.LayerNorm(
                hidden_dim
            ),
        )

        # Stage 1:
        # True vs Harmful
        self.integrity_head = nn.Linear(
            hidden_dim,
            2,
        )

        # Stage 2:
        # MDMH classification
        self.threat_head = nn.Linear(
            hidden_dim,
            4,
        )

    def forward(
        self,
        embeddings,
    ):
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape "
                "[batch_size, input_dimension]."
            )

        if (
            embeddings.shape[-1]
            != self.input_dim
        ):
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.input_dim}, "
                f"received "
                f"{embeddings.shape[-1]}."
            )

        features = self.shared_backbone(
            embeddings
        )

        integrity_logits = (
            self.integrity_head(
                features
            )
        )

        threat_logits = (
            self.threat_head(
                features
            )
        )

        return {
            "features": features,
            "integrity_logits": (
                integrity_logits
            ),
            "threat_logits": (
                threat_logits
            ),
        }
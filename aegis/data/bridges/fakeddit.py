"""
AEGIS Fakeddit Empirical Representation Bridge
==============================================

Version: 0.24.0

Bridges audited Fakeddit EmpiricalSample objects into the canonical
AEGIS preprocessing and frozen representation pipeline.

Pipeline
--------
EmpiricalSample
    -> CanonicalSample
    -> TextRepresentation
    -> VisionRepresentation
    -> Fakeddit binary integrity target
    -> BinaryIntegrityBatch

Important
---------
This module does not reinterpret Fakeddit native labels as AEGIS
threat subtypes.

Only the explicit Fakeddit 2-way -> AEGIS Stage-1 integrity contract
is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch

from aegis.data.adapters.empirical import (
    EmpiricalSample,
    ImageStatus,
)
from aegis.data.tasks import (
    build_fakeddit_binary_target_from_labels,
)
from aegis.preprocessing import CanonicalSample
from aegis.training import BinaryIntegrityBatch


@dataclass(frozen=True)
class FakedditRepresentedSample:
    """
    One Fakeddit sample after frozen multimodal representation.
    """

    sample_id: str

    text_embedding: torch.Tensor
    vision_embedding: torch.Tensor

    integrity_target: int
    native_label: int

    text_model: str
    vision_model: str

    image_path: str


def empirical_to_canonical(
    sample: EmpiricalSample,
) -> CanonicalSample:
    """
    Convert an audited Fakeddit EmpiricalSample into the canonical
    model-ready AEGIS representation.

    Fakeddit is English-dominant, so the language is explicitly
    recorded as English for this empirical phase.
    """

    if not isinstance(sample, EmpiricalSample):
        raise TypeError(
            "empirical_to_canonical expects an EmpiricalSample."
        )

    if not sample.text:
        raise ValueError(
            f"Sample {sample.sample_id} does not contain text."
        )

    if sample.image_status is not ImageStatus.AVAILABLE:
        raise ValueError(
            f"Sample {sample.sample_id} does not have an "
            "available local image."
        )

    if sample.image_path is None:
        raise ValueError(
            f"Sample {sample.sample_id} has AVAILABLE image status "
            "but no image_path."
        )

    metadata = dict(sample.metadata)

    metadata.update(
        {
            "dataset": sample.dataset,
            "split": sample.split,
            "native_labels": dict(
                sample.native_labels
            ),
            "provenance": dict(
                sample.provenance
            ),
        }
    )

    return CanonicalSample(
        sample_id=sample.sample_id,
        text=sample.text,
        image_path=str(sample.image_path),
        language="en",
        source=sample.dataset,
        platform="reddit",
        label=None,
        has_text=True,
        has_image=True,
        is_multimodal=True,
        metadata=metadata,
        preprocessing={
            "empirical_bridge": (
                "fakeddit_v0.24.0"
            ),
            "text_field": "clean_title",
            "image_status": (
                sample.image_status.value
            ),
        },
    )


class FakedditRepresentationBridge:
    """
    Produce real frozen text and vision representations from
    Fakeddit empirical samples.

    The supplied encoders must expose:

        encode(CanonicalSample)

    and return objects containing:

        sample_id
        embedding
        model_name
        dimension
    """

    def __init__(
        self,
        text_encoder,
        vision_encoder,
        expected_text_dim: int = 768,
        expected_vision_dim: int = 512,
    ):
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

        self.expected_text_dim = int(
            expected_text_dim
        )

        self.expected_vision_dim = int(
            expected_vision_dim
        )

        if self.expected_text_dim <= 0:
            raise ValueError(
                "expected_text_dim must be positive."
            )

        if self.expected_vision_dim <= 0:
            raise ValueError(
                "expected_vision_dim must be positive."
            )

    @staticmethod
    def _normalize_embedding(
        embedding,
        expected_dim: int,
        name: str,
    ) -> torch.Tensor:
        """
        Normalize an encoder output into a detached 1-D CPU tensor.
        """

        if not torch.is_tensor(embedding):
            try:
                embedding = torch.as_tensor(
                    embedding
                )
            except Exception as exc:
                raise TypeError(
                    f"{name} embedding cannot be converted "
                    "to a torch.Tensor."
                ) from exc

        embedding = (
            embedding
            .detach()
            .cpu()
            .float()
        )

        if embedding.ndim == 2:
            if embedding.shape[0] != 1:
                raise ValueError(
                    f"{name} embedding must represent one sample."
                )

            embedding = embedding.squeeze(0)

        if embedding.ndim != 1:
            raise ValueError(
                f"{name} embedding must be one-dimensional; "
                f"received shape {tuple(embedding.shape)}."
            )

        if embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {name} embedding dimension "
                f"{expected_dim}, received "
                f"{embedding.shape[0]}."
            )

        if not torch.isfinite(
            embedding
        ).all():
            raise ValueError(
                f"{name} embedding contains non-finite values."
            )

        return embedding.contiguous()

    def represent(
        self,
        sample: EmpiricalSample,
    ) -> FakedditRepresentedSample:
        """
        Encode one real Fakeddit sample.
        """

        canonical = empirical_to_canonical(
            sample
        )

        target = (
            build_fakeddit_binary_target_from_labels(
                sample_id=sample.sample_id,
                native_labels=sample.native_labels,
            )
        )

        text_representation = (
            self.text_encoder.encode(
                canonical
            )
        )

        vision_representation = (
            self.vision_encoder.encode(
                canonical
            )
        )

        if (
            text_representation.sample_id
            != sample.sample_id
        ):
            raise ValueError(
                "Text representation sample ID mismatch."
            )

        if (
            vision_representation.sample_id
            != sample.sample_id
        ):
            raise ValueError(
                "Vision representation sample ID mismatch."
            )

        text_embedding = (
            self._normalize_embedding(
                text_representation.embedding,
                expected_dim=(
                    self.expected_text_dim
                ),
                name="text",
            )
        )

        vision_embedding = (
            self._normalize_embedding(
                vision_representation.embedding,
                expected_dim=(
                    self.expected_vision_dim
                ),
                name="vision",
            )
        )

        return FakedditRepresentedSample(
            sample_id=sample.sample_id,

            text_embedding=(
                text_embedding
            ),

            vision_embedding=(
                vision_embedding
            ),

            integrity_target=(
                target.integrity_target
            ),

            native_label=(
                target.native_label
            ),

            text_model=(
                text_representation.model_name
            ),

            vision_model=(
                vision_representation.model_name
            ),

            image_path=str(
                sample.image_path
            ),
        )

    def iter_representations(
        self,
        samples: Iterable[
            EmpiricalSample
        ],
    ) -> Iterator[
        FakedditRepresentedSample
    ]:
        """
        Lazily represent empirical samples.
        """

        for sample in samples:
            yield self.represent(
                sample
            )

    def build_batch(
        self,
        samples: Iterable[
            EmpiricalSample
        ],
    ) -> BinaryIntegrityBatch:
        """
        Encode empirical samples and construct one binary batch.
        """

        represented: List[
            FakedditRepresentedSample
        ] = list(
            self.iter_representations(
                samples
            )
        )

        if not represented:
            raise ValueError(
                "Cannot build a binary batch from zero samples."
            )

        text_embeddings = torch.stack(
            [
                item.text_embedding
                for item in represented
            ],
            dim=0,
        )

        vision_embeddings = torch.stack(
            [
                item.vision_embedding
                for item in represented
            ],
            dim=0,
        )

        integrity_targets = torch.tensor(
            [
                item.integrity_target
                for item in represented
            ],
            dtype=torch.long,
        )

        sample_ids = [
            item.sample_id
            for item in represented
        ]

        batch = BinaryIntegrityBatch(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            integrity_targets=integrity_targets,
            sample_ids=sample_ids,
        )

        batch.validate(
            text_dim=self.expected_text_dim,
            vision_dim=self.expected_vision_dim,
        )

        return batch

    def iter_batches(
        self,
        samples: Iterable[
            EmpiricalSample
        ],
        batch_size: int,
    ) -> Iterator[
        BinaryIntegrityBatch
    ]:
        """
        Stream represented Fakeddit samples into binary batches.

        This method never loads the entire dataset into memory.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        buffer: List[
            EmpiricalSample
        ] = []

        for sample in samples:
            buffer.append(
                sample
            )

            if len(buffer) == batch_size:
                yield self.build_batch(
                    buffer
                )

                buffer = []

        if buffer:
            yield self.build_batch(
                buffer
            )
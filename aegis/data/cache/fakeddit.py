"""
AEGIS Fakeddit Frozen Representation Cache
==========================================

Version: 0.24.0

Provides a reproducible, sharded cache for frozen Fakeddit
text and vision representations.

The cache stores representations produced by frozen upstream
encoders so that XLM-R and CLIP do not need to be recomputed
during every AEGIS training epoch.

This cache contains empirical representations, not model
checkpoints and not raw source data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch

from aegis.data.bridges import (
    FakedditRepresentedSample,
)
from aegis.training import (
    BinaryIntegrityBatch,
)


CACHE_SCHEMA = "aegis_fakeddit_frozen_embeddings"
CACHE_VERSION = "1.0"


@dataclass(frozen=True)
class FakedditCacheManifest:
    """
    Metadata describing one frozen representation cache.
    """

    cache_schema: str
    cache_version: str

    dataset: str
    split: str

    sample_count: int
    shard_count: int
    shard_size: int

    text_model: str
    vision_model: str

    text_dimension: int
    vision_dimension: int

    sampling_strategy: str
    sampling_seed: Optional[int]

    created_at_utc: str

    def as_dict(self) -> Dict:
        return {
            "cache_schema": self.cache_schema,
            "cache_version": self.cache_version,
            "dataset": self.dataset,
            "split": self.split,
            "sample_count": self.sample_count,
            "shard_count": self.shard_count,
            "shard_size": self.shard_size,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "text_dimension": self.text_dimension,
            "vision_dimension": self.vision_dimension,
            "sampling_strategy": self.sampling_strategy,
            "sampling_seed": self.sampling_seed,
            "created_at_utc": self.created_at_utc,
        }


class FakedditRepresentationCacheWriter:
    """
    Write frozen Fakeddit representations into deterministic shards.
    """

    def __init__(
        self,
        cache_root: Path | str,
        split: str,
        shard_size: int = 1000,
        sampling_strategy: str = "sequential",
        sampling_seed: Optional[int] = None,
    ):
        self.cache_root = Path(
            cache_root
        )

        self.split = str(
            split
        ).strip().lower()

        if not self.split:
            raise ValueError(
                "split must be non-empty."
            )

        self.shard_size = int(
            shard_size
        )

        if self.shard_size <= 0:
            raise ValueError(
                "shard_size must be greater than zero."
            )

        self.sampling_strategy = str(
            sampling_strategy
        ).strip()

        if not self.sampling_strategy:
            raise ValueError(
                "sampling_strategy must be non-empty."
            )

        self.sampling_seed = (
            None
            if sampling_seed is None
            else int(sampling_seed)
        )

        self.shard_root = (
            self.cache_root
            / "shards"
        )

    @staticmethod
    def _validate_item(
        item: FakedditRepresentedSample,
        text_dimension: int,
        vision_dimension: int,
    ) -> None:

        if not isinstance(
            item,
            FakedditRepresentedSample,
        ):
            raise TypeError(
                "Cache items must be "
                "FakedditRepresentedSample objects."
            )

        if item.text_embedding.ndim != 1:
            raise ValueError(
                "text_embedding must be one-dimensional."
            )

        if item.vision_embedding.ndim != 1:
            raise ValueError(
                "vision_embedding must be one-dimensional."
            )

        if (
            item.text_embedding.shape[0]
            != text_dimension
        ):
            raise ValueError(
                "Unexpected text embedding dimension."
            )

        if (
            item.vision_embedding.shape[0]
            != vision_dimension
        ):
            raise ValueError(
                "Unexpected vision embedding dimension."
            )

        if item.integrity_target not in {
            0,
            1,
        }:
            raise ValueError(
                "integrity_target must be 0 or 1."
            )

    @staticmethod
    def _build_shard(
        items: List[
            FakedditRepresentedSample
        ],
    ) -> Dict:

        return {
            "sample_ids": [
                item.sample_id
                for item in items
            ],

            "text_embeddings": torch.stack(
                [
                    item.text_embedding
                    .detach()
                    .cpu()
                    .float()
                    for item in items
                ],
                dim=0,
            ),

            "vision_embeddings": torch.stack(
                [
                    item.vision_embedding
                    .detach()
                    .cpu()
                    .float()
                    for item in items
                ],
                dim=0,
            ),

            "integrity_targets": torch.tensor(
                [
                    item.integrity_target
                    for item in items
                ],
                dtype=torch.long,
            ),

            "native_labels": torch.tensor(
                [
                    item.native_label
                    for item in items
                ],
                dtype=torch.long,
            ),
        }

    def write(
        self,
        representations: Iterable[
            FakedditRepresentedSample
        ],
    ) -> FakedditCacheManifest:
        """
        Write an iterable of represented samples to the cache.
        """

        self.cache_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.shard_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        items = iter(
            representations
        )

        try:
            first = next(
                items
            )
        except StopIteration as exc:
            raise ValueError(
                "Cannot create a cache from zero samples."
            ) from exc

        text_dimension = int(
            first.text_embedding.shape[-1]
        )

        vision_dimension = int(
            first.vision_embedding.shape[-1]
        )

        text_model = first.text_model
        vision_model = first.vision_model

        sample_count = 0
        shard_count = 0

        index_path = (
            self.cache_root
            / "index.jsonl"
        )

        with index_path.open(
            "w",
            encoding="utf-8",
        ) as index_handle:

            buffer: List[
                FakedditRepresentedSample
            ] = []

            def add_item(
                item: FakedditRepresentedSample,
            ) -> None:
                nonlocal sample_count

                self._validate_item(
                    item,
                    text_dimension=text_dimension,
                    vision_dimension=vision_dimension,
                )

                if item.text_model != text_model:
                    raise ValueError(
                        "All cached text representations "
                        "must use the same text model."
                    )

                if item.vision_model != vision_model:
                    raise ValueError(
                        "All cached vision representations "
                        "must use the same vision model."
                    )

                buffer.append(
                    item
                )

                sample_count += 1

            add_item(
                first
            )

            for item in items:

                add_item(
                    item
                )

                if (
                    len(buffer)
                    >= self.shard_size
                ):
                    shard_count = (
                        self._flush_shard(
                            buffer=buffer,
                            shard_number=shard_count,
                            index_handle=index_handle,
                        )
                    )

                    buffer = []

            if buffer:
                shard_count = (
                    self._flush_shard(
                        buffer=buffer,
                        shard_number=shard_count,
                        index_handle=index_handle,
                    )
                )

        manifest = FakedditCacheManifest(
            cache_schema=CACHE_SCHEMA,
            cache_version=CACHE_VERSION,

            dataset="Fakeddit",
            split=self.split,

            sample_count=sample_count,
            shard_count=shard_count,
            shard_size=self.shard_size,

            text_model=text_model,
            vision_model=vision_model,

            text_dimension=text_dimension,
            vision_dimension=vision_dimension,

            sampling_strategy=(
                self.sampling_strategy
            ),

            sampling_seed=(
                self.sampling_seed
            ),

            created_at_utc=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

        manifest_path = (
            self.cache_root
            / "cache_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                manifest.as_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        return manifest

    def _flush_shard(
        self,
        buffer: List[
            FakedditRepresentedSample
        ],
        shard_number: int,
        index_handle,
    ) -> int:

        shard_name = (
            f"shard_{shard_number:05d}.pt"
        )

        shard_path = (
            self.shard_root
            / shard_name
        )

        payload = (
            self._build_shard(
                buffer
            )
        )

        torch.save(
            payload,
            shard_path,
        )

        for offset, item in enumerate(
            buffer
        ):
            record = {
                "sample_id": item.sample_id,
                "shard": shard_name,
                "offset": offset,
                "integrity_target": (
                    item.integrity_target
                ),
                "native_label": (
                    item.native_label
                ),
            }

            index_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return (
            shard_number
            + 1
        )


class FakedditRepresentationCache:
    """
    Read an existing Fakeddit frozen-representation cache.
    """

    def __init__(
        self,
        cache_root: Path | str,
    ):
        self.cache_root = Path(
            cache_root
        )

        self.manifest_path = (
            self.cache_root
            / "cache_manifest.json"
        )

        self.index_path = (
            self.cache_root
            / "index.jsonl"
        )

        self.shard_root = (
            self.cache_root
            / "shards"
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Cache manifest not found: "
                f"{self.manifest_path}"
            )

        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Cache index not found: "
                f"{self.index_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            self.manifest = json.load(
                handle
            )

        if (
            self.manifest.get(
                "cache_schema"
            )
            != CACHE_SCHEMA
        ):
            raise ValueError(
                "Unsupported Fakeddit cache schema."
            )

    @property
    def sample_count(
        self,
    ) -> int:
        return int(
            self.manifest[
                "sample_count"
            ]
        )

    @property
    def text_dimension(
        self,
    ) -> int:
        return int(
            self.manifest[
                "text_dimension"
            ]
        )

    @property
    def vision_dimension(
        self,
    ) -> int:
        return int(
            self.manifest[
                "vision_dimension"
            ]
        )

    def iter_shards(
        self,
    ) -> Iterator[
        Dict
    ]:

        shard_count = int(
            self.manifest[
                "shard_count"
            ]
        )

        for shard_number in range(
            shard_count
        ):

            shard_path = (
                self.shard_root
                / (
                    f"shard_"
                    f"{shard_number:05d}.pt"
                )
            )

            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Missing cache shard: "
                    f"{shard_path}"
                )

            yield torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

    def iter_batches(
        self,
        batch_size: int,
    ) -> Iterator[
        BinaryIntegrityBatch
    ]:
        """
        Stream cached representations as BinaryIntegrityBatch objects.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        text_buffer: List[
            torch.Tensor
        ] = []

        vision_buffer: List[
            torch.Tensor
        ] = []

        target_buffer: List[int] = []
        id_buffer: List[str] = []

        for shard in self.iter_shards():

            sample_ids = shard[
                "sample_ids"
            ]

            text_embeddings = shard[
                "text_embeddings"
            ]

            vision_embeddings = shard[
                "vision_embeddings"
            ]

            integrity_targets = shard[
                "integrity_targets"
            ]

            shard_size = len(
                sample_ids
            )

            for index in range(
                shard_size
            ):

                text_buffer.append(
                    text_embeddings[
                        index
                    ]
                )

                vision_buffer.append(
                    vision_embeddings[
                        index
                    ]
                )

                target_buffer.append(
                    int(
                        integrity_targets[
                            index
                        ].item()
                    )
                )

                id_buffer.append(
                    sample_ids[
                        index
                    ]
                )

                if (
                    len(id_buffer)
                    == batch_size
                ):

                    yield (
                        self._make_batch(
                            text_buffer,
                            vision_buffer,
                            target_buffer,
                            id_buffer,
                        )
                    )

                    text_buffer = []
                    vision_buffer = []
                    target_buffer = []
                    id_buffer = []

        if id_buffer:

            yield self._make_batch(
                text_buffer,
                vision_buffer,
                target_buffer,
                id_buffer,
            )

    def _make_batch(
        self,
        text_embeddings,
        vision_embeddings,
        targets,
        sample_ids,
    ) -> BinaryIntegrityBatch:

        batch = BinaryIntegrityBatch(
            text_embeddings=torch.stack(
                text_embeddings,
                dim=0,
            ),

            vision_embeddings=torch.stack(
                vision_embeddings,
                dim=0,
            ),

            integrity_targets=torch.tensor(
                targets,
                dtype=torch.long,
            ),

            sample_ids=list(
                sample_ids
            ),
        )

        batch.validate(
            text_dim=self.text_dimension,
            vision_dim=self.vision_dimension,
        )

        return batch
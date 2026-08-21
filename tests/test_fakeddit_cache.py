"""
Tests for the AEGIS Fakeddit frozen representation cache.

Version: 0.24.0
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from aegis.data.bridges import (
    FakedditRepresentedSample,
)

from aegis.data.cache import (
    CACHE_SCHEMA,
    FakedditRepresentationCache,
    FakedditRepresentationCacheWriter,
)


class TestFakedditRepresentationCache(
    unittest.TestCase
):

    def setUp(self):

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.cache_root = Path(
            self.temp_directory.name
        )

    def tearDown(self):

        self.temp_directory.cleanup()

    def make_item(
        self,
        index: int,
    ) -> FakedditRepresentedSample:

        native_label = (
            index % 2
        )

        integrity_target = (
            0
            if native_label == 1
            else 1
        )

        return FakedditRepresentedSample(
            sample_id=f"sample-{index}",

            text_embedding=torch.full(
                (768,),
                float(index),
            ),

            vision_embedding=torch.full(
                (512,),
                float(index),
            ),

            integrity_target=(
                integrity_target
            ),

            native_label=(
                native_label
            ),

            text_model="mock-text",

            vision_model="mock-vision",

            image_path=(
                f"sample-{index}.jpg"
            ),
        )

    def test_cache_write(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
                shard_size=2,
            )
        )

        manifest = writer.write(
            [
                self.make_item(0),
                self.make_item(1),
                self.make_item(2),
            ]
        )

        self.assertEqual(
            manifest.sample_count,
            3,
        )

        self.assertEqual(
            manifest.shard_count,
            2,
        )

        self.assertTrue(
            (
                self.cache_root
                / "cache_manifest.json"
            ).is_file()
        )

        self.assertTrue(
            (
                self.cache_root
                / "index.jsonl"
            ).is_file()
        )

    def test_manifest_schema(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
            )
        )

        writer.write(
            [
                self.make_item(0)
            ]
        )

        with (
            self.cache_root
            / "cache_manifest.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as handle:

            manifest = json.load(
                handle
            )

        self.assertEqual(
            manifest[
                "cache_schema"
            ],
            CACHE_SCHEMA,
        )

        self.assertEqual(
            manifest[
                "text_dimension"
            ],
            768,
        )

        self.assertEqual(
            manifest[
                "vision_dimension"
            ],
            512,
        )

    def test_cache_read(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
                shard_size=2,
            )
        )

        writer.write(
            [
                self.make_item(i)
                for i in range(5)
            ]
        )

        cache = (
            FakedditRepresentationCache(
                self.cache_root
            )
        )

        self.assertEqual(
            cache.sample_count,
            5,
        )

        shards = list(
            cache.iter_shards()
        )

        self.assertEqual(
            len(shards),
            3,
        )

    def test_cached_batches(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
                shard_size=3,
            )
        )

        writer.write(
            [
                self.make_item(i)
                for i in range(5)
            ]
        )

        cache = (
            FakedditRepresentationCache(
                self.cache_root
            )
        )

        batches = list(
            cache.iter_batches(
                batch_size=2
            )
        )

        self.assertEqual(
            len(batches),
            3,
        )

        self.assertEqual(
            batches[0].batch_size,
            2,
        )

        self.assertEqual(
            batches[1].batch_size,
            2,
        )

        self.assertEqual(
            batches[2].batch_size,
            1,
        )

        self.assertEqual(
            batches[0].sample_ids,
            [
                "sample-0",
                "sample-1",
            ],
        )

    def test_targets_preserved(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
            )
        )

        writer.write(
            [
                self.make_item(0),
                self.make_item(1),
            ]
        )

        cache = (
            FakedditRepresentationCache(
                self.cache_root
            )
        )

        batch = next(
            cache.iter_batches(
                batch_size=2
            )
        )

        self.assertEqual(
            batch.integrity_targets.tolist(),
            [1, 0],
        )

    def test_empty_cache_rejected(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
            )
        )

        with self.assertRaises(
            ValueError
        ):
            writer.write(
                []
            )

    def test_invalid_shard_size(
        self
    ):

        with self.assertRaises(
            ValueError
        ):
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
                shard_size=0,
            )

    def test_invalid_batch_size(
        self
    ):

        writer = (
            FakedditRepresentationCacheWriter(
                cache_root=self.cache_root,
                split="train",
            )
        )

        writer.write(
            [
                self.make_item(0)
            ]
        )

        cache = (
            FakedditRepresentationCache(
                self.cache_root
            )
        )

        with self.assertRaises(
            ValueError
        ):
            list(
                cache.iter_batches(
                    batch_size=0
                )
            )


if __name__ == "__main__":
    unittest.main()
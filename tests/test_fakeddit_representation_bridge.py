"""
Tests for the AEGIS Fakeddit representation bridge.

Version: 0.24.0
"""

import tempfile
import unittest
from pathlib import Path

import torch

from aegis.data.adapters.empirical import (
    EmpiricalSample,
    ImageStatus,
)
from aegis.data.bridges import (
    FakedditRepresentationBridge,
    empirical_to_canonical,
)
from aegis.representation.output import (
    TextRepresentation,
)
from aegis.representation.vision_output import (
    VisionRepresentation,
)


class MockTextEncoder:

    def encode(
        self,
        sample,
    ):
        return TextRepresentation(
            sample_id=sample.sample_id,
            embedding=torch.ones(
                768
            ),
            language=sample.language,
            model_name="mock-text",
            dimension=768,
            text=sample.text,
        )


class MockVisionEncoder:

    def encode(
        self,
        sample,
    ):
        return VisionRepresentation(
            sample_id=sample.sample_id,
            embedding=torch.ones(
                512
            ),
            model_name="mock-vision",
            dimension=512,
            image_path=sample.image_path,
        )


class TestFakedditRepresentationBridge(
    unittest.TestCase
):

    def setUp(self):
        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.image_path = (
            Path(
                self.temp_directory.name
            )
            / "sample.jpg"
        )

        self.image_path.write_bytes(
            b"test-image-placeholder"
        )

        self.bridge = (
            FakedditRepresentationBridge(
                text_encoder=(
                    MockTextEncoder()
                ),
                vision_encoder=(
                    MockVisionEncoder()
                ),
            )
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def make_sample(
        self,
        sample_id="sample-1",
        native_label=1,
    ):
        return EmpiricalSample(
            sample_id=sample_id,
            dataset="fakeddit",
            split="train",
            text="Example Fakeddit title",
            image_path=self.image_path,
            image_status=(
                ImageStatus.AVAILABLE
            ),
            native_labels={
                "2_way_label": (
                    native_label
                ),
                "3_way_label": 0,
                "6_way_label": 0,
            },
            metadata={
                "subreddit": "example",
            },
            provenance={
                "dataset": "Fakeddit",
            },
        )

    def test_empirical_to_canonical(self):
        canonical = (
            empirical_to_canonical(
                self.make_sample()
            )
        )

        self.assertEqual(
            canonical.sample_id,
            "sample-1",
        )

        self.assertTrue(
            canonical.has_text
        )

        self.assertTrue(
            canonical.has_image
        )

        self.assertTrue(
            canonical.is_multimodal
        )

        self.assertEqual(
            canonical.language,
            "en",
        )

    def test_true_target_mapping(self):
        represented = (
            self.bridge.represent(
                self.make_sample(
                    native_label=1
                )
            )
        )

        self.assertEqual(
            represented.integrity_target,
            0,
        )

        self.assertEqual(
            represented.native_label,
            1,
        )

    def test_fake_target_mapping(self):
        represented = (
            self.bridge.represent(
                self.make_sample(
                    native_label=0
                )
            )
        )

        self.assertEqual(
            represented.integrity_target,
            1,
        )

        self.assertEqual(
            represented.native_label,
            0,
        )

    def test_embedding_dimensions(self):
        represented = (
            self.bridge.represent(
                self.make_sample()
            )
        )

        self.assertEqual(
            represented.text_embedding.shape,
            torch.Size([768]),
        )

        self.assertEqual(
            represented.vision_embedding.shape,
            torch.Size([512]),
        )

    def test_build_batch(self):
        samples = [
            self.make_sample(
                sample_id="a",
                native_label=1,
            ),
            self.make_sample(
                sample_id="b",
                native_label=0,
            ),
        ]

        batch = (
            self.bridge.build_batch(
                samples
            )
        )

        self.assertEqual(
            batch.text_embeddings.shape,
            torch.Size(
                [2, 768]
            ),
        )

        self.assertEqual(
            batch.vision_embeddings.shape,
            torch.Size(
                [2, 512]
            ),
        )

        self.assertEqual(
            batch.integrity_targets.tolist(),
            [0, 1],
        )

        self.assertEqual(
            batch.sample_ids,
            ["a", "b"],
        )

    def test_iter_batches(self):
        samples = [
            self.make_sample(
                sample_id=f"s{i}",
                native_label=(
                    i % 2
                ),
            )
            for i in range(5)
        ]

        batches = list(
            self.bridge.iter_batches(
                samples,
                batch_size=2,
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

    def test_empty_batch_fails(self):
        with self.assertRaises(
            ValueError
        ):
            self.bridge.build_batch(
                []
            )

    def test_invalid_batch_size_fails(self):
        with self.assertRaises(
            ValueError
        ):
            list(
                self.bridge.iter_batches(
                    [
                        self.make_sample()
                    ],
                    batch_size=0,
                )
            )

    def test_missing_image_rejected(self):
        sample = self.make_sample()

        sample.image_path = None
        sample.image_status = (
            ImageStatus.MISSING
        )

        with self.assertRaises(
            ValueError
        ):
            empirical_to_canonical(
                sample
            )

    def test_wrong_text_dimension_rejected(
        self
    ):
        class WrongTextEncoder:

            def encode(
                self,
                sample,
            ):
                return TextRepresentation(
                    sample_id=(
                        sample.sample_id
                    ),
                    embedding=torch.ones(
                        10
                    ),
                    language="en",
                    model_name="wrong",
                    dimension=10,
                )

        bridge = (
            FakedditRepresentationBridge(
                text_encoder=(
                    WrongTextEncoder()
                ),
                vision_encoder=(
                    MockVisionEncoder()
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            bridge.represent(
                self.make_sample()
            )


if __name__ == "__main__":
    unittest.main()
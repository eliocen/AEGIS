"""
Tests for AEGIS visual representation learning.

Version: 0.8.0
"""

import unittest

from aegis.preprocessing import (
    CanonicalSample,
)

from aegis.representation import (
    VisionEncoder,
    VisionRepresentation,
    VisionRepresentationLayer,
)


class MockVisionEncoder(
    VisionEncoder
):
    """
    Lightweight visual encoder used for unit tests.

    No neural model download is required.
    """

    def encode(
        self,
        sample: CanonicalSample,
    ) -> VisionRepresentation:

        if not sample.has_image:
            raise ValueError(
                "Image is required."
            )

        embedding = [
            0.1,
            0.2,
            0.3,
            0.4,
        ]

        return VisionRepresentation(
            sample_id=sample.sample_id,
            embedding=embedding,
            model_name="mock-vision-model",
            dimension=4,
            image_path=sample.image_path,
        )


class TestVisionRepresentation(
    unittest.TestCase
):

    def test_representation_creation(
        self,
    ):

        representation = (
            VisionRepresentation(
                sample_id="IMG1",
                embedding=[
                    0.1,
                    0.2,
                ],
                model_name="test",
                dimension=2,
                image_path="example.jpg",
            )
        )

        self.assertEqual(
            representation.dimension,
            2,
        )

        self.assertEqual(
            representation.image_path,
            "example.jpg",
        )


class TestVisionRepresentationLayer(
    unittest.TestCase
):

    def test_visual_encoding(
        self,
    ):

        layer = (
            VisionRepresentationLayer(
                encoder=MockVisionEncoder()
            )
        )

        sample = CanonicalSample(
            sample_id="IMG1",
            image_path="example.jpg",
            has_image=True,
        )

        result = layer.process(
            sample
        )

        self.assertEqual(
            result.dimension,
            4,
        )

        self.assertEqual(
            result.sample_id,
            "IMG1",
        )

    def test_missing_image_fails(
        self,
    ):

        layer = (
            VisionRepresentationLayer(
                encoder=MockVisionEncoder()
            )
        )

        sample = CanonicalSample(
            sample_id="TEXT1",
            text="Text only",
            has_text=True,
            has_image=False,
        )

        with self.assertRaises(
            ValueError
        ):

            layer.process(
                sample
            )


if __name__ == "__main__":
    unittest.main()
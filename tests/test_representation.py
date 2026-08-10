"""
Tests for AEGIS Layer 3:
Multilingual Representation Learning.

Version: 0.7.0
"""

import unittest

from aegis.preprocessing import (
    CanonicalSample,
)

from aegis.representation import (
    MultilingualRepresentationLayer,
    TextEncoder,
    TextRepresentation,
    resolve_device,
)


class MockTextEncoder(TextEncoder):
    """
    Lightweight encoder used for unit testing.

    No neural model download is required.
    """

    def encode(
        self,
        sample: CanonicalSample,
    ) -> TextRepresentation:

        if not sample.has_text:
            raise ValueError(
                "Text is required."
            )

        embedding = [
            0.1,
            0.2,
            0.3,
            0.4,
        ]

        return TextRepresentation(
            sample_id=sample.sample_id,
            embedding=embedding,
            language=sample.language,
            model_name="mock-multilingual-model",
            dimension=4,
            text=sample.text,
        )


class TestDeviceResolution(unittest.TestCase):

    def test_cpu_resolution(self):

        self.assertEqual(
            resolve_device("cpu"),
            "cpu",
        )


class TestTextRepresentation(unittest.TestCase):

    def test_representation_creation(self):

        output = TextRepresentation(
            sample_id="001",
            embedding=[
                0.1,
                0.2,
            ],
            language="en",
            model_name="test",
            dimension=2,
        )

        self.assertEqual(
            output.dimension,
            2,
        )


class TestRepresentationLayer(
    unittest.TestCase
):

    def test_multilingual_encoding(self):

        layer = (
            MultilingualRepresentationLayer(
                encoder=MockTextEncoder()
            )
        )

        english = CanonicalSample(
            sample_id="EN1",
            text="Information integrity",
            language="en",
            has_text=True,
        )

        chinese = CanonicalSample(
            sample_id="ZH1",
            text="信息完整性",
            language="zh",
            has_text=True,
        )

        english_result = layer.process(
            english
        )

        chinese_result = layer.process(
            chinese
        )

        self.assertEqual(
            english_result.dimension,
            4,
        )

        self.assertEqual(
            chinese_result.language,
            "zh",
        )

        self.assertEqual(
            chinese_result.text,
            "信息完整性",
        )

    def test_missing_text_fails(self):

        layer = (
            MultilingualRepresentationLayer(
                encoder=MockTextEncoder()
            )
        )

        sample = CanonicalSample(
            sample_id="IMG1",
            language="en",
            has_text=False,
        )

        with self.assertRaises(
            ValueError
        ):
            layer.process(
                sample
            )


if __name__ == "__main__":
    unittest.main()
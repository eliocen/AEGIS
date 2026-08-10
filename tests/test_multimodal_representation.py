import unittest

from aegis.preprocessing import (
    CanonicalSample,
)

from aegis.representation import (
    ParallelMultimodalEncoder,
    ParallelMultimodalRepresentationLayer,
    TextEncoder,
    TextRepresentation,
    VisionEncoder,
    VisionRepresentation,
)


class MockTextEncoder(TextEncoder):

    def encode(self, sample):

        return TextRepresentation(
            sample_id=sample.sample_id,
            embedding=[0.1] * 768,
            language=sample.language,
            model_name="mock-text",
            dimension=768,
            text=sample.text,
        )


class MockVisionEncoder(VisionEncoder):

    def encode(self, sample):

        return VisionRepresentation(
            sample_id=sample.sample_id,
            embedding=[0.2] * 512,
            model_name="mock-vision",
            dimension=512,
            image_path=sample.image_path,
        )


class TestParallelMultimodalEncoding(
    unittest.TestCase
):

    def setUp(self):

        self.encoder = (
            ParallelMultimodalEncoder(
                text_encoder=MockTextEncoder(),
                vision_encoder=MockVisionEncoder(),
            )
        )

    def test_multimodal_bundle(self):

        sample = CanonicalSample(
            sample_id="MM1",
            text="Information integrity",
            image_path="image.jpg",
            language="en",
            has_text=True,
            has_image=True,
            is_multimodal=True,
        )

        bundle = self.encoder.encode(
            sample
        )

        self.assertTrue(
            bundle.text_available
        )

        self.assertTrue(
            bundle.vision_available
        )

        self.assertTrue(
            bundle.is_multimodal
        )

        self.assertEqual(
            bundle.text.dimension,
            768,
        )

        self.assertEqual(
            bundle.vision.dimension,
            512,
        )

    def test_text_only_bundle(self):

        sample = CanonicalSample(
            sample_id="TXT1",
            text="Text only",
            language="en",
            has_text=True,
            has_image=False,
        )

        bundle = self.encoder.encode(
            sample
        )

        self.assertTrue(
            bundle.text_available
        )

        self.assertFalse(
            bundle.vision_available
        )

        self.assertFalse(
            bundle.is_multimodal
        )

    def test_pipeline_layer(self):

        layer = (
            ParallelMultimodalRepresentationLayer(
                encoder=self.encoder
            )
        )

        sample = CanonicalSample(
            sample_id="MM2",
            text="Multimodal test",
            image_path="image.jpg",
            language="en",
            has_text=True,
            has_image=True,
            is_multimodal=True,
        )

        result = layer.process(
            sample
        )

        self.assertEqual(
            result.sample_id,
            "MM2",
        )

        self.assertTrue(
            result.is_multimodal
        )


if __name__ == "__main__":
    unittest.main()
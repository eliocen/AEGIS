"""
Tests for AEGIS Layer 2:
Multilingual Multimodal Preprocessing.

Version: 0.6.0
"""

import unittest

from aegis.acquisition import (
    AcquisitionRecord,
)

from aegis.preprocessing import (
    CanonicalSample,
    InvalidSampleError,
    PreprocessingLayer,
    UnsupportedLabelError,
    normalize_label,
    normalize_language,
    normalize_text,
)


class TestTextNormalization(unittest.TestCase):

    def test_whitespace_normalization(self):
        text = "Hello    world\n\nAEGIS"

        result = normalize_text(text)

        self.assertEqual(
            result,
            "Hello world AEGIS",
        )

    def test_multilingual_text_preservation(self):
        text = "Hello 世界 مرحبا Uganda 🇺🇬"

        result = normalize_text(text)

        self.assertIn(
            "世界",
            result,
        )

        self.assertIn(
            "مرحبا",
            result,
        )


class TestLanguageNormalization(unittest.TestCase):

    def test_english(self):
        self.assertEqual(
            normalize_language("English"),
            "en",
        )

    def test_chinese(self):
        self.assertEqual(
            normalize_language("zh-CN"),
            "zh",
        )

    def test_unknown_language(self):
        self.assertEqual(
            normalize_language(None),
            "und",
        )


class TestLabelNormalization(unittest.TestCase):

    def test_misinformation(self):
        self.assertEqual(
            normalize_label("Misinfo"),
            "misinformation",
        )

    def test_hate_speech(self):
        self.assertEqual(
            normalize_label("Hate Speech"),
            "hate_speech",
        )

    def test_strict_unknown_label(self):
        with self.assertRaises(
            UnsupportedLabelError
        ):
            normalize_label(
                "unknown_class",
                strict=True,
            )


class TestPreprocessingLayer(unittest.TestCase):

    def setUp(self):
        self.processor = PreprocessingLayer()

    def test_text_only_sample(self):
        record = AcquisitionRecord(
            sample_id="001",
            text="  Example   information ",
            language="English",
            label="True",
        )

        sample = self.processor.process(
            record
        )

        self.assertIsInstance(
            sample,
            CanonicalSample,
        )

        self.assertEqual(
            sample.text,
            "Example information",
        )

        self.assertEqual(
            sample.language,
            "en",
        )

        self.assertEqual(
            sample.label,
            "true",
        )

        self.assertTrue(
            sample.has_text
        )

        self.assertFalse(
            sample.is_multimodal
        )

    def test_multimodal_sample(self):
        record = AcquisitionRecord(
            sample_id="002",
            text="Multimodal example",
            image_path="image.jpg",
            language="en",
            label="Disinformation",
        )

        sample = self.processor.process(
            record
        )

        self.assertTrue(
            sample.has_text
        )

        self.assertTrue(
            sample.has_image
        )

        self.assertTrue(
            sample.is_multimodal
        )

        self.assertEqual(
            sample.label,
            "disinformation",
        )

    def test_invalid_sample(self):
        record = AcquisitionRecord(
            sample_id="003",
            text="   ",
        )

        with self.assertRaises(
            InvalidSampleError
        ):
            self.processor.process(
                record
            )

    def test_metadata_normalization(self):
        record = AcquisitionRecord(
            sample_id="004",
            text="Metadata example",
            metadata={
                "Country Name": "Uganda",
                "Engagement Count": 500,
            },
        )

        sample = self.processor.process(
            record
        )

        self.assertEqual(
            sample.metadata["country_name"],
            "Uganda",
        )

        self.assertEqual(
            sample.metadata["engagement_count"],
            500,
        )


if __name__ == "__main__":
    unittest.main()

class TestPipelineIntegration(unittest.TestCase):

    def test_aegis_preprocessing_pipeline(self):

        from aegis import AEGIS

        framework = AEGIS()

        framework.add_layer(
            PreprocessingLayer()
        )

        record = AcquisitionRecord(
            sample_id="PIPE-001",
            text="   Multilingual   information   ",
            language="English",
            label="Misinformation",
        )

        result = framework.analyze(
            record
        )

        self.assertIsInstance(
            result,
            CanonicalSample,
        )

        self.assertEqual(
            result.text,
            "Multilingual information",
        )

        self.assertEqual(
            result.language,
            "en",
        )

        self.assertEqual(
            result.label,
            "misinformation",
        )
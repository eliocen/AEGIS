"""
Tests for AEGIS empirical image validation.

Version: 0.24.0
"""

import tempfile
import unittest

from pathlib import Path

from PIL import Image

from aegis.data.image_validation import (
    validate_decodable_image,
)


class TestImageValidation(
    unittest.TestCase
):

    def setUp(
        self
    ):

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_directory.name
        )

    def tearDown(
        self
    ):

        self.temp_directory.cleanup()

    def test_valid_image(
        self
    ):

        image_path = (
            self.root
            / "valid.jpg"
        )

        image = Image.new(
            "RGB",
            (
                32,
                32,
            ),
        )

        image.save(
            image_path,
            format="JPEG",
        )

        result = (
            validate_decodable_image(
                image_path
            )
        )

        self.assertTrue(
            result.valid
        )

        self.assertTrue(
            result.decodable
        )

        self.assertEqual(
            result.width,
            32,
        )

        self.assertEqual(
            result.height,
            32,
        )

        self.assertEqual(
            result.format,
            "JPEG",
        )

    def test_missing_image(
        self
    ):

        result = (
            validate_decodable_image(
                self.root
                / "missing.jpg"
            )
        )

        self.assertFalse(
            result.valid
        )

        self.assertEqual(
            result.error_type,
            "file_not_found",
        )

    def test_fake_jpeg_rejected(
        self
    ):

        path = (
            self.root
            / "fake.jpg"
        )

        path.write_text(
            "<html>not an image</html>",
            encoding="utf-8",
        )

        result = (
            validate_decodable_image(
                path
            )
        )

        self.assertFalse(
            result.valid
        )

        self.assertFalse(
            result.decodable
        )

        self.assertEqual(
            result.error_type,
            "unidentified_image",
        )

    def test_empty_file_rejected(
        self
    ):

        path = (
            self.root
            / "empty.jpg"
        )

        path.write_bytes(
            b""
        )

        result = (
            validate_decodable_image(
                path
            )
        )

        self.assertFalse(
            result.valid
        )

        self.assertFalse(
            result.decodable
        )

    def test_result_dictionary(
        self
    ):

        path = (
            self.root
            / "valid.png"
        )

        image = Image.new(
            "RGB",
            (
                10,
                20,
            ),
        )

        image.save(
            path,
            format="PNG",
        )

        result = (
            validate_decodable_image(
                path
            )
        )

        payload = (
            result.as_dict()
        )

        self.assertTrue(
            payload[
                "valid"
            ]
        )

        self.assertEqual(
            payload[
                "format"
            ],
            "PNG",
        )

        self.assertEqual(
            payload[
                "width"
            ],
            10,
        )

        self.assertEqual(
            payload[
                "height"
            ],
            20,
        )


if __name__ == "__main__":
    unittest.main()
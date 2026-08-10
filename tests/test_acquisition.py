"""
Tests for AEGIS Layer 1:
Multisource Data Acquisition.

Version: 0.5.0
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from aegis.acquisition import (
    AcquisitionRecord,
    AcquisitionValidationError,
    StructuredFileDataSource,
    validate_record,
)


class TestAcquisitionRecord(unittest.TestCase):

    def test_text_record(self):
        record = AcquisitionRecord(
            sample_id="001",
            text="Example content",
            language="en",
        )

        self.assertTrue(record.has_text())
        self.assertFalse(record.has_image())
        self.assertFalse(record.is_multimodal())

    def test_multimodal_record(self):
        record = AcquisitionRecord(
            sample_id="002",
            text="Example content",
            image_path="example.jpg",
        )

        self.assertTrue(record.is_multimodal())


class TestValidation(unittest.TestCase):

    def test_missing_content_fails(self):
        record = AcquisitionRecord(
            sample_id="003",
        )

        with self.assertRaises(
            AcquisitionValidationError
        ):
            validate_record(record)

    def test_valid_text_record(self):
        record = AcquisitionRecord(
            sample_id="004",
            text="Valid content",
        )

        validate_record(record)


class TestStructuredFileDataSource(unittest.TestCase):

    def test_csv_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.csv"

            with path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "id",
                        "content",
                        "lang",
                        "class",
                    ],
                )

                writer.writeheader()

                writer.writerow(
                    {
                        "id": "100",
                        "content": "Test information",
                        "lang": "en",
                        "class": "True",
                    }
                )

            source = StructuredFileDataSource(
                str(path),
                field_map={
                    "sample_id": "id",
                    "text": "content",
                    "language": "lang",
                    "label": "class",
                },
            )

            records = list(source.load())

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].sample_id,
                "100",
            )
            self.assertEqual(
                records[0].language,
                "en",
            )

    def test_json_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"

            data = [
                {
                    "sample_id": "200",
                    "text": "JSON information",
                    "language": "en",
                }
            ]

            with path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(data, file)

            source = StructuredFileDataSource(
                str(path)
            )

            records = list(source.load())

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0].sample_id,
                "200",
            )


if __name__ == "__main__":
    unittest.main()
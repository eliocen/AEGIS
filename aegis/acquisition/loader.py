"""
Generic file loaders for AEGIS Layer 1.

Supported formats:
- CSV
- JSON
- JSONL

Version: 0.5.0
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .base import DataSource
from .record import AcquisitionRecord
from .validators import validate_record


class StructuredFileDataSource(DataSource):
    """
    Generic loader for structured dataset files.

    Source columns may be mapped into the canonical AEGIS schema.
    """

    def __init__(
        self,
        path: str,
        field_map: Optional[Dict[str, str]] = None,
        validate: bool = True,
    ):
        self.path = Path(path)
        self.field_map = field_map or {}
        self.validate = validate

    def load(self) -> Iterable[AcquisitionRecord]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}"
            )

        extension = self.path.suffix.lower()

        if extension == ".csv":
            yield from self._load_csv()

        elif extension == ".json":
            yield from self._load_json()

        elif extension in {".jsonl", ".ndjson"}:
            yield from self._load_jsonl()

        else:
            raise ValueError(
                f"Unsupported dataset format: {extension}"
            )

    def _load_csv(self) -> Iterator[AcquisitionRecord]:
        with self.path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                yield self._convert(row)

    def _load_json(self) -> Iterator[AcquisitionRecord]:
        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            data = data.get("records", [data])

        if not isinstance(data, list):
            raise ValueError(
                "JSON dataset must contain a list of records."
            )

        for row in data:
            yield self._convert(row)

    def _load_jsonl(self) -> Iterator[AcquisitionRecord]:
        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)

                yield self._convert(row)

    def _source_value(
        self,
        row: Dict,
        canonical_field: str,
        default=None,
    ):
        source_field = self.field_map.get(
            canonical_field,
            canonical_field,
        )

        return row.get(source_field, default)

    def _convert(self, row: Dict) -> AcquisitionRecord:
        known_fields = {
            self.field_map.get(field, field)
            for field in {
                "sample_id",
                "text",
                "image_path",
                "language",
                "source",
                "platform",
                "timestamp",
                "label",
            }
        }

        metadata = {
            key: value
            for key, value in row.items()
            if key not in known_fields
        }

        record = AcquisitionRecord(
            sample_id=str(
                self._source_value(
                    row,
                    "sample_id",
                    "",
                )
            ),
            text=self._source_value(
                row,
                "text",
            ),
            image_path=self._source_value(
                row,
                "image_path",
            ),
            language=self._source_value(
                row,
                "language",
            ),
            source=self._source_value(
                row,
                "source",
            ),
            platform=self._source_value(
                row,
                "platform",
            ),
            timestamp=self._source_value(
                row,
                "timestamp",
            ),
            label=self._source_value(
                row,
                "label",
            ),
            metadata=metadata,
        )

        if self.validate:
            validate_record(record)

        return record
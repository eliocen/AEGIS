"""
Fakeddit empirical dataset adapter for AEGIS.

The adapter preserves Fakeddit's native labels and does not
silently convert them into the AEGIS Information Integrity
taxonomy.

Dataset:
    r/Fakeddit

Primary empirical subset:
    multimodal_only_samples
"""

from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import pandas as pd

from .empirical import (
    EmpiricalSample,
    ImageStatus,
)


FAKEDDIT_REQUIRED_COLUMNS = (
    "clean_title",
    "id",
    "image_url",
    "hasImage",
    "2_way_label",
    "3_way_label",
    "6_way_label",
)


FAKEDDIT_NATIVE_LABEL_COLUMNS = (
    "2_way_label",
    "3_way_label",
    "6_way_label",
)


FAKEDDIT_SPLIT_FILES = {
    "train": "multimodal_train.tsv",
    "validation": "multimodal_validate.tsv",
    "test": "multimodal_test_public.tsv",
}


class FakedditAdapter:
    """
    Adapter for the official Fakeddit multimodal-only dataset.

    Responsibilities
    ----------------
    - Read official Fakeddit TSV splits.
    - Use ``clean_title`` as the canonical text field.
    - Preserve native 2-way, 3-way, and 6-way labels.
    - Resolve local image files without enumerating the entire
      image directory.
    - Preserve dataset provenance and metadata.
    - Stream large split files incrementally.
    - Avoid performing AEGIS label harmonization inside the
      dataset adapter.

    Label harmonization is intentionally handled by a separate
    empirical research layer so that original dataset semantics
    remain auditable and reproducible.
    """

    dataset_name = "fakeddit"

    def __init__(
        self,
        dataset_root: Path | str,
        image_extensions: Optional[
            Sequence[str]
        ] = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_root:
            Root directory containing the local Fakeddit dataset.

            Expected structure::

                fakeddit/
                    multimodal_only_samples/
                        multimodal_train.tsv
                        multimodal_validate.tsv
                        multimodal_test_public.tsv

                    images/
                        public_image_set/

        image_extensions:
            Optional image extensions to test when resolving
            local images.

            The adapter performs targeted path checks rather than
            enumerating the large image directory.
        """

        self.dataset_root = Path(
            dataset_root
        )

        self.multimodal_root = (
            self.dataset_root
            / "multimodal_only_samples"
        )

        self.image_root = (
            self.dataset_root
            / "images"
            / "public_image_set"
        )

        self.image_extensions = tuple(
            image_extensions
            or (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            )
        )

    def validate_structure(
        self,
    ) -> None:
        """
        Validate the expected local Fakeddit directory structure.

        Raises
        ------
        FileNotFoundError
            If required directories or split files are missing.
        """

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                "Fakeddit dataset root does not exist: "
                f"{self.dataset_root}"
            )

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                "Fakeddit dataset root is not a directory: "
                f"{self.dataset_root}"
            )

        if not self.multimodal_root.exists():
            raise FileNotFoundError(
                "Fakeddit multimodal_only_samples directory "
                "does not exist: "
                f"{self.multimodal_root}"
            )

        if not self.multimodal_root.is_dir():
            raise FileNotFoundError(
                "Fakeddit multimodal_only_samples path is not "
                "a directory: "
                f"{self.multimodal_root}"
            )

        if not self.image_root.exists():
            raise FileNotFoundError(
                "Fakeddit image directory does not exist: "
                f"{self.image_root}"
            )

        if not self.image_root.is_dir():
            raise FileNotFoundError(
                "Fakeddit image path is not a directory: "
                f"{self.image_root}"
            )

        for filename in (
            FAKEDDIT_SPLIT_FILES.values()
        ):

            path = (
                self.multimodal_root
                / filename
            )

            if not path.exists():
                raise FileNotFoundError(
                    "Missing Fakeddit split file: "
                    f"{path}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    "Fakeddit split path is not a file: "
                    f"{path}"
                )

    @staticmethod
    def canonical_split(
        split: str,
    ) -> str:
        """
        Normalize a split name to the canonical AEGIS split name.

        Supported canonical names
        -------------------------
        train
        validation
        test
        """

        normalized = (
            str(split)
            .strip()
            .lower()
        )

        aliases = {
            "train": "train",
            "training": "train",

            "val": "validation",
            "valid": "validation",
            "validate": "validation",
            "validation": "validation",

            "test": "test",
            "test_public": "test",
            "public_test": "test",
        }

        if normalized not in aliases:
            raise ValueError(
                "Unknown Fakeddit split: "
                f"{split}"
            )

        return aliases[
            normalized
        ]

    def split_path(
        self,
        split: str,
    ) -> Path:
        """
        Return the local TSV path for a Fakeddit split.
        """

        canonical = (
            self.canonical_split(
                split
            )
        )

        return (
            self.multimodal_root
            / FAKEDDIT_SPLIT_FILES[
                canonical
            ]
        )

    def inspect_schema(
        self,
        split: str,
    ) -> Sequence[str]:
        """
        Inspect a Fakeddit split schema without loading data rows.

        Returns
        -------
        Sequence[str]
            Column names in source order.

        Raises
        ------
        ValueError
            If required columns are absent.
        """

        path = self.split_path(
            split
        )

        frame = pd.read_csv(
            path,
            sep="\t",
            nrows=0,
        )

        columns = list(
            frame.columns
        )

        missing = [
            column
            for column
            in FAKEDDIT_REQUIRED_COLUMNS
            if column not in columns
        ]

        if missing:
            raise ValueError(
                "Fakeddit split is missing required columns: "
                + ", ".join(
                    missing
                )
            )

        return columns

    def resolve_image(
        self,
        sample_id: str,
    ) -> Optional[Path]:
        """
        Resolve a Fakeddit sample ID to a local image.

        This method deliberately avoids directory enumeration.

        It checks only the expected candidate paths:

            <sample_id>.jpg
            <sample_id>.jpeg
            <sample_id>.png
            <sample_id>.webp

        This is important because the official Fakeddit image
        directory can contain a very large number of files.
        """

        sample_id = (
            str(sample_id)
            .strip()
        )

        if not sample_id:
            return None

        for extension in (
            self.image_extensions
        ):

            candidate = (
                self.image_root
                / f"{sample_id}{extension}"
            )

            if candidate.is_file():
                return candidate

        return None

    def image_status(
        self,
        sample_id: str,
        has_image: bool = True,
    ) -> ImageStatus:
        """
        Determine local image availability for one sample.
        """

        if not has_image:
            return (
                ImageStatus.NOT_APPLICABLE
            )

        image_path = (
            self.resolve_image(
                sample_id
            )
        )

        if image_path is None:
            return (
                ImageStatus.MISSING
            )

        return (
            ImageStatus.AVAILABLE
        )

    @staticmethod
    def _safe_bool(
        value,
    ) -> bool:
        """
        Convert common Fakeddit boolean representations safely.
        """

        if pd.isna(
            value
        ):
            return False

        if isinstance(
            value,
            bool,
        ):
            return value

        normalized = (
            str(value)
            .strip()
            .lower()
        )

        return normalized in {
            "true",
            "1",
            "yes",
            "y",
        }

    @staticmethod
    def _safe_optional_value(
        value,
    ):
        """
        Convert pandas NaN values into standard Python None.
        """

        if pd.isna(
            value
        ):
            return None

        return value

    @staticmethod
    def _native_label(
        row,
        column: str,
    ) -> Optional[int]:
        """
        Read one native Fakeddit label while preserving
        missing labels as None.
        """

        value = row.get(
            column
        )

        if pd.isna(
            value
        ):
            return None

        return int(
            value
        )

    def row_to_sample(
        self,
        row,
        split: str,
    ) -> EmpiricalSample:
        """
        Convert one native Fakeddit TSV row into an AEGIS
        EmpiricalSample.

        No AEGIS taxonomy mapping is performed here.
        """

        canonical_split = (
            self.canonical_split(
                split
            )
        )

        sample_id_value = row.get(
            "id"
        )

        if pd.isna(
            sample_id_value
        ):
            raise ValueError(
                "Fakeddit row contains a missing sample ID."
            )

        sample_id = (
            str(
                sample_id_value
            )
            .strip()
        )

        if not sample_id:
            raise ValueError(
                "Fakeddit row contains an empty sample ID."
            )

        text_value = row.get(
            "clean_title"
        )

        if pd.isna(
            text_value
        ):
            text = ""
        else:
            text = (
                str(
                    text_value
                )
                .strip()
            )

        if not text:
            raise ValueError(
                "Fakeddit sample "
                f"{sample_id} "
                "has empty clean_title."
            )

        has_image = (
            self._safe_bool(
                row.get(
                    "hasImage",
                    False,
                )
            )
        )

        if has_image:

            resolved_image = (
                self.resolve_image(
                    sample_id
                )
            )

        else:

            resolved_image = None

        if not has_image:

            status = (
                ImageStatus.NOT_APPLICABLE
            )

        elif resolved_image is None:

            status = (
                ImageStatus.MISSING
            )

        else:

            status = (
                ImageStatus.AVAILABLE
            )

        native_labels = {
            column: self._native_label(
                row,
                column,
            )
            for column
            in FAKEDDIT_NATIVE_LABEL_COLUMNS
        }

        metadata: Dict[
            str,
            object,
        ] = {}

        metadata_columns = (
            "author",
            "created_utc",
            "domain",
            "image_url",
            "linked_submission_id",
            "num_comments",
            "score",
            "subreddit",
            "title",
            "upvote_ratio",
        )

        for column in (
            metadata_columns
        ):

            if column in row.index:

                metadata[
                    column
                ] = (
                    self._safe_optional_value(
                        row[
                            column
                        ]
                    )
                )

        metadata[
            "has_image_native"
        ] = has_image

        provenance = {
            "dataset": (
                "Fakeddit"
            ),

            "dataset_key": (
                self.dataset_name
            ),

            "subset": (
                "multimodal_only_samples"
            ),

            "split": (
                canonical_split
            ),

            "source_file": str(
                self.split_path(
                    canonical_split
                )
            ),

            "native_sample_id": (
                sample_id
            ),

            "text_field": (
                "clean_title"
            ),

            "image_store": str(
                self.image_root
            ),

            "label_status": (
                "native_unharmonized"
            ),
        }

        return EmpiricalSample(
            sample_id=sample_id,

            dataset=(
                self.dataset_name
            ),

            split=(
                canonical_split
            ),

            text=text,

            image_path=(
                resolved_image
            ),

            image_status=(
                status
            ),

            native_labels=(
                native_labels
            ),

            metadata=(
                metadata
            ),

            provenance=(
                provenance
            ),
        )

    def iter_samples(
        self,
        split: str,
        chunksize: int = 10_000,
        limit: Optional[int] = None,
    ) -> Iterator[
        EmpiricalSample
    ]:
        """
        Stream samples from a Fakeddit split.

        The source TSV is processed incrementally so that large
        empirical datasets do not need to be fully loaded into
        memory.

        The pandas TextFileReader is explicitly closed even when
        iteration stops early because ``limit`` is reached.

        Parameters
        ----------
        split:
            train, validation, or test.

        chunksize:
            Number of TSV rows loaded per pandas chunk.

        limit:
            Optional maximum number of samples to emit.

        Yields
        ------
        EmpiricalSample
            Canonical dataset-independent empirical samples.
        """

        if chunksize <= 0:
            raise ValueError(
                "chunksize must be greater than zero."
            )

        if (
            limit is not None
            and limit <= 0
        ):
            raise ValueError(
                "limit must be greater than zero "
                "when provided."
            )

        canonical_split = (
            self.canonical_split(
                split
            )
        )

        path = (
            self.split_path(
                canonical_split
            )
        )

        self.inspect_schema(
            canonical_split
        )

        emitted = 0

        with pd.read_csv(
            path,
            sep="\t",
            chunksize=chunksize,
        ) as reader:

            for chunk in reader:

                for _, row in (
                    chunk.iterrows()
                ):

                    sample = (
                        self.row_to_sample(
                            row,
                            canonical_split,
                        )
                    )

                    yield sample

                    emitted += 1

                    if (
                        limit is not None
                        and emitted >= limit
                    ):
                        return
# AEGIS audit write test
"""
AEGIS Fakeddit Full Dataset Audit
=================================

AEGIS Empirical Research Phase 1:
Real Dataset Discovery, Acquisition and Label Harmonization.

This module performs a complete integrity and reproducibility audit of
the official Fakeddit multimodal dataset splits.

The audit covers:

1. Dataset structure validation
2. Schema validation
3. Full split sample counts
4. Unique and duplicate sample IDs
5. Missing or empty clean_title values
6. Native 2-way, 3-way and 6-way label distributions
7. Missing and invalid native labels
8. Local image availability
9. Image availability by 6-way class
10. Domain distributions
11. Subreddit distributions
12. Strict multimodal eligibility
13. Cross-split sample-ID leakage
14. SHA-256 source-file fingerprints
15. Reproducible research manifests

Important
---------
This audit preserves the original Fakeddit labels.

It DOES NOT map Fakeddit labels into the AEGIS five-class taxonomy.
Label harmonization is handled separately after raw dataset integrity
has been established.
"""

from __future__ import annotations

import hashlib
import json

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd


# =====================================================================
# FAKEDDIT DATASET CONSTANTS
# =====================================================================

FAKEDDIT_NATIVE_LABEL_COLUMNS = (
    "2_way_label",
    "3_way_label",
    "6_way_label",
)


FAKEDDIT_REQUIRED_AUDIT_COLUMNS = (
    "id",
    "clean_title",
    "hasImage",
    "domain",
    "subreddit",
    "2_way_label",
    "3_way_label",
    "6_way_label",
)


FAKEDDIT_SPLIT_FILES = {
    "train": "multimodal_train.tsv",
    "validation": "multimodal_validate.tsv",
    "test": "multimodal_test_public.tsv",
}


VALID_NATIVE_LABELS = {
    "2_way_label": {0, 1},
    "3_way_label": {0, 1, 2},
    "6_way_label": {0, 1, 2, 3, 4, 5},
}


SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)


# =====================================================================
# DATA CLASSES
# =====================================================================


@dataclass
class SplitAuditResult:
    """
    Complete integrity statistics for one Fakeddit split.
    """

    split: str

    source_file: str
    source_sha256: str

    total_rows: int

    unique_ids: int
    duplicate_rows: int
    duplicate_unique_ids: int
    missing_ids: int

    missing_text: int
    empty_text: int

    native_label_distributions: Dict[str, Dict[str, int]]

    missing_native_labels: Dict[str, int]
    invalid_native_labels: Dict[str, int]

    image_available: int
    image_missing: int
    image_not_applicable: int
    image_coverage: float

    image_status_by_6_way_label: Dict[
        str,
        Dict[str, int],
    ]

    domain_distribution: Dict[str, int]
    subreddit_distribution: Dict[str, int]

    top_domains: Dict[str, int]
    top_subreddits: Dict[str, int]

    schema_columns: list[str]

    eligible_strict_multimodal: int
    excluded_strict_multimodal: int

    exclusion_reasons: Dict[str, int]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.
        """

        return asdict(self)


@dataclass
class CrossSplitAuditResult:
    """
    Exact sample-ID overlap diagnostics between official splits.
    """

    train_validation_overlap: int
    train_test_overlap: int
    validation_test_overlap: int

    train_validation_ids: list[str]
    train_test_ids: list[str]
    validation_test_ids: list[str]

    any_cross_split_leakage: bool

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.
        """

        return asdict(self)


@dataclass
class FakedditDatasetAudit:
    """
    Complete Fakeddit empirical dataset audit.
    """

    dataset: str
    dataset_root: str

    audit_version: str
    generated_at_utc: str

    splits: Dict[str, Dict[str, Any]]

    cross_split: Dict[str, Any]

    dataset_totals: Dict[str, Any]

    integrity_status: str

    methodology: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.
        """

        return asdict(self)


# =====================================================================
# GENERAL UTILITIES
# =====================================================================


def sha256_file(
    path: Path | str,
    buffer_size: int = 1024 * 1024,
) -> str:
    """
    Compute SHA-256 fingerprint for a file.

    Parameters
    ----------
    path:
        File path.

    buffer_size:
        Number of bytes read per iteration.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot compute SHA-256. File does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(buffer_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def _normalize_counter(
    counter: Counter,
) -> Dict[str, int]:
    """
    Convert Counter into deterministic JSON-safe dictionary.
    """

    return {
        str(key): int(value)
        for key, value in sorted(
            counter.items(),
            key=lambda item: str(item[0]),
        )
    }


def _safe_category(
    value: Any,
) -> str:
    """
    Normalize categorical metadata such as domain or subreddit.
    """

    if pd.isna(value):
        return "<missing>"

    text = str(value).strip()

    if not text:
        return "<empty>"

    return text


def _safe_label(
    value: Any,
) -> Optional[int]:
    """
    Convert native label to integer.

    Returns None when the value cannot be interpreted as a label.
    """

    if pd.isna(value):
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def _safe_bool(
    value: Any,
) -> bool:
    """
    Normalize common boolean representations.
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()

    return normalized in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


# =====================================================================
# FAKEDDIT FULL AUDITOR
# =====================================================================


class FakedditFullAuditor:
    """
    Full Fakeddit empirical integrity and reproducibility auditor.

    The implementation reads the TSV files in chunks so that the full
    training dataset does not need to be loaded into memory.

    Images are checked using direct sample-ID path lookup. The enormous
    public_image_set directory is therefore not recursively enumerated.
    """

    audit_version = "1.0"

    def __init__(
        self,
        dataset_root: Path | str,
        output_root: Path | str,
        chunksize: int = 25_000,
        top_k_categories: int = 25,
    ) -> None:

        if chunksize <= 0:
            raise ValueError(
                "chunksize must be greater than zero."
            )

        if top_k_categories <= 0:
            raise ValueError(
                "top_k_categories must be greater than zero."
            )

        self.dataset_root = Path(dataset_root)

        self.multimodal_root = (
            self.dataset_root
            / "multimodal_only_samples"
        )

        self.image_root = (
            self.dataset_root
            / "images"
            / "public_image_set"
        )

        self.output_root = Path(output_root)

        self.chunksize = int(chunksize)

        self.top_k_categories = int(
            top_k_categories
        )

    # =================================================================
    # DATASET STRUCTURE
    # =================================================================

    def canonical_split(
        self,
        split: str,
    ) -> str:
        """
        Normalize supported split aliases.
        """

        normalized = str(split).strip().lower()

        aliases = {
            "train": "train",
            "training": "train",

            "validation": "validation",
            "validate": "validation",
            "val": "validation",
            "dev": "validation",

            "test": "test",
            "testing": "test",
            "test_public": "test",
            "public_test": "test",
        }

        if normalized not in aliases:
            raise ValueError(
                f"Unsupported Fakeddit split: {split}"
            )

        return aliases[normalized]

    def split_path(
        self,
        split: str,
    ) -> Path:
        """
        Resolve an official multimodal Fakeddit split.
        """

        canonical = self.canonical_split(split)

        filename = FAKEDDIT_SPLIT_FILES[
            canonical
        ]

        return (
            self.multimodal_root
            / filename
        )

    def validate_structure(
        self,
    ) -> None:
        """
        Validate required Fakeddit directories and files.
        """

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Fakeddit dataset root does not exist: "
                f"{self.dataset_root}"
            )

        if not self.multimodal_root.is_dir():
            raise FileNotFoundError(
                "Fakeddit multimodal directory does not exist: "
                f"{self.multimodal_root}"
            )

        if not self.image_root.is_dir():
            raise FileNotFoundError(
                "Fakeddit image directory does not exist: "
                f"{self.image_root}"
            )

        for split in (
            "train",
            "validation",
            "test",
        ):

            path = self.split_path(split)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing Fakeddit {split} split: {path}"
                )

            schema = self.inspect_schema(
                split
            )

            missing_columns = [
                column
                for column
                in FAKEDDIT_REQUIRED_AUDIT_COLUMNS
                if column not in schema
            ]

            if missing_columns:
                raise ValueError(
                    f"Fakeddit {split} split is missing "
                    f"required columns: {missing_columns}"
                )

    def inspect_schema(
        self,
        split: str,
    ) -> list[str]:
        """
        Read only the TSV header and return its columns.
        """

        path = self.split_path(split)

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Fakeddit split: {path}"
            )

        frame = pd.read_csv(
            path,
            sep="\t",
            nrows=0,
        )

        return list(frame.columns)

    # =================================================================
    # IMAGE RESOLUTION
    # =================================================================

    def resolve_image(
        self,
        sample_id: str,
    ) -> Optional[Path]:
        """
        Resolve a local image using the native Fakeddit sample ID.

        No directory enumeration is performed.
        """

        sample_id = str(
            sample_id
        ).strip()

        if not sample_id:
            return None

        for extension in (
            SUPPORTED_IMAGE_EXTENSIONS
        ):

            candidate = (
                self.image_root
                / f"{sample_id}{extension}"
            )

            if candidate.is_file():
                return candidate

        return None

    def _resolve_image_status(
        self,
        sample_id: str,
        has_image: Any,
    ) -> str:
        """
        Resolve one sample's local image status.
        """

        native_has_image = _safe_bool(
            has_image
        )

        if not native_has_image:
            return "not_applicable"

        image_path = self.resolve_image(
            sample_id
        )

        if image_path is not None:
            return "available"

        return "missing"

    # =================================================================
    # SPLIT AUDIT
    # =================================================================

    def audit_split(
        self,
        split: str,
    ) -> Tuple[
        SplitAuditResult,
        Set[str],
        list[Dict[str, Any]],
    ]:
        """
        Audit one complete official Fakeddit split.

        Returns
        -------
        SplitAuditResult
            Aggregated split statistics.

        Set[str]
            Unique sample IDs for leakage analysis.

        list[dict]
            Strict-multimodal exclusion records.
        """

        canonical_split = (
            self.canonical_split(
                split
            )
        )

        path = self.split_path(
            canonical_split
        )

        schema = self.inspect_schema(
            canonical_split
        )

        missing_columns = [
            column
            for column
            in FAKEDDIT_REQUIRED_AUDIT_COLUMNS
            if column not in schema
        ]

        if missing_columns:
            raise ValueError(
                f"Cannot audit {canonical_split}. "
                f"Missing columns: {missing_columns}"
            )

        source_hash = sha256_file(
            path
        )

        total_rows = 0

        id_counts: Counter = Counter()

        missing_ids = 0

        missing_text = 0
        empty_text = 0

        label_counts: Dict[
            str,
            Counter,
        ] = {
            column: Counter()
            for column
            in FAKEDDIT_NATIVE_LABEL_COLUMNS
        }

        missing_labels = {
            column: 0
            for column
            in FAKEDDIT_NATIVE_LABEL_COLUMNS
        }

        invalid_labels = {
            column: 0
            for column
            in FAKEDDIT_NATIVE_LABEL_COLUMNS
        }

        image_status_counts: Counter = (
            Counter()
        )

        image_by_six_way: Dict[
            str,
            Counter,
        ] = {}

        domain_counts: Counter = Counter()

        subreddit_counts: Counter = Counter()

        unique_ids: Set[str] = set()

        exclusions: list[
            Dict[str, Any]
        ] = []

        reader = pd.read_csv(
            path,
            sep="\t",
            usecols=list(
                FAKEDDIT_REQUIRED_AUDIT_COLUMNS
            ),
              chunksize=self.chunksize,
            )

        required_columns = list(
            FAKEDDIT_REQUIRED_AUDIT_COLUMNS
        )

        for chunk in reader:


            chunk = chunk.loc[
                :,
                required_columns,
            ]

            for row in chunk.itertuples(
                index=False,
                name=None,
            ):

                row_dict = dict(
                    zip(
                        FAKEDDIT_REQUIRED_AUDIT_COLUMNS,
                        row,
                    )
                )

                total_rows += 1

                # =================================================
                # SAMPLE ID
                # =================================================

                raw_id = row_dict["id"]

                if pd.isna(raw_id):

                    sample_id = ""

                    missing_ids += 1

                else:

                    sample_id = str(
                        raw_id
                    ).strip()

                    if not sample_id:
                        missing_ids += 1

                if sample_id:

                    id_counts[
                        sample_id
                    ] += 1

                    unique_ids.add(
                        sample_id
                    )

                # =================================================
                # TEXT
                # =================================================

                raw_text = row_dict[
                    "clean_title"
                ]

                text_missing = pd.isna(
                    raw_text
                )

                text_empty = False

                if text_missing:

                    missing_text += 1

                else:

                    normalized_text = str(
                        raw_text
                    ).strip()

                    if not normalized_text:

                        empty_text += 1

                        text_empty = True

                # =================================================
                # NATIVE LABELS
                # =================================================

                native_labels: Dict[
                    str,
                    Optional[int],
                ] = {}

                for label_column in (
                    FAKEDDIT_NATIVE_LABEL_COLUMNS
                ):

                    raw_label = row_dict[
                        label_column
                    ]

                    label = _safe_label(
                        raw_label
                    )

                    native_labels[
                        label_column
                    ] = label

                    if label is None:

                        missing_labels[
                            label_column
                        ] += 1

                        continue

                    label_counts[
                        label_column
                    ][label] += 1

                    if (
                        label
                        not in
                        VALID_NATIVE_LABELS[
                            label_column
                        ]
                    ):

                        invalid_labels[
                            label_column
                        ] += 1

                # =================================================
                # DOMAIN
                # =================================================

                domain = _safe_category(
                    row_dict["domain"]
                )

                domain_counts[
                    domain
                ] += 1

                # =================================================
                # SUBREDDIT
                # =================================================

                subreddit = _safe_category(
                    row_dict["subreddit"]
                )

                subreddit_counts[
                    subreddit
                ] += 1

                # =================================================
                # IMAGE
                # =================================================

                if sample_id:

                    image_status = (
                        self._resolve_image_status(
                            sample_id=sample_id,
                            has_image=(
                                row_dict[
                                    "hasImage"
                                ]
                            ),
                        )
                    )

                else:

                    image_status = "missing"

                image_status_counts[
                    image_status
                ] += 1

                # =================================================
                # IMAGE STATUS BY 6-WAY LABEL
                # =================================================

                six_way_label = (
                    native_labels[
                        "6_way_label"
                    ]
                )

                if six_way_label is None:

                    six_key = "<missing>"

                else:

                    six_key = str(
                        six_way_label
                    )

                if six_key not in (
                    image_by_six_way
                ):

                    image_by_six_way[
                        six_key
                    ] = Counter()

                image_by_six_way[
                    six_key
                ][image_status] += 1

                # =================================================
                # STRICT MULTIMODAL ELIGIBILITY
                # =================================================

                exclusion_reasons = []

                if not sample_id:

                    exclusion_reasons.append(
                        "missing_sample_id"
                    )

                if text_missing:

                    exclusion_reasons.append(
                        "missing_text"
                    )

                if text_empty:

                    exclusion_reasons.append(
                        "empty_text"
                    )

                if image_status != "available":

                    exclusion_reasons.append(
                        "image_not_available"
                    )

                missing_any_label = any(
                    native_labels[column]
                    is None
                    for column
                    in FAKEDDIT_NATIVE_LABEL_COLUMNS
                )

                if missing_any_label:

                    exclusion_reasons.append(
                        "missing_native_label"
                    )

                invalid_any_label = any(
                    (
                        native_labels[column]
                        is not None
                        and native_labels[column]
                        not in VALID_NATIVE_LABELS[
                            column
                        ]
                    )
                    for column
                    in FAKEDDIT_NATIVE_LABEL_COLUMNS
                )

                if invalid_any_label:

                    exclusion_reasons.append(
                        "invalid_native_label"
                    )

                if exclusion_reasons:

                    exclusions.append(
                        {
                            "sample_id": (
                                sample_id
                                if sample_id
                                else None
                            ),

                            "split": (
                                canonical_split
                            ),

                            "reasons": (
                                exclusion_reasons
                            ),
                        }
                    )

        # =========================================================
        # DUPLICATE STATISTICS
        # =========================================================

        duplicate_ids = {
            sample_id: count
            for sample_id, count
            in id_counts.items()
            if count > 1
        }

        duplicate_rows = sum(
            count - 1
            for count
            in duplicate_ids.values()
        )

        duplicate_unique_ids = len(
            duplicate_ids
        )

        # =========================================================
        # IMAGE STATISTICS
        # =========================================================

        image_available = int(
            image_status_counts[
                "available"
            ]
        )

        image_missing = int(
            image_status_counts[
                "missing"
            ]
        )

        image_not_applicable = int(
            image_status_counts[
                "not_applicable"
            ]
        )

        image_coverage = (
            image_available
            / total_rows
            if total_rows
            else 0.0
        )

        # =========================================================
        # EXCLUSION STATISTICS
        # =========================================================

        exclusion_reason_counts: Counter = (
            Counter()
        )

        for exclusion in exclusions:

            for reason in (
                exclusion["reasons"]
            ):

                exclusion_reason_counts[
                    reason
                ] += 1

        excluded_count = len(
            exclusions
        )

        eligible_count = (
            total_rows
            - excluded_count
        )

        # =========================================================
        # SPLIT RESULT
        # =========================================================

        result = SplitAuditResult(
            split=canonical_split,

            source_file=str(
                path.resolve()
            ),

            source_sha256=source_hash,

            total_rows=int(
                total_rows
            ),

            unique_ids=int(
                len(unique_ids)
            ),

            duplicate_rows=int(
                duplicate_rows
            ),

            duplicate_unique_ids=int(
                duplicate_unique_ids
            ),

            missing_ids=int(
                missing_ids
            ),

            missing_text=int(
                missing_text
            ),

            empty_text=int(
                empty_text
            ),

            native_label_distributions={
                column: _normalize_counter(
                    label_counts[column]
                )
                for column
                in FAKEDDIT_NATIVE_LABEL_COLUMNS
            },

            missing_native_labels={
                column: int(count)
                for column, count
                in missing_labels.items()
            },

            invalid_native_labels={
                column: int(count)
                for column, count
                in invalid_labels.items()
            },

            image_available=(
                image_available
            ),

            image_missing=(
                image_missing
            ),

            image_not_applicable=(
                image_not_applicable
            ),

            image_coverage=float(
                image_coverage
            ),

            image_status_by_6_way_label={
                label: _normalize_counter(
                    counter
                )
                for label, counter
                in sorted(
                    image_by_six_way.items(),
                    key=lambda item: item[0],
                )
            },

            domain_distribution=(
                _normalize_counter(
                    domain_counts
                )
            ),

            subreddit_distribution=(
                _normalize_counter(
                    subreddit_counts
                )
            ),

            top_domains={
                str(key): int(value)
                for key, value
                in domain_counts.most_common(
                    self.top_k_categories
                )
            },

            top_subreddits={
                str(key): int(value)
                for key, value
                in subreddit_counts.most_common(
                    self.top_k_categories
                )
            },

            schema_columns=schema,

            eligible_strict_multimodal=int(
                eligible_count
            ),

            excluded_strict_multimodal=int(
                excluded_count
            ),

            exclusion_reasons=(
                _normalize_counter(
                    exclusion_reason_counts
                )
            ),

            metadata={
                "chunksize": (
                    self.chunksize
                ),

                "text_field": (
                    "clean_title"
                ),

                "strict_multimodal_rule": (
                    "valid sample ID + non-empty clean_title "
                    "+ locally available image "
                    "+ complete valid native labels"
                ),
            },
        )

        return (
            result,
            unique_ids,
            exclusions,
        )

    # =================================================================
    # CROSS-SPLIT LEAKAGE
    # =================================================================

    @staticmethod
    def audit_cross_split(
        train_ids: Set[str],
        validation_ids: Set[str],
        test_ids: Set[str],
    ) -> CrossSplitAuditResult:
        """
        Detect exact sample-ID overlap between official splits.
        """

        train_validation = sorted(
            train_ids.intersection(
                validation_ids
            )
        )

        train_test = sorted(
            train_ids.intersection(
                test_ids
            )
        )

        validation_test = sorted(
            validation_ids.intersection(
                test_ids
            )
        )

        any_leakage = bool(
            train_validation
            or train_test
            or validation_test
        )

        return CrossSplitAuditResult(
            train_validation_overlap=len(
                train_validation
            ),

            train_test_overlap=len(
                train_test
            ),

            validation_test_overlap=len(
                validation_test
            ),

            train_validation_ids=(
                train_validation
            ),

            train_test_ids=(
                train_test
            ),

            validation_test_ids=(
                validation_test
            ),

            any_cross_split_leakage=(
                any_leakage
            ),
        )

    # =================================================================
    # COMPLETE DATASET AUDIT
    # =================================================================

    def run(
        self,
    ) -> FakedditDatasetAudit:
        """
        Execute the complete Fakeddit audit.
        """

        self.validate_structure()

        results: Dict[
            str,
            SplitAuditResult,
        ] = {}

        id_sets: Dict[
            str,
            Set[str],
        ] = {}

        all_exclusions: Dict[
            str,
            list[Dict[str, Any]],
        ] = {}

        for split in (
            "train",
            "validation",
            "test",
        ):

            (
                split_result,
                split_ids,
                split_exclusions,
            ) = self.audit_split(
                split
            )

            results[
                split
            ] = split_result

            id_sets[
                split
            ] = split_ids

            all_exclusions[
                split
            ] = split_exclusions

        # =========================================================
        # CROSS-SPLIT ANALYSIS
        # =========================================================

        cross_split = (
            self.audit_cross_split(
                train_ids=(
                    id_sets["train"]
                ),

                validation_ids=(
                    id_sets["validation"]
                ),

                test_ids=(
                    id_sets["test"]
                ),
            )
        )

        # =========================================================
        # GLOBAL TOTALS
        # =========================================================

        total_rows = sum(
            result.total_rows
            for result
            in results.values()
        )

        total_available_images = sum(
            result.image_available
            for result
            in results.values()
        )

        total_missing_images = sum(
            result.image_missing
            for result
            in results.values()
        )

        total_not_applicable_images = sum(
            result.image_not_applicable
            for result
            in results.values()
        )

        total_excluded = sum(
            result.excluded_strict_multimodal
            for result
            in results.values()
        )

        total_eligible = sum(
            result.eligible_strict_multimodal
            for result
            in results.values()
        )

        total_duplicate_rows = sum(
            result.duplicate_rows
            for result
            in results.values()
        )

        total_missing_or_empty_text = sum(
            (
                result.missing_text
                + result.empty_text
            )
            for result
            in results.values()
        )

        total_missing_labels = sum(
            sum(
                result.missing_native_labels.values()
            )
            for result
            in results.values()
        )

        total_invalid_labels = sum(
            sum(
                result.invalid_native_labels.values()
            )
            for result
            in results.values()
        )

        # =========================================================
        # INTEGRITY STATUS
        # =========================================================

        problems_detected = any(
            (
                total_missing_images > 0,
                total_duplicate_rows > 0,
                total_missing_or_empty_text > 0,
                total_missing_labels > 0,
                total_invalid_labels > 0,
                cross_split.any_cross_split_leakage,
            )
        )

        integrity_status = (
            "review_required"
            if problems_detected
            else "passed"
        )

        generated_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # =========================================================
        # FINAL AUDIT OBJECT
        # =========================================================

        audit = FakedditDatasetAudit(
            dataset="Fakeddit",

            dataset_root=str(
                self.dataset_root.resolve()
            ),

            audit_version=(
                self.audit_version
            ),

            generated_at_utc=(
                generated_at
            ),

            splits={
                split: result.as_dict()
                for split, result
                in results.items()
            },

            cross_split=(
                cross_split.as_dict()
            ),

            dataset_totals={
                "total_rows": int(
                    total_rows
                ),

                "strict_multimodal_eligible": int(
                    total_eligible
                ),

                "strict_multimodal_excluded": int(
                    total_excluded
                ),

                "images_available": int(
                    total_available_images
                ),

                "images_missing": int(
                    total_missing_images
                ),

                "images_not_applicable": int(
                    total_not_applicable_images
                ),

                "duplicate_rows_within_splits": int(
                    total_duplicate_rows
                ),

                "missing_or_empty_text": int(
                    total_missing_or_empty_text
                ),

                "missing_native_labels": int(
                    total_missing_labels
                ),

                "invalid_native_labels": int(
                    total_invalid_labels
                ),

                "overall_image_coverage": float(
                    (
                        total_available_images
                        / total_rows
                    )
                    if total_rows
                    else 0.0
                ),
            },

            integrity_status=(
                integrity_status
            ),

            methodology={
                "dataset": (
                    "Fakeddit"
                ),

                "dataset_subset": (
                    "multimodal_only_samples"
                ),

                "text_field": (
                    "clean_title"
                ),

                "image_directory": (
                    "images/public_image_set"
                ),

                "image_resolution": (
                    "direct sample-ID path resolution"
                ),

                "native_labels_preserved": [
                    "2_way_label",
                    "3_way_label",
                    "6_way_label",
                ],

                "aegis_label_harmonization": (
                    "not performed during raw dataset audit"
                ),

                "official_splits_preserved": True,

                "cross_split_leakage_check": (
                    "exact native sample-ID intersection"
                ),

                "source_fingerprinting": (
                    "SHA-256 per official split TSV file"
                ),

                "strict_multimodal_eligibility": (
                    "valid sample ID + non-empty clean_title "
                    "+ locally available image "
                    "+ complete valid native labels"
                ),

                "image_directory_enumeration": False,

                "chunked_processing": True,

                "chunksize": (
                    self.chunksize
                ),
            },
        )

        # =========================================================
        # SAVE MANIFESTS
        # =========================================================

        self.save_outputs(
            audit=audit,
            exclusions=all_exclusions,
        )

        return audit

    # =================================================================
    # MANIFEST GENERATION
    # =================================================================

    def save_outputs(
        self,
        audit: FakedditDatasetAudit,
        exclusions: Dict[
            str,
            list[Dict[str, Any]],
        ],
    ) -> None:
        """
        Save reproducibility and audit artifacts.
        """

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataset_audit_path = (
            self.output_root
            / "dataset_audit.json"
        )

        split_statistics_path = (
            self.output_root
            / "split_statistics.json"
        )

        provenance_path = (
            self.output_root
            / "provenance.json"
        )

        exclusions_path = (
            self.output_root
            / "excluded_samples.jsonl"
        )

        # =========================================================
        # MASTER DATASET AUDIT
        # =========================================================

        with dataset_audit_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                audit.as_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        # =========================================================
        # SPLIT STATISTICS
        # =========================================================

        split_statistics = {}

        for split, data in (
            audit.splits.items()
        ):

            split_statistics[
                split
            ] = {
                "total_rows": (
                    data["total_rows"]
                ),

                "unique_ids": (
                    data["unique_ids"]
                ),

                "duplicate_rows": (
                    data["duplicate_rows"]
                ),

                "duplicate_unique_ids": (
                    data[
                        "duplicate_unique_ids"
                    ]
                ),

                "missing_ids": (
                    data["missing_ids"]
                ),

                "missing_text": (
                    data["missing_text"]
                ),

                "empty_text": (
                    data["empty_text"]
                ),

                "native_label_distributions": (
                    data[
                        "native_label_distributions"
                    ]
                ),

                "missing_native_labels": (
                    data[
                        "missing_native_labels"
                    ]
                ),

                "invalid_native_labels": (
                    data[
                        "invalid_native_labels"
                    ]
                ),

                "image_available": (
                    data["image_available"]
                ),

                "image_missing": (
                    data["image_missing"]
                ),

                "image_not_applicable": (
                    data[
                        "image_not_applicable"
                    ]
                ),

                "image_coverage": (
                    data["image_coverage"]
                ),

                "image_status_by_6_way_label": (
                    data[
                        "image_status_by_6_way_label"
                    ]
                ),

                "eligible_strict_multimodal": (
                    data[
                        "eligible_strict_multimodal"
                    ]
                ),

                "excluded_strict_multimodal": (
                    data[
                        "excluded_strict_multimodal"
                    ]
                ),

                "exclusion_reasons": (
                    data[
                        "exclusion_reasons"
                    ]
                ),

                "top_domains": (
                    data["top_domains"]
                ),

                "top_subreddits": (
                    data["top_subreddits"]
                ),
            }

        with split_statistics_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                split_statistics,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        # =========================================================
        # PROVENANCE
        # =========================================================

        provenance = {
            "dataset": (
                audit.dataset
            ),

            "dataset_root": (
                audit.dataset_root
            ),

            "audit_version": (
                audit.audit_version
            ),

            "generated_at_utc": (
                audit.generated_at_utc
            ),

            "integrity_status": (
                audit.integrity_status
            ),

            "methodology": (
                audit.methodology
            ),

            "source_files": {
                split: {
                    "path": (
                        data["source_file"]
                    ),

                    "sha256": (
                        data["source_sha256"]
                    ),

                    "total_rows": (
                        data["total_rows"]
                    ),
                }

                for split, data
                in audit.splits.items()
            },

            "cross_split": (
                audit.cross_split
            ),
        }

        with provenance_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                provenance,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        # =========================================================
        # EXCLUDED SAMPLE MANIFEST
        # =========================================================

        with exclusions_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for split in (
                "train",
                "validation",
                "test",
            ):

                for exclusion in (
                    exclusions.get(
                        split,
                        [],
                    )
                ):

                    handle.write(
                        json.dumps(
                            exclusion,
                            ensure_ascii=False,
                        )
                    )

                    handle.write("\n")
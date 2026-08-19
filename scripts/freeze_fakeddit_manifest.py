"""
AEGIS Empirical Research Phase 1

Freeze the audited Fakeddit dataset state and create
the first formal label-harmonization specification.

This script does not modify raw data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


AUDIT_PATH = Path(
    "data/manifests/fakeddit/dataset_audit.json"
)

PROVENANCE_PATH = Path(
    "data/manifests/fakeddit/provenance.json"
)

FROZEN_MANIFEST_PATH = Path(
    "data/manifests/fakeddit/frozen_manifest.json"
)

HARMONIZATION_PATH = Path(
    "data/manifests/fakeddit/harmonization_spec.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def save_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def build_frozen_manifest(
    audit: dict,
    provenance: dict,
) -> dict:

    if audit[
        "integrity_status"
    ] != "passed":
        raise RuntimeError(
            "Fakeddit cannot be frozen because "
            "the dataset audit did not pass."
        )

    splits = audit["splits"]

    return {
        "manifest_version": "1.0",

        "dataset": "Fakeddit",

        "dataset_subset": (
            "multimodal_only_samples"
        ),

        "frozen_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "audit_version": (
            audit["audit_version"]
        ),

        "integrity_status": (
            audit["integrity_status"]
        ),

        "text_field": (
            "clean_title"
        ),

        "image_store": (
            "data/raw/fakeddit/"
            "images/public_image_set"
        ),

        "official_splits_preserved": True,

        "split_counts": {
            split: data[
                "total_rows"
            ]
            for split, data
            in splits.items()
        },

        "total_samples": (
            audit[
                "dataset_totals"
            ][
                "total_rows"
            ]
        ),

        "strict_multimodal_eligible": (
            audit[
                "dataset_totals"
            ][
                "strict_multimodal_eligible"
            ]
        ),

        "strict_multimodal_excluded": (
            audit[
                "dataset_totals"
            ][
                "strict_multimodal_excluded"
            ]
        ),

        "image_coverage": (
            audit[
                "dataset_totals"
            ][
                "overall_image_coverage"
            ]
        ),

        "cross_split_leakage": (
            audit[
                "cross_split"
            ][
                "any_cross_split_leakage"
            ]
        ),

        "native_label_distributions": {
            split: data[
                "native_label_distributions"
            ]
            for split, data
            in splits.items()
        },

        "source_files": (
            provenance[
                "source_files"
            ]
        ),

        "eligibility_policy": (
            audit[
                "methodology"
            ][
                "strict_multimodal_eligibility"
            ]
        ),

        "label_policy": {
            "native_labels_preserved": True,
            "automatic_aegis_mapping": False,
            "harmonization_required": True,
        },

        "research_status": (
            "frozen_for_empirical_use"
        ),
    }


def build_harmonization_spec() -> dict:

    return {
        "specification_version": "1.0",

        "dataset": "Fakeddit",

        "purpose": (
            "Define how Fakeddit native labels may be used "
            "within AEGIS without introducing unsupported "
            "semantic assumptions."
        ),

        "principles": [
            (
                "Native Fakeddit labels must remain preserved "
                "for reproducible benchmark experiments."
            ),
            (
                "Fakeddit fake-news labels must not be "
                "automatically interpreted as misinformation "
                "or disinformation because intent is not "
                "directly annotated."
            ),
            (
                "The AEGIS five-class taxonomy must only be "
                "used where the source annotation semantics "
                "support the target class."
            ),
            (
                "Unsupported mappings must remain unmapped "
                "rather than being inferred."
            ),
            (
                "Benchmark-task labels and AEGIS-task labels "
                "must be stored separately."
            ),
        ],

        "aegis_target_taxonomy": {
            "stage_1": [
                "TRUE",
                "HARMFUL",
            ],

            "stage_2": [
                "MISINFORMATION",
                "DISINFORMATION",
                "MALINFORMATION",
                "HATE_SPEECH",
            ],
        },

        "native_supervision": {
            "2_way_label": {
                "status": (
                    "benchmark_supervision"
                ),

                "use": (
                    "Retain for binary Fakeddit benchmark "
                    "comparison."
                ),

                "aegis_mapping": (
                    "not automatically assigned"
                ),
            },

            "3_way_label": {
                "status": (
                    "benchmark_supervision"
                ),

                "use": (
                    "Retain for native three-way Fakeddit "
                    "experiments and auxiliary analysis."
                ),

                "aegis_mapping": (
                    "requires semantic review"
                ),
            },

            "6_way_label": {
                "status": (
                    "fine_grained_native_supervision"
                ),

                "use": (
                    "Retain as the primary fine-grained "
                    "Fakeddit benchmark target."
                ),

                "aegis_mapping": (
                    "class-specific review required"
                ),
            },
        },

        "mapping_status_definitions": {
            "direct": (
                "Source label semantics directly support "
                "the AEGIS class."
            ),

            "partial": (
                "Source semantics overlap with an AEGIS "
                "class but do not fully establish it."
            ),

            "unsupported": (
                "Source annotation does not provide the "
                "information required to assign the AEGIS "
                "class defensibly."
            ),
        },

        "current_mapping_policy": {
            "TRUE": {
                "status": (
                    "to_be_verified_against_native_semantics"
                )
            },

            "MISINFORMATION": {
                "status": "unsupported",
                "reason": (
                    "Fakeddit does not directly annotate "
                    "whether false information was shared "
                    "without intent to deceive."
                ),
            },

            "DISINFORMATION": {
                "status": "unsupported",
                "reason": (
                    "Fakeddit does not directly annotate "
                    "intentional deception."
                ),
            },

            "MALINFORMATION": {
                "status": "unsupported",
                "reason": (
                    "Fakeddit does not directly annotate "
                    "truthful information weaponized or "
                    "used maliciously."
                ),
            },

            "HATE_SPEECH": {
                "status": "unsupported",
                "reason": (
                    "Fakeddit is not a hate-speech "
                    "annotation dataset."
                ),
            },
        },

        "experimental_protocols": {
            "protocol_A_native_benchmark": {
                "description": (
                    "Train and evaluate using original "
                    "Fakeddit native labels and official "
                    "splits."
                ),

                "purpose": (
                    "Enable comparison against prior "
                    "multimodal fake-news literature."
                ),
            },

            "protocol_B_aegis_harmonized": {
                "description": (
                    "Use only samples/classes whose source "
                    "semantics support an AEGIS target "
                    "under a documented harmonization rule."
                ),

                "purpose": (
                    "Support AEGIS information-integrity "
                    "classification without relabeling "
                    "beyond available evidence."
                ),
            },

            "protocol_C_representation_transfer": {
                "description": (
                    "Use Fakeddit multimodal data for "
                    "representation learning or auxiliary "
                    "pretraining while keeping AEGIS "
                    "taxonomy supervision separate."
                ),

                "purpose": (
                    "Exploit large-scale multimodal signal "
                    "without forcing incompatible labels."
                ),
            },
        },

        "status": (
            "native_label_semantic_review_pending"
        ),
    }


def main() -> None:

    print(
        "AEGIS Fakeddit Provenance Freeze"
    )

    print(
        "=" * 72
    )

    audit = load_json(
        AUDIT_PATH
    )

    provenance = load_json(
        PROVENANCE_PATH
    )

    frozen_manifest = (
        build_frozen_manifest(
            audit,
            provenance,
        )
    )

    harmonization_spec = (
        build_harmonization_spec()
    )

    save_json(
        FROZEN_MANIFEST_PATH,
        frozen_manifest,
    )

    save_json(
        HARMONIZATION_PATH,
        harmonization_spec,
    )

    print(
        "Dataset:",
        frozen_manifest[
            "dataset"
        ],
    )

    print(
        "Total samples:",
        frozen_manifest[
            "total_samples"
        ],
    )

    print(
        "Image coverage:",
        (
            f"{frozen_manifest['image_coverage'] * 100:.4f}%"
        ),
    )

    print(
        "Cross-split leakage:",
        frozen_manifest[
            "cross_split_leakage"
        ],
    )

    print(
        "Integrity status:",
        frozen_manifest[
            "integrity_status"
        ].upper(),
    )

    print()
    print(
        "Created:"
    )

    print(
        "  data/manifests/fakeddit/"
        "frozen_manifest.json"
    )

    print(
        "  data/manifests/fakeddit/"
        "harmonization_spec.json"
    )

    print()
    print(
        "Fakeddit is frozen for empirical "
        "research use."
    )


if __name__ == "__main__":
    main()
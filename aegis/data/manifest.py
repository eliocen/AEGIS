"""
AEGIS Dataset Split Manifest Utilities.

Version: 0.17.0
"""

import json
from pathlib import Path

from .statistics import (
    compute_dataset_statistics,
)


def build_split_manifest(
    split,
    strategy: str,
    seed: int,
    group_attribute=None,
):
    """
    Build a reproducible manifest for one
    AEGIS dataset split.
    """

    manifest = {
        "strategy": strategy,
        "seed": seed,
        "group_attribute": (
            group_attribute
        ),
        "splits": {},
    }

    for name in [
        "train",
        "validation",
        "test",
    ]:

        samples = list(
            split.get(
                name,
                []
            )
        )

        manifest["splits"][
            name
        ] = {
            "sample_ids": [
                sample.sample_id
                for sample in samples
            ],

            "statistics": (
                compute_dataset_statistics(
                    samples
                )
            ),
        }

    return manifest


def save_split_manifest(
    manifest,
    path,
):
    """
    Save a split manifest as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


def load_split_manifest(
    path,
):
    """
    Load a previously saved split manifest.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )
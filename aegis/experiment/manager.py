"""
AEGIS Experiment Manager.

Version: 0.19.0
"""

import json
import re
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from .config import (
    ExperimentConfig,
)

from .metadata import (
    ExperimentMetadata,
)


def _safe_name(
    value: str,
) -> str:

    value = value.strip()

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip(
        "._-"
    ) or "experiment"


class ExperimentManager:
    """
    Creates and manages reproducible AEGIS
    experiment directories and metadata.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        root_directory="experiments",
        experiment_id=None,
    ):

        if not isinstance(
            config,
            ExperimentConfig,
        ):
            raise TypeError(
                "config must be ExperimentConfig."
            )

        self.config = config

        self.root_directory = Path(
            root_directory
        )

        if experiment_id is None:

            timestamp = (
                datetime.now(
                    timezone.utc
                )
                .strftime(
                    "%Y%m%dT%H%M%SZ"
                )
            )

            experiment_id = (
                f"{_safe_name(config.experiment_name)}"
                f"_{timestamp}"
            )

        self.experiment_id = (
            _safe_name(
                experiment_id
            )
        )

        self.run_directory = (
            self.root_directory
            / self.experiment_id
        )

        self.checkpoint_directory = (
            self.run_directory
            / "checkpoints"
        )

        self.artifact_directory = (
            self.run_directory
            / "artifacts"
        )

        self.config_path = (
            self.run_directory
            / "config.json"
        )

        self.metadata_path = (
            self.run_directory
            / "metadata.json"
        )

        self.history_path = (
            self.run_directory
            / "history.json"
        )

    def initialize(self):

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.artifact_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = (
            ExperimentMetadata.capture(
                self.experiment_id
            )
        )

        with self.config_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.config.as_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata.as_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return self

    def load_config_snapshot(self):

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    def load_metadata_snapshot(self):

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )
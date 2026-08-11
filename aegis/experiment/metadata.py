"""
AEGIS Experiment Metadata.

Version: 0.19.0
"""

import platform
import sys

from dataclasses import (
    asdict,
    dataclass,
)

from datetime import (
    datetime,
    timezone,
)

import torch


@dataclass
class ExperimentMetadata:
    """
    Environment and runtime metadata for one
    AEGIS experiment.
    """

    experiment_id: str

    created_at_utc: str

    python_version: str

    platform: str

    pytorch_version: str

    cuda_available: bool

    cuda_version: str | None

    device_count: int

    @classmethod
    def capture(
        cls,
        experiment_id: str,
    ):

        return cls(
            experiment_id=(
                experiment_id
            ),

            created_at_utc=(
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),

            python_version=(
                sys.version.split()[0]
            ),

            platform=(
                platform.platform()
            ),

            pytorch_version=(
                torch.__version__
            ),

            cuda_available=(
                torch.cuda.is_available()
            ),

            cuda_version=(
                torch.version.cuda
            ),

            device_count=(
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
        )

    def as_dict(self):

        return asdict(
            self
        )
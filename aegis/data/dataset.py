"""
AEGIS PyTorch Dataset.

Version: 0.17.0
"""

from typing import Iterable, List

from torch.utils.data import Dataset

from .record import (
    ResearchSample,
)

from .validation import (
    validate_research_sample,
)


class AEGISResearchDataset(
    Dataset
):
    """
    PyTorch-compatible dataset for AEGIS
    multimodal Information Integrity research.
    """

    def __init__(
        self,
        samples: Iterable[
            ResearchSample
        ],
        validate: bool = True,
    ):

        self.samples: List[
            ResearchSample
        ] = list(
            samples
        )

        if not self.samples:
            raise ValueError(
                "AEGISResearchDataset cannot "
                "be empty."
            )

        if validate:

            seen_ids = set()

            for sample in self.samples:

                validate_research_sample(
                    sample
                )

                if (
                    sample.sample_id
                    in seen_ids
                ):
                    raise ValueError(
                        "Duplicate sample_id detected: "
                        f"{sample.sample_id}"
                    )

                seen_ids.add(
                    sample.sample_id
                )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        return self.samples[
            index
        ]

    @property
    def labels(self):
        return [
            sample.label
            for sample
            in self.samples
        ]

    @property
    def sample_ids(self):
        return [
            sample.sample_id
            for sample
            in self.samples
        ]
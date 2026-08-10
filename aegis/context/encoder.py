"""
AEGIS Context Intelligence encoder.

Version: 0.11.0
"""

import hashlib

import torch
from torch import nn

from .domains import OperationalDomain
from .input import ContextInput
from .output import ContextRepresentation


class ContextEncoder(nn.Module):
    """
    Encodes structured operational context into
    a dense context representation.
    """

    def __init__(
        self,
        domain_dim: int = 32,
        auxiliary_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()

        self.domain_dim = domain_dim
        self.auxiliary_dim = auxiliary_dim
        self.output_dim = output_dim

        self.domains = list(
            OperationalDomain
        )

        self.domain_to_index = {
            domain: index
            for index, domain
            in enumerate(self.domains)
        }

        self.domain_embedding = nn.Embedding(
            num_embeddings=len(self.domains),
            embedding_dim=domain_dim,
        )

        input_dim = (
            domain_dim
            + auxiliary_dim * 4
        )

        self.projection = nn.Sequential(
            nn.Linear(
                input_dim,
                output_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                output_dim
            ),
        )

    def _hashed_vector(
        self,
        value,
        dimension,
        device,
    ):
        """
        Deterministically encode lightweight categorical
        context without maintaining a fixed vocabulary.
        """

        if not value:
            return torch.zeros(
                dimension,
                device=device,
            )

        text = str(value).strip().lower()

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        values = []

        while len(values) < dimension:
            for byte in digest:
                normalized = (
                    float(byte) / 127.5
                ) - 1.0

                values.append(
                    normalized
                )

                if len(values) == dimension:
                    break

            digest = hashlib.sha256(
                digest
            ).digest()

        return torch.tensor(
            values,
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        context: ContextInput,
    ) -> ContextRepresentation:

        if not isinstance(
            context,
            ContextInput,
        ):
            raise TypeError(
                "ContextEncoder expects ContextInput."
            )

        device = (
            self.domain_embedding
            .weight
            .device
        )

        domain_index = torch.tensor(
            [
                self.domain_to_index[
                    context.domain
                ]
            ],
            dtype=torch.long,
            device=device,
        )

        domain_vector = (
            self.domain_embedding(
                domain_index
            )
            .squeeze(0)
        )

        platform_vector = (
            self._hashed_vector(
                context.platform,
                self.auxiliary_dim,
                device,
            )
        )

        country_vector = (
            self._hashed_vector(
                context.country,
                self.auxiliary_dim,
                device,
            )
        )

        language_vector = (
            self._hashed_vector(
                context.language,
                self.auxiliary_dim,
                device,
            )
        )

        event_vector = (
            self._hashed_vector(
                context.event,
                self.auxiliary_dim,
                device,
            )
        )

        combined = torch.cat(
            [
                domain_vector,
                platform_vector,
                country_vector,
                language_vector,
                event_vector,
            ],
            dim=-1,
        )

        embedding = self.projection(
            combined
        )

        return ContextRepresentation(
            sample_id=context.sample_id,

            embedding=embedding,

            dimension=self.output_dim,

            metadata={
                "domain": (
                    context.domain.value
                ),
                "platform": (
                    context.platform
                ),
                "country": (
                    context.country
                ),
                "language": (
                    context.language
                ),
                "event": (
                    context.event
                ),
            },
        )
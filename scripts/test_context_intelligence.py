"""
AEGIS v0.11.0
Context Intelligence sanity experiment.
"""

import torch

from aegis.alignment import MultimodalRepresentation

from aegis.context import (
    ContextAwareFusion,
    ContextEncoder,
    ContextInput,
    ContextIntelligenceLayer,
    OperationalDomain,
)


torch.manual_seed(42)


representation = MultimodalRepresentation(
    sample_id="SEC-001",
    fused_embedding=torch.randn(512),
    shared_dimension=512,
    text_available=True,
    vision_available=True,
    is_multimodal=True,
)


encoder = ContextEncoder()

fusion = ContextAwareFusion()

layer = ContextIntelligenceLayer(
    encoder=encoder,
    fusion=fusion,
    device="cpu",
)


contexts = [
    ContextInput(
        sample_id="SEC-001",
        domain=OperationalDomain.ELECTION,
        country="Uganda",
        platform="X",
        language="en",
        event="National Election",
    ),

    ContextInput(
        sample_id="SEC-001",
        domain=OperationalDomain.CONFLICT,
        country="Uganda",
        platform="X",
        language="en",
        event="Regional Security Crisis",
    ),

    ContextInput(
        sample_id="SEC-001",
        domain=OperationalDomain.PUBLIC_HEALTH,
        country="Uganda",
        platform="X",
        language="en",
        event="Disease Outbreak",
    ),
]


for context in contexts:

    result = layer.process(
        (
            representation,
            context,
        )
    )

    print(
        "Domain:",
        context.domain.value,
    )

    print(
        "Context dimension:",
        result.context_dimension,
    )

    print(
        "Output dimension:",
        result.fused_dimension,
    )

    print(
        "Norm:",
        result.fused_embedding.norm().item(),
    )

    print("-" * 40)
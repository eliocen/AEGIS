"""
Manual multilingual representation test.

AEGIS v0.7.0
"""

import torch

from aegis.preprocessing import (
    CanonicalSample,
)

from aegis.representation import (
    TransformerTextEncoder,
)


encoder = TransformerTextEncoder()

samples = [
    CanonicalSample(
        sample_id="EN1",
        text=(
            "Artificial intelligence can amplify "
            "misinformation during conflict."
        ),
        language="en",
        has_text=True,
    ),

    CanonicalSample(
        sample_id="ZH1",
        text="人工智能可能在冲突期间放大虚假信息。",
        language="zh",
        has_text=True,
    ),
]


representations = [
    encoder.encode(sample)
    for sample in samples
]


for representation in representations:

    print(
        representation.sample_id,
        representation.language,
        representation.dimension,
    )


similarity = (
    torch.nn.functional.cosine_similarity(
        representations[0]
        .embedding
        .unsqueeze(0),

        representations[1]
        .embedding
        .unsqueeze(0),
    )
)


print(
    "Cross-lingual cosine similarity:",
    similarity.item(),
)
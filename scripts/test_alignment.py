"""
AEGIS v0.9.0
Cross-modal alignment sanity experiment.
"""

import torch

from aegis.alignment import (
    CrossModalAlignmentModel,
)


torch.manual_seed(42)


model = CrossModalAlignmentModel(
    text_dim=768,
    vision_dim=512,
    shared_dim=512,
    dropout=0.0,
)


text = torch.randn(
    4,
    768,
)

vision = torch.randn(
    4,
    512,
)


result = model(
    text,
    vision,
    compute_loss=True,
)


print(
    "Aligned text:",
    result["aligned_text"].shape,
)

print(
    "Aligned vision:",
    result["aligned_vision"].shape,
)

print(
    "Fused:",
    result["fused_embedding"].shape,
)

print(
    "Contrastive loss:",
    result["alignment_loss"].item(),
)
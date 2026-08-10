"""
AEGIS v0.10.0
Hierarchical Information Integrity Classification
sanity experiment.
"""

import torch

from aegis.classification import (
    HierarchicalInformationIntegrityClassifier,
    WeightedHierarchicalLoss,
)


torch.manual_seed(42)


model = (
    HierarchicalInformationIntegrityClassifier(
        input_dim=512,
        hidden_dim=256,
        dropout=0.0,
    )
)


batch_size = 8


embeddings = torch.randn(
    batch_size,
    512,
)


outputs = model(
    embeddings
)


integrity_targets = torch.tensor(
    [
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
    ]
)


threat_targets = torch.tensor(
    [
        -1,
        0,
        1,
        2,
        -1,
        3,
        1,
        -1,
    ]
)


loss_fn = WeightedHierarchicalLoss(
    integrity_weight=1.0,
    threat_weight=1.0,
)


losses = loss_fn(
    outputs[
        "integrity_logits"
    ],
    outputs[
        "threat_logits"
    ],
    integrity_targets,
    threat_targets,
)


print(
    "Feature shape:",
    outputs[
        "features"
    ].shape,
)


print(
    "Integrity logits:",
    outputs[
        "integrity_logits"
    ].shape,
)


print(
    "Threat logits:",
    outputs[
        "threat_logits"
    ].shape,
)


print(
    "Harmful samples:",
    losses[
        "harmful_count"
    ],
)


print(
    "Integrity loss:",
    losses[
        "integrity_loss"
    ].item(),
)


print(
    "Threat loss:",
    losses[
        "threat_loss"
    ].item(),
)


print(
    "Total hierarchical loss:",
    losses[
        "loss"
    ].item(),
)
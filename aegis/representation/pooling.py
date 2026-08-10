"""
Pooling strategies for transformer representations.

Version: 0.7.0
"""


def masked_mean_pooling(
    hidden_states,
    attention_mask,
):
    """
    Perform attention-mask-aware mean pooling.

    Padding tokens are excluded from the final
    sentence representation.
    """

    import torch

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(hidden_states.size())
        .float()
    )

    masked_embeddings = (
        hidden_states * mask
    )

    summed = torch.sum(
        masked_embeddings,
        dim=1,
    )

    token_counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed / token_counts
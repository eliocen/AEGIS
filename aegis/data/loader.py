"""
AEGIS DataLoader Factory.

Version: 0.17.0
"""

from torch.utils.data import (
    DataLoader,
)

from .collate import (
    aegis_collate_fn,
)


def create_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
):
    """
    Create an AEGIS PyTorch DataLoader.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=(
            aegis_collate_fn
        ),
    )
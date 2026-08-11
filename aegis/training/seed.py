"""
AEGIS Reproducibility Utilities.

Version: 0.16.0
"""

import os
import random

import torch


def set_global_seed(
    seed: int = 42,
) -> None:
    """
    Seed Python and PyTorch random number generators.
    """

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )
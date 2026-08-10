"""
Device resolution utilities for AEGIS.

Version: 0.7.0
"""


def resolve_device(
    requested: str = "auto",
) -> str:
    """
    Resolve the execution device.

    Supported values:
    - auto
    - cpu
    - cuda

    CUDA availability is determined dynamically.
    """

    requested = requested.lower().strip()

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch "
                "is not installed."
            )

        return "cpu"

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return "cuda"

    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    raise ValueError(
        f"Unsupported device setting: {requested}"
    )
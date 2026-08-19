"""
AEGIS v0.24.0
Real Fakeddit Representation Sanity Check
=========================================

Purpose
-------
Verify the complete real-data representation path:

    FakedditAdapter
        -> EmpiricalSample
        -> CanonicalSample
        -> XLM-R text representation
        -> CLIP vision representation
        -> FakedditRepresentationBridge
        -> BinaryIntegrityBatch

This is NOT a training experiment.

Only two real Fakeddit training samples are processed.

The pretrained encoders remain frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch

from aegis.data.adapters import (
    FakedditAdapter,
)

from aegis.data.bridges import (
    FakedditRepresentationBridge,
)

from aegis.representation import (
    TransformerTextEncoder,
    TransformerVisionEncoder,
)


DATASET_ROOT = Path(
    "data/raw/fakeddit"
)

SPLIT = "train"

SAMPLE_LIMIT = 2

TEXT_MODEL = (
    "FacebookAI/xlm-roberta-base"
)

VISION_MODEL = (
    "openai/clip-vit-base-patch32"
)


def separator() -> None:
    print("=" * 78)


def print_environment() -> None:
    separator()
    print("AEGIS REAL FAKEDDIT REPRESENTATION SANITY CHECK")
    separator()

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "CUDA build:",
        torch.version.cuda,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "GPU memory:",
            (
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            ),
        )

    print()


def main() -> None:

    print_environment()

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Fakeddit dataset root does not exist: "
            f"{DATASET_ROOT.resolve()}"
        )

    print(
        "Dataset root:",
        DATASET_ROOT.resolve(),
    )

    print(
        "Split:",
        SPLIT,
    )

    print(
        "Sample limit:",
        SAMPLE_LIMIT,
    )

    print()

    separator()
    print("INITIALIZING FAKEDDIT ADAPTER")
    separator()

    adapter = FakedditAdapter(
        dataset_root=DATASET_ROOT
    )

    adapter.validate_structure()

    print(
        "Fakeddit adapter: READY"
    )

    print()

    separator()
    print("INITIALIZING REAL PRETRAINED ENCODERS")
    separator()

    print(
        "Loading text encoder:",
        TEXT_MODEL,
    )

    text_encoder = (
        TransformerTextEncoder(
            model_name=TEXT_MODEL,
            device="auto",
            max_length=256,
        )
    )

    print(
        "Text encoder device:",
        text_encoder.device,
    )

    print(
        "Loading vision encoder:",
        VISION_MODEL,
    )

    vision_encoder = (
        TransformerVisionEncoder(
            model_name=VISION_MODEL,
            device="auto",
        )
    )

    print(
        "Vision encoder device:",
        vision_encoder.device,
    )

    print()

    separator()
    print("INITIALIZING REPRESENTATION BRIDGE")
    separator()

    bridge = (
        FakedditRepresentationBridge(
            text_encoder=text_encoder,
            vision_encoder=vision_encoder,
            expected_text_dim=768,
            expected_vision_dim=512,
        )
    )

    print(
        "Representation bridge: READY"
    )

    print()

    separator()
    print("STREAMING REAL FAKEDDIT SAMPLES")
    separator()

    samples = list(
        adapter.iter_samples(
            split=SPLIT,
            limit=SAMPLE_LIMIT,
        )
    )

    if len(samples) != SAMPLE_LIMIT:
        raise RuntimeError(
            f"Expected {SAMPLE_LIMIT} samples, "
            f"received {len(samples)}."
        )

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        print()
        print(
            f"SAMPLE {index}"
        )

        print(
            "-" * 78
        )

        print(
            "Sample ID:",
            sample.sample_id,
        )

        print(
            "Split:",
            sample.split,
        )

        print(
            "Text:",
            sample.text,
        )

        print(
            "Image:",
            sample.image_path,
        )

        print(
            "Image status:",
            sample.image_status.value,
        )

        print(
            "Native labels:",
            sample.native_labels,
        )

    print()

    separator()
    print("GENERATING REAL REPRESENTATIONS")
    separator()

    represented = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        print(
            f"Encoding sample {index}/{SAMPLE_LIMIT}: "
            f"{sample.sample_id}"
        )

        item = bridge.represent(
            sample
        )

        represented.append(
            item
        )

        print(
            "  Native label:",
            item.native_label,
        )

        print(
            "  Integrity target:",
            item.integrity_target,
        )

        print(
            "  Text shape:",
            tuple(
                item.text_embedding.shape
            ),
        )

        print(
            "  Vision shape:",
            tuple(
                item.vision_embedding.shape
            ),
        )

        print(
            "  Text norm:",
            round(
                float(
                    torch.linalg.vector_norm(
                        item.text_embedding
                    ).item()
                ),
                6,
            ),
        )

        print(
            "  Vision norm:",
            round(
                float(
                    torch.linalg.vector_norm(
                        item.vision_embedding
                    ).item()
                ),
                6,
            ),
        )

        print(
            "  Text model:",
            item.text_model,
        )

        print(
            "  Vision model:",
            item.vision_model,
        )

        print()

    separator()
    print("BUILDING REAL BINARY TRAINING BATCH")
    separator()

    batch = bridge.build_batch(
        samples
    )

    print(
        "Batch size:",
        batch.batch_size,
    )

    print(
        "Text tensor shape:",
        tuple(
            batch.text_embeddings.shape
        ),
    )

    print(
        "Vision tensor shape:",
        tuple(
            batch.vision_embeddings.shape
        ),
    )

    print(
        "Integrity targets:",
        batch.integrity_targets.tolist(),
    )

    print(
        "Sample IDs:",
        batch.sample_ids,
    )

    print(
        "Text tensor device:",
        batch.text_embeddings.device,
    )

    print(
        "Vision tensor device:",
        batch.vision_embeddings.device,
    )

    if torch.cuda.is_available():
        print()
        print(
            "CUDA allocated memory:",
            (
                f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            ),
        )

        print(
            "CUDA reserved memory:",
            (
                f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
            ),
        )

    print()
    separator()
    print("RESULT")
    separator()

    print(
        "REAL FAKEDDIT REPRESENTATION PIPELINE: PASSED"
    )

    print()
    print(
        "Real Fakeddit text and images were successfully "
        "encoded through XLM-R and CLIP and converted into "
        "an AEGIS BinaryIntegrityBatch."
    )

    print()
    print(
        "IMPORTANT: No model training occurred in this sanity test."
    )


if __name__ == "__main__":
    main()
"""
AEGIS v0.24.0
Real Fakeddit Frozen Representation Cache Builder
=================================================

Creates a deterministic class-stratified Fakeddit subset using
bounded-memory streaming reservoir sampling with image-decodability
eligibility validation.

Only after the final sample population has passed eligibility
validation are frozen XLM-R and CLIP representations generated.

This script performs representation extraction only.

It does NOT train AEGIS.
"""

from __future__ import annotations

import argparse
import json

from collections import Counter
from pathlib import Path

import torch

from aegis.data.adapters import (
    FakedditAdapter,
)

from aegis.data.bridges import (
    FakedditRepresentationBridge,
)

from aegis.data.cache import (
    FakedditRepresentationCache,
    FakedditRepresentationCacheWriter,
)

from aegis.data.sampling import (
    select_streaming_decodable_fakeddit_samples,
)

from aegis.representation import (
    TransformerTextEncoder,
    TransformerVisionEncoder,
)


DEFAULT_DATASET_ROOT = Path(
    "data/raw/fakeddit"
)

DEFAULT_CACHE_BASE = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings"
)

TEXT_MODEL = (
    "FacebookAI/xlm-roberta-base"
)

VISION_MODEL = (
    "openai/clip-vit-base-patch32"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, "
            "decodability-aware real Fakeddit "
            "representation cache."
        )
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=[
            "train",
            "validation",
            "test",
        ],
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--reserve-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--min-reserve-per-class",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(
            DEFAULT_DATASET_ROOT
        ),
    )

    parser.add_argument(
        "--cache-base",
        type=Path,
        default=(
            DEFAULT_CACHE_BASE
        ),
    )

    return parser.parse_args()


def separator():
    print(
        "=" * 78
    )


def cache_name(
    split: str,
    samples: int,
    seed: int,
) -> str:

    return (
        f"{split}_"
        f"n{samples}_"
        f"seed{seed}"
    )


def validate_arguments(
    args,
) -> None:

    if args.samples <= 0:
        raise ValueError(
            "--samples must be greater "
            "than zero."
        )

    if args.shard_size <= 0:
        raise ValueError(
            "--shard-size must be "
            "greater than zero."
        )

    if args.reserve_fraction < 0:
        raise ValueError(
            "--reserve-fraction must "
            "be >= 0."
        )

    if (
        args.min_reserve_per_class
        < 0
    ):
        raise ValueError(
            "--min-reserve-per-class "
            "must be >= 0."
        )

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            "Fakeddit dataset root "
            f"does not exist: "
            f"{args.dataset_root.resolve()}"
        )


def print_environment(
    args,
) -> None:

    separator()

    print(
        "AEGIS REAL FAKEDDIT "
        "REPRESENTATION CACHE"
    )

    separator()

    print(
        "Dataset root:",
        args.dataset_root.resolve(),
    )

    print(
        "Split:",
        args.split,
    )

    print(
        "Requested samples:",
        args.samples,
    )

    print(
        "Sampling seed:",
        args.seed,
    )

    print(
        "Shard size:",
        args.shard_size,
    )

    print(
        "Reserve fraction:",
        args.reserve_fraction,
    )

    print(
        "Minimum reserve per class:",
        args.min_reserve_per_class,
    )

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
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "GPU memory:",
            (
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            ),
        )

    print()


def initialize_adapter(
    args,
) -> FakedditAdapter:

    separator()

    print(
        "INITIALIZING "
        "FAKEDDIT ADAPTER"
    )

    separator()

    adapter = (
        FakedditAdapter(
            dataset_root=(
                args.dataset_root
            )
        )
    )

    adapter.validate_structure()

    print(
        "Fakeddit adapter: READY"
    )

    print()

    return adapter


def select_samples(
    adapter: FakedditAdapter,
    args,
):

    separator()

    print(
        "DECODABILITY-AWARE "
        "DETERMINISTIC SAMPLING"
    )

    separator()

    print(
        "Streaming full Fakeddit split "
        "with bounded-memory reservoir "
        "sampling..."
    )

    print(
        "Candidate images will be "
        "validated before neural encoding."
    )

    print()

    selection = (
        select_streaming_decodable_fakeddit_samples(
            adapter.iter_samples(
                split=args.split
            ),
            sample_count=(
                args.samples
            ),
            seed=(
                args.seed
            ),
            reserve_fraction=(
                args.reserve_fraction
            ),
            min_reserve_per_class=(
                args.min_reserve_per_class
            ),
        )
    )

    print(
        "Population scanned:",
        selection.population_seen,
    )

    print(
        "Candidate samples checked:",
        selection.candidate_count,
    )

    print(
        "Valid candidates:",
        selection.valid_candidate_count,
    )

    print(
        "Invalid candidates excluded:",
        selection.exclusion_count,
    )

    print(
        "Selected samples:",
        selection.actual_count,
    )

    print(
        "Sampling strategy:",
        selection.strategy,
    )

    print(
        "Native label counts:",
        selection.native_label_counts,
    )

    print()

    if selection.exclusions:

        print(
            "EXCLUDED CANDIDATES"
        )

        for exclusion in (
            selection.exclusions
        ):

            print(
                " ",
                exclusion.sample_id,
                "| label=",
                exclusion.native_label,
                "| reason=",
                exclusion.reason,
            )

        print()

    print(
        "FINAL SELECTED SAMPLE IDS:"
    )

    for sample in (
        selection.samples
    ):

        print(
            " ",
            sample.sample_id,
        )

    print()

    return selection


def build_cache_root(
    args,
) -> Path:

    return (
        args.cache_base
        / cache_name(
            split=args.split,
            samples=args.samples,
            seed=args.seed,
        )
    )


def write_pre_encoding_provenance(
    cache_root: Path,
    selection,
    args,
):
    """
    Persist selection and exclusions BEFORE GPU representation work.

    This means the empirical population is known and auditable even
    if representation extraction later fails.
    """

    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
        cache_root
        / "selection.json"
    )

    exclusions_path = (
        cache_root
        / "excluded_samples.jsonl"
    )

    selection_metadata = {
        "dataset": "Fakeddit",

        "split": (
            args.split
        ),

        "requested_sample_count": (
            args.samples
        ),

        "actual_sample_count": (
            selection.actual_count
        ),

        "population_seen": (
            selection.population_seen
        ),

        "candidate_count": (
            selection.candidate_count
        ),

        "valid_candidate_count": (
            selection.valid_candidate_count
        ),

        "excluded_candidate_count": (
            selection.exclusion_count
        ),

        "sampling_strategy": (
            selection.strategy
        ),

        "sampling_seed": (
            selection.seed
        ),

        "reserve_fraction": (
            args.reserve_fraction
        ),

        "min_reserve_per_class": (
            args.min_reserve_per_class
        ),

        "selected_sample_ids": (
            selection.sample_ids
        ),

        "native_label_counts": (
            selection.native_label_counts
        ),

        "text_model": (
            TEXT_MODEL
        ),

        "vision_model": (
            VISION_MODEL
        ),

        "eligibility": {
            "requires_text": True,
            "requires_local_image": True,
            "requires_decodable_image": True,
        },
    }

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            selection_metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with exclusions_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for exclusion in (
            selection.exclusions
        ):

            handle.write(
                json.dumps(
                    exclusion.as_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Selection provenance:",
        selection_path,
    )

    print(
        "Exclusion provenance:",
        exclusions_path,
    )

    print()

    return (
        selection_path,
        exclusions_path,
    )


def initialize_encoders():

    separator()

    print(
        "INITIALIZING "
        "REAL ENCODERS"
    )

    separator()

    print(
        "All final samples have passed "
        "image-decodability validation."
    )

    print()

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

    return (
        text_encoder,
        vision_encoder,
    )


def generate_representations(
    selection,
    text_encoder,
    vision_encoder,
):

    separator()

    print(
        "INITIALIZING "
        "REPRESENTATION BRIDGE"
    )

    separator()

    bridge = (
        FakedditRepresentationBridge(
            text_encoder=(
                text_encoder
            ),
            vision_encoder=(
                vision_encoder
            ),
            expected_text_dim=768,
            expected_vision_dim=512,
        )
    )

    print(
        "Representation bridge: READY"
    )

    print()

    separator()

    print(
        "GENERATING "
        "REAL REPRESENTATIONS"
    )

    separator()

    representations = []

    total = len(
        selection.samples
    )

    for index, sample in enumerate(
        selection.samples,
        start=1,
    ):

        print(
            f"[{index}/{total}] "
            f"{sample.sample_id}"
        )

        represented = (
            bridge.represent(
                sample
            )
        )

        representations.append(
            represented
        )

    print()

    return representations


def write_cache(
    cache_root: Path,
    selection,
    representations,
    args,
):

    separator()

    print(
        "WRITING CACHE"
    )

    separator()

    print(
        "Cache root:",
        cache_root.resolve(),
    )

    writer = (
        FakedditRepresentationCacheWriter(
            cache_root=cache_root,
            split=args.split,
            shard_size=(
                args.shard_size
            ),
            sampling_strategy=(
                selection.strategy
            ),
            sampling_seed=(
                args.seed
            ),
        )
    )

    manifest = (
        writer.write(
            representations
        )
    )

    print(
        "Samples written:",
        manifest.sample_count,
    )

    print(
        "Shards written:",
        manifest.shard_count,
    )

    print(
        "Manifest sampling strategy:",
        manifest.sampling_strategy,
    )

    print()

    return manifest


def verify_cache(
    cache_root: Path,
    selection,
    args,
):

    separator()

    print(
        "RELOADING CACHE "
        "FOR VERIFICATION"
    )

    separator()

    cache = (
        FakedditRepresentationCache(
            cache_root
        )
    )

    if (
        cache.sample_count
        != args.samples
    ):
        raise RuntimeError(
            "Reloaded cache sample count "
            "does not match requested count."
        )

    loaded_ids = []

    loaded_targets = []

    verification_batch_size = min(
        64,
        max(
            1,
            args.samples,
        ),
    )

    for batch in (
        cache.iter_batches(
            batch_size=(
                verification_batch_size
            )
        )
    ):

        loaded_ids.extend(
            batch.sample_ids
        )

        loaded_targets.extend(
            batch.integrity_targets.tolist()
        )

    selected_ids = (
        selection.sample_ids
    )

    if loaded_ids != selected_ids:
        raise RuntimeError(
            "Cache verification failed: "
            "sample IDs changed."
        )

    expected_targets = [
        (
            0
            if int(
                sample.native_labels[
                    "2_way_label"
                ]
            )
            == 1
            else 1
        )
        for sample
        in selection.samples
    ]

    if (
        loaded_targets
        != expected_targets
    ):
        raise RuntimeError(
            "Cache verification failed: "
            "integrity targets changed."
        )

    target_counts = dict(
        sorted(
            Counter(
                loaded_targets
            ).items()
        )
    )

    print(
        "Reloaded sample count:",
        cache.sample_count,
    )

    print(
        "Text dimension:",
        cache.text_dimension,
    )

    print(
        "Vision dimension:",
        cache.vision_dimension,
    )

    print(
        "Integrity target counts:",
        target_counts,
    )

    print(
        "Sample-ID verification:",
        "PASSED",
    )

    print(
        "Target verification:",
        "PASSED",
    )

    print()

    return (
        cache,
        target_counts,
    )


def update_selection_metadata(
    cache_root: Path,
    target_counts,
):

    selection_path = (
        cache_root
        / "selection.json"
    )

    with selection_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        metadata = (
            json.load(
                handle
            )
        )

    metadata[
        "integrity_target_counts"
    ] = (
        target_counts
    )

    metadata[
        "cache_verification"
    ] = {
        "sample_ids": "passed",
        "targets": "passed",
    }

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def print_cuda_memory():

    if not torch.cuda.is_available():
        return

    print(
        "CUDA allocated:",
        (
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        ),
    )

    print(
        "CUDA reserved:",
        (
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        ),
    )

    print()


def main():

    args = parse_args()

    validate_arguments(
        args
    )

    print_environment(
        args
    )

    adapter = (
        initialize_adapter(
            args
        )
    )

    selection = (
        select_samples(
            adapter=adapter,
            args=args,
        )
    )

    if (
        selection.actual_count
        != args.samples
    ):
        raise RuntimeError(
            "Eligibility-aware sampling "
            "did not return the requested "
            "number of samples."
        )

    cache_root = (
        build_cache_root(
            args
        )
    )

    write_pre_encoding_provenance(
        cache_root=(
            cache_root
        ),
        selection=(
            selection
        ),
        args=args,
    )

    (
        text_encoder,
        vision_encoder,
    ) = initialize_encoders()

    representations = (
        generate_representations(
            selection=selection,
            text_encoder=(
                text_encoder
            ),
            vision_encoder=(
                vision_encoder
            ),
        )
    )

    manifest = (
        write_cache(
            cache_root=(
                cache_root
            ),
            selection=(
                selection
            ),
            representations=(
                representations
            ),
            args=args,
        )
    )

    (
        cache,
        target_counts,
    ) = verify_cache(
        cache_root=(
            cache_root
        ),
        selection=(
            selection
        ),
        args=args,
    )

    update_selection_metadata(
        cache_root=(
            cache_root
        ),
        target_counts=(
            target_counts
        ),
    )

    print_cuda_memory()

    separator()

    print(
        "REAL FAKEDDIT CACHE: PASSED"
    )

    separator()

    print()

    print(
        "Frozen representations were "
        "generated, cached, reloaded "
        "and verified successfully."
    )

    print()

    print(
        "Sampling strategy:",
        selection.strategy,
    )

    print(
        "Population scanned:",
        selection.population_seen,
    )

    print(
        "Candidate samples checked:",
        selection.candidate_count,
    )

    print(
        "Invalid candidates excluded:",
        selection.exclusion_count,
    )

    print(
        "Selected samples:",
        selection.actual_count,
    )

    print(
        "Cache samples:",
        cache.sample_count,
    )

    print(
        "Cache shards:",
        manifest.shard_count,
    )

    print()

    print(
        "No AEGIS model training "
        "occurred."
    )


if __name__ == "__main__":
    main()
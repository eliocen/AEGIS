"""Runner-level v0.28 deterministic preparation tests."""

import torch

from aegis.training import BinaryIntegrityBatch
from scripts.run_fakeddit_ablation import (
    prepare_m4qcs_training_batch,
    requires_quality_feature_statistics,
)


def _batch(size=32):
    return BinaryIntegrityBatch(
        text_embeddings=torch.randn(size, 8),
        vision_embeddings=torch.randn(size, 6),
        integrity_targets=torch.arange(size) % 2,
        sample_ids=[f"v028-{i}" for i in range(size)],
    )


def test_all_m4qcs_variants_initialize_quality_feature_statistics():
    for architecture in (
        "quality_compatibility_selective_weights",
        "quality_compatibility_selective_interaction",
        "quality_compatibility_selective_fusion",
    ):
        assert requires_quality_feature_statistics(architecture)


def test_nonquality_architecture_does_not_initialize_quality_statistics():
    assert not requires_quality_feature_statistics("gated_interaction")


def test_m4qcs_preparation_is_deterministic_and_preserves_batch_size():
    torch.manual_seed(28)
    batch = _batch()
    kwargs = {
        "text_feature_std": torch.ones(8),
        "vision_feature_std": torch.ones(6),
        "seed": 28_000_042,
    }
    first = prepare_m4qcs_training_batch(batch, **kwargs)
    second = prepare_m4qcs_training_batch(batch, **kwargs)
    first_batch, first_q, first_c, first_w, first_counts = first
    second_batch, second_q, second_c, second_w, second_counts = second
    assert first_batch.batch_size == batch.batch_size
    assert torch.equal(first_batch.text_embeddings, second_batch.text_embeddings)
    assert torch.equal(first_batch.vision_embeddings, second_batch.vision_embeddings)
    assert all(torch.equal(a, b) for a, b in zip(first_q, second_q))
    assert torch.equal(first_c, second_c)
    assert all(torch.equal(a, b) for a, b in zip(first_w, second_w))
    assert first_counts == second_counts


def test_only_text_gaussian_examples_receive_multiplier_two():
    batch = _batch(128)
    _, _, _, (text_weights, vision_weights), counts = (
        prepare_m4qcs_training_batch(
            batch,
            text_feature_std=torch.ones(8),
            vision_feature_std=torch.ones(6),
            seed=28_000_043,
        )
    )
    assert set(text_weights.unique().tolist()).issubset({1.0, 2.0})
    assert torch.equal(vision_weights, torch.ones_like(vision_weights))
    assert int((text_weights == 2.0).sum()) == counts["text_gaussian_weight_2"]


def test_text_gaussian_weight_count_matches_frozen_sampling_draws():
    batch = _batch(64)
    seed = 28_000_044
    _, _, _, _, counts = prepare_m4qcs_training_batch(
        batch,
        text_feature_std=torch.ones(8),
        vision_feature_std=torch.ones(6),
        seed=seed,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    condition = torch.rand(batch.batch_size, generator=generator)
    family = torch.rand(batch.batch_size, generator=generator)
    expected = int(
        (((condition >= 0.4) & (condition < 0.6) & (family < 0.4)).sum())
    )
    assert counts["text_gaussian_weight_2"] == expected

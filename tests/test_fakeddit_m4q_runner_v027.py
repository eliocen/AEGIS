"""AEGIS v0.27 Step 7 tests for M4q runner integration."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aegis.alignment import CrossModalAlignmentModel
from scripts.run_fakeddit_ablation import (
    M4Q_SEVERITIES,
    collect_component_fingerprints,
    effective_parameter_counts,
    prepare_m4q_training_batch,
)


def _fake_batch(batch_size: int = 128):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    text = torch.randn(batch_size, 8, generator=generator)
    vision = torch.randn(batch_size, 6, generator=generator)
    return SimpleNamespace(
        text_embeddings=text,
        vision_embeddings=vision,
        integrity_targets=torch.arange(batch_size) % 2,
        sample_ids=[f"s{i}" for i in range(batch_size)],
        batch_size=batch_size,
    )


def _prepare(seed: int = 42, batch_size: int = 128):
    batch = _fake_batch(batch_size)
    text_std = torch.std(batch.text_embeddings, dim=0, unbiased=False)
    vision_std = torch.std(batch.vision_embeddings, dim=0, unbiased=False)
    return batch, prepare_m4q_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=seed,
    )


def test_m4q_severities_are_frozen():
    assert M4Q_SEVERITIES == (0.25, 0.50, 0.75, 1.00)


def test_prepare_m4q_is_deterministic():
    batch_a, (corrupted_a, targets_a, counts_a) = _prepare(seed=77)
    batch_b, (corrupted_b, targets_b, counts_b) = _prepare(seed=77)

    assert torch.equal(batch_a.text_embeddings, batch_b.text_embeddings)
    assert torch.equal(corrupted_a.text_embeddings, corrupted_b.text_embeddings)
    assert torch.equal(corrupted_a.vision_embeddings, corrupted_b.vision_embeddings)
    assert torch.equal(targets_a[0], targets_b[0])
    assert torch.equal(targets_a[1], targets_b[1])
    assert counts_a == counts_b


def test_prepare_m4q_does_not_mutate_source_batch():
    batch = _fake_batch(64)
    text_before = batch.text_embeddings.clone()
    vision_before = batch.vision_embeddings.clone()
    text_std = torch.std(text_before, dim=0, unbiased=False)
    vision_std = torch.std(vision_before, dim=0, unbiased=False)

    prepare_m4q_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=9,
    )

    assert torch.equal(batch.text_embeddings, text_before)
    assert torch.equal(batch.vision_embeddings, vision_before)


def test_prepare_m4q_targets_have_expected_shape_and_range():
    batch, (_, (text_target, vision_target), _) = _prepare(seed=13, batch_size=96)
    assert text_target.shape == (batch.batch_size, 1)
    assert vision_target.shape == (batch.batch_size, 1)
    assert torch.all((text_target >= 0.0) & (text_target <= 1.0))
    assert torch.all((vision_target >= 0.0) & (vision_target <= 1.0))


def test_prepare_m4q_condition_counts_cover_batch():
    batch, (_, _, counts) = _prepare(seed=21, batch_size=256)
    primary = (
        counts["clean"]
        + counts["text_quality"]
        + counts["vision_quality"]
        + counts["mismatch"]
    )
    quality_families = (
        counts["gaussian_noise"]
        + counts["attenuation"]
        + counts["zero_dropout"]
    )
    assert primary == batch.batch_size
    assert quality_families == counts["text_quality"] + counts["vision_quality"]


def test_m4q_model_exposes_quality_heads():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        quality_dropout=0.0,
    )
    assert model.quality_supervised is True
    assert model.fusion_architecture == "quality_supervised"
    assert model.text_quality_estimator is not None
    assert model.vision_quality_estimator is not None
    assert model.gated_interaction_fusion is not None


def test_m4q_forward_returns_quality_scores():
    torch.manual_seed(5)
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        quality_dropout=0.0,
    )
    model.eval()
    text = torch.randn(7, 8)
    vision = torch.randn(7, 6)
    out = model(text, vision, compute_loss=False)
    assert out["text_quality"].shape == (7, 1)
    assert out["vision_quality"].shape == (7, 1)
    assert torch.all((out["text_quality"] >= 0.0) & (out["text_quality"] <= 1.0))
    assert torch.all((out["vision_quality"] >= 0.0) & (out["vision_quality"] <= 1.0))


def test_m4q_fused_representation_equals_m1b_under_paired_initialization():
    text = torch.randn(11, 8, generator=torch.Generator().manual_seed(700))
    vision = torch.randn(11, 6, generator=torch.Generator().manual_seed(701))

    torch.manual_seed(99)
    m1b = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )
    torch.manual_seed(99)
    m4q = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
    )

    assert m1b.gated_interaction_fusion is not None
    assert m4q.gated_interaction_fusion is not None
    m1b.gated_interaction_fusion.gated_fusion.load_state_dict(m1b.fusion.state_dict())
    m4q.gated_interaction_fusion.gated_fusion.load_state_dict(m4q.fusion.state_dict())
    m1b.eval()
    m4q.eval()

    out_m1b = m1b(text, vision, compute_loss=False)
    out_m4q = m4q(text, vision, compute_loss=False)
    assert torch.equal(out_m1b["aligned_text"], out_m4q["aligned_text"])
    assert torch.equal(out_m1b["aligned_vision"], out_m4q["aligned_vision"])
    assert torch.equal(out_m1b["fused_embedding"], out_m4q["fused_embedding"])


def test_m4q_fingerprints_include_quality_heads():
    torch.manual_seed(17)
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        quality_dropout=0.0,
    )
    classifier = nn.Linear(4, 2)
    fingerprints = collect_component_fingerprints(
        alignment_model=model,
        classification_model=classifier,
        mode="multimodal",
        fusion_architecture="quality_supervised",
    )
    assert "gated_interaction_fusion" in fingerprints
    assert "gated_base_fusion" in fingerprints
    assert "text_quality_estimator" in fingerprints
    assert "vision_quality_estimator" in fingerprints


def test_m4q_parameter_accounting_includes_both_quality_heads():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        quality_dropout=0.0,
    )
    classifier = nn.Linear(4, 2)
    counts = effective_parameter_counts(
        alignment_model=model,
        classification_model=classifier,
        mode="multimodal",
        fusion_architecture="quality_supervised",
    )
    expected = sum(p.numel() for p in model.text_projection.parameters())
    expected += sum(p.numel() for p in model.vision_projection.parameters())
    expected += sum(p.numel() for p in model.gated_interaction_fusion.parameters())
    expected += sum(p.numel() for p in model.text_quality_estimator.parameters())
    expected += sum(p.numel() for p in model.vision_quality_estimator.parameters())
    assert counts["representation"] == expected
    assert counts["effective_total"] == expected + sum(
        p.numel() for p in classifier.parameters()
    )


def test_singleton_mismatch_is_rejected_if_sampled():
    batch = _fake_batch(1)
    text_std = torch.std(batch.text_embeddings, dim=0, unbiased=False)
    vision_std = torch.std(batch.vision_embeddings, dim=0, unbiased=False)

    # Find a deterministic seed whose first condition draw falls in mismatch.
    mismatch_seed = None
    for seed in range(1000):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if float(torch.rand(1, generator=generator).item()) >= 0.80:
            mismatch_seed = seed
            break
    assert mismatch_seed is not None

    with pytest.raises(ValueError, match="batch_size >= 2"):
        prepare_m4q_training_batch(
            batch,
            text_feature_std=text_std,
            vision_feature_std=vision_std,
            seed=mismatch_seed,
        )

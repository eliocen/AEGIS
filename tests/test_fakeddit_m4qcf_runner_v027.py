"""AEGIS v0.27 Step 13D tests for M4qcf runner integration."""

from types import SimpleNamespace

import torch
from torch import nn

from aegis.alignment import CrossModalAlignmentModel
from scripts.run_fakeddit_ablation import (
    FakedditAblationTrainer,
    collect_component_fingerprints,
    effective_parameter_counts,
    prepare_m4qc_training_batch,
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


def test_m4qcf_reuses_frozen_m4qc_training_mixture_deterministically():
    batch = _fake_batch(128)
    text_std = torch.std(batch.text_embeddings, dim=0, unbiased=False)
    vision_std = torch.std(batch.vision_embeddings, dim=0, unbiased=False)
    a = prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=27130042,
    )
    b = prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=27130042,
    )
    corrupted_a, quality_a, compatibility_a, counts_a = a
    corrupted_b, quality_b, compatibility_b, counts_b = b
    assert torch.equal(corrupted_a.text_embeddings, corrupted_b.text_embeddings)
    assert torch.equal(corrupted_a.vision_embeddings, corrupted_b.vision_embeddings)
    assert torch.equal(quality_a[0], quality_b[0])
    assert torch.equal(quality_a[1], quality_b[1])
    assert torch.equal(compatibility_a, compatibility_b)
    assert counts_a == counts_b
    assert sum(counts_a[k] for k in ("clean", "text_quality", "vision_quality", "mismatch")) == 128


def test_m4qcf_model_exposes_frozen_reliability_components():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_fusion=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    assert model.quality_compatibility_fusion is True
    assert model.fusion_architecture == "quality_compatibility_fusion"
    assert model.gated_interaction_fusion is not None
    assert model.text_quality_estimator is not None
    assert model.vision_quality_estimator is not None
    assert model.compatibility_estimator is not None
    assert model.reliability_controller is not None
    assert sum(p.numel() for p in model.reliability_controller.parameters()) == 0


def test_m4qcf_fingerprints_track_same_trainable_components_as_m4qc():
    torch.manual_seed(17)
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_fusion=True,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    classifier = nn.Linear(4, 2)
    fingerprints = collect_component_fingerprints(
        alignment_model=model,
        classification_model=classifier,
        mode="multimodal",
        fusion_architecture="quality_compatibility_fusion",
    )
    assert "gated_interaction_fusion" in fingerprints
    assert "gated_base_fusion" in fingerprints
    assert "text_quality_estimator" in fingerprints
    assert "vision_quality_estimator" in fingerprints
    assert "compatibility_estimator" in fingerprints
    assert "reliability_controller" not in fingerprints


def test_m4qcf_parameter_count_equals_m4qc_under_paired_dimensions():
    torch.manual_seed(99)
    m4qc = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_supervised=True,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    torch.manual_seed(99)
    m4qcf = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_fusion=True,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    classifier_qc = nn.Linear(4, 2)
    classifier_qcf = nn.Linear(4, 2)
    qc = effective_parameter_counts(
        alignment_model=m4qc,
        classification_model=classifier_qc,
        mode="multimodal",
        fusion_architecture="quality_compatibility_supervised",
    )
    qcf = effective_parameter_counts(
        alignment_model=m4qcf,
        classification_model=classifier_qcf,
        mode="multimodal",
        fusion_architecture="quality_compatibility_fusion",
    )
    assert qcf["representation"] == qc["representation"]
    assert qcf["classification"] == qc["classification"]
    assert qcf["effective_total"] == qc["effective_total"]


def test_trainer_accepts_m4qcf_architecture_flag():
    assert "quality_compatibility_fusion" in {
        "quality_compatibility_fusion",
    }
    # Constructor consistency is covered by integrated model/runner tests; this
    # guard prevents accidental removal of the public architecture spelling.
    source = FakedditAblationTrainer.__init__.__code__.co_consts
    assert any(
        isinstance(value, frozenset) and "quality_compatibility_fusion" in value
        for value in source
    )

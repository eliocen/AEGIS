"""
AEGIS v0.27 Step 11C M4qc runner integration tests.

Scope:
- frozen M4qc per-sample training mixture;
- deterministic cyclic vision mismatch;
- quality and compatibility targets;
- diagnostic-only M4qc loss integration;
- parameter/fingerprint accounting.

No formal H4-C/H5-C evaluation is performed here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from scripts.run_fakeddit_ablation import (
    FakedditAblationTrainer,
    collect_component_fingerprints,
    effective_parameter_counts,
    prepare_m4qc_training_batch,
)


class _Batch:
    def __init__(self, text, vision, targets):
        self.text_embeddings = text
        self.vision_embeddings = vision
        self.integrity_targets = targets
        self.sample_ids = [f"s{i}" for i in range(text.shape[0])]
        self.batch_size = int(text.shape[0])

    def validate(self, *, text_dim, vision_dim):
        assert self.text_embeddings.shape == (self.batch_size, text_dim)
        assert self.vision_embeddings.shape == (self.batch_size, vision_dim)
        assert self.integrity_targets.shape == (self.batch_size,)

    def to(self, device):
        return _Batch(
            self.text_embeddings.to(device),
            self.vision_embeddings.to(device),
            self.integrity_targets.to(device),
        )


def _fake_batch(batch_size=128):
    generator = torch.Generator().manual_seed(11131)
    text = torch.randn(batch_size, 8, generator=generator)
    vision = torch.randn(batch_size, 6, generator=generator)
    return SimpleNamespace(
        text_embeddings=text,
        vision_embeddings=vision,
        integrity_targets=torch.arange(batch_size) % 2,
        sample_ids=[f"s{i}" for i in range(batch_size)],
        batch_size=batch_size,
    )


def _prepare(seed=42, batch_size=128):
    batch = _fake_batch(batch_size)
    text_std = torch.std(batch.text_embeddings, dim=0, unbiased=False)
    vision_std = torch.std(batch.vision_embeddings, dim=0, unbiased=False)
    return batch, prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=seed,
    )


def _find_seed_with_mismatch(batch_size=64):
    for seed in range(5000):
        generator = torch.Generator().manual_seed(seed)
        condition = torch.rand(batch_size, generator=generator)
        if torch.any(condition >= 0.80):
            return seed
    raise AssertionError("could not find deterministic mismatch seed")


def _make_trainer():
    torch.manual_seed(22)
    alignment = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    assert alignment.gated_interaction_fusion is not None
    alignment.gated_interaction_fusion.gated_fusion.load_state_dict(
        alignment.fusion.state_dict()
    )

    classifier = HierarchicalInformationIntegrityClassifier(
        input_dim=4,
        hidden_dim=4,
        dropout=0.0,
    )

    optimizer = torch.optim.AdamW(
        list(alignment.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )

    return FakedditAblationTrainer(
        alignment_model=alignment,
        classification_model=classifier,
        optimizer=optimizer,
        device=torch.device("cpu"),
        alignment_loss_weight=0.0,
        classification_loss_weight=1.0,
        gradient_clip_norm=1.0,
        mode="multimodal",
        fusion_architecture="quality_compatibility_supervised",
    )


def test_prepare_m4qc_is_deterministic():
    _, first = _prepare(seed=87)
    _, second = _prepare(seed=87)

    batch_a, q_a, c_a, counts_a = first
    batch_b, q_b, c_b, counts_b = second

    assert torch.equal(batch_a.text_embeddings, batch_b.text_embeddings)
    assert torch.equal(batch_a.vision_embeddings, batch_b.vision_embeddings)
    assert torch.equal(q_a[0], q_b[0])
    assert torch.equal(q_a[1], q_b[1])
    assert torch.equal(c_a, c_b)
    assert counts_a == counts_b


def test_prepare_m4qc_does_not_mutate_source_batch():
    batch = _fake_batch(64)
    text_before = batch.text_embeddings.clone()
    vision_before = batch.vision_embeddings.clone()
    text_std = torch.std(text_before, dim=0, unbiased=False)
    vision_std = torch.std(vision_before, dim=0, unbiased=False)

    prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=31,
    )

    assert torch.equal(batch.text_embeddings, text_before)
    assert torch.equal(batch.vision_embeddings, vision_before)


def test_prepare_m4qc_targets_have_frozen_shapes_and_ranges():
    batch, (_, (q_text, q_vision), compatibility, _) = _prepare(
        seed=93,
        batch_size=96,
    )

    assert q_text.shape == (batch.batch_size, 1)
    assert q_vision.shape == (batch.batch_size, 1)
    assert compatibility.shape == (batch.batch_size, 1)

    for value in (q_text, q_vision, compatibility):
        assert torch.all(value >= 0.0)
        assert torch.all(value <= 1.0)

    assert torch.all(
        (compatibility == 0.0) | (compatibility == 1.0)
    )


def test_mismatch_has_q_targets_one_and_compatibility_zero():
    seed = _find_seed_with_mismatch()
    _, (_, (q_text, q_vision), compatibility, counts) = _prepare(
        seed=seed,
        batch_size=64,
    )

    mismatch = compatibility[:, 0] == 0.0
    assert int(mismatch.sum().item()) == counts["mismatch"]
    assert torch.any(mismatch)
    assert torch.all(q_text[mismatch] == 1.0)
    assert torch.all(q_vision[mismatch] == 1.0)


def test_non_mismatch_rows_have_compatibility_target_one():
    seed = _find_seed_with_mismatch()
    _, (_, _, compatibility, _) = _prepare(
        seed=seed,
        batch_size=64,
    )

    assert torch.all(
        compatibility[compatibility[:, 0] != 0.0] == 1.0
    )


def test_m4qc_mismatch_changes_only_vision_and_uses_one_cyclic_offset():
    seed = _find_seed_with_mismatch()
    source, (corrupted, _, compatibility, counts) = _prepare(
        seed=seed,
        batch_size=64,
    )

    mismatch_rows = torch.nonzero(
        compatibility[:, 0] == 0.0,
        as_tuple=False,
    ).reshape(-1)
    assert mismatch_rows.numel() == counts["mismatch"]

    # Mismatch corruption is vision-only. Other rows in the same mixed
    # training batch may legitimately receive text-quality corruption.
    assert torch.equal(
        corrupted.text_embeddings[mismatch_rows],
        source.text_embeddings[mismatch_rows],
    )

    donor_offsets = set()
    for receiver in mismatch_rows.tolist():
        matching = torch.nonzero(
            torch.all(
                source.vision_embeddings
                == corrupted.vision_embeddings[receiver],
                dim=1,
            ),
            as_tuple=False,
        ).reshape(-1)
        assert matching.numel() == 1
        donor = int(matching.item())
        offset = (donor - receiver) % source.batch_size
        assert offset != 0
        donor_offsets.add(offset)

    assert len(donor_offsets) == 1


def test_singleton_mismatch_fails_loudly():
    batch = _fake_batch(1)
    text_std = torch.std(batch.text_embeddings, dim=0, unbiased=False)
    vision_std = torch.std(batch.vision_embeddings, dim=0, unbiased=False)

    mismatch_seed = None
    for seed in range(1000):
        generator = torch.Generator().manual_seed(seed)
        if float(torch.rand(1, generator=generator).item()) >= 0.80:
            mismatch_seed = seed
            break
    assert mismatch_seed is not None

    with pytest.raises(ValueError, match="batch_size >= 2"):
        prepare_m4qc_training_batch(
            batch,
            text_feature_std=text_std,
            vision_feature_std=vision_std,
            seed=mismatch_seed,
        )


def test_m4qc_fingerprints_include_all_three_diagnostic_heads():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_supervised=True,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    classifier = nn.Linear(4, 2)

    fingerprints = collect_component_fingerprints(
        alignment_model=model,
        classification_model=classifier,
        mode="multimodal",
        fusion_architecture="quality_compatibility_supervised",
    )

    assert "gated_interaction_fusion" in fingerprints
    assert "gated_base_fusion" in fingerprints
    assert "text_quality_estimator" in fingerprints
    assert "vision_quality_estimator" in fingerprints
    assert "compatibility_estimator" in fingerprints


def test_m4qc_parameter_accounting_includes_compatibility_head():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_compatibility_supervised=True,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    classifier = nn.Linear(4, 2)

    counts = effective_parameter_counts(
        alignment_model=model,
        classification_model=classifier,
        mode="multimodal",
        fusion_architecture="quality_compatibility_supervised",
    )

    expected = sum(p.numel() for p in model.text_projection.parameters())
    expected += sum(p.numel() for p in model.vision_projection.parameters())
    expected += sum(p.numel() for p in model.gated_interaction_fusion.parameters())
    expected += sum(p.numel() for p in model.text_quality_estimator.parameters())
    expected += sum(p.numel() for p in model.vision_quality_estimator.parameters())
    expected += sum(p.numel() for p in model.compatibility_estimator.parameters())

    assert counts["representation"] == expected


def test_forward_batch_adds_compatibility_bce_without_feeding_fusion():
    trainer = _make_trainer()

    generator = torch.Generator().manual_seed(180)
    batch = _Batch(
        torch.randn(8, 8, generator=generator),
        torch.randn(8, 6, generator=generator),
        torch.arange(8) % 2,
    )
    q_targets = (
        torch.ones(8, 1),
        torch.ones(8, 1),
    )
    c_targets = torch.tensor(
        [[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]
    )

    outputs = trainer.forward_batch(
        batch,
        compute_alignment_loss=False,
        quality_targets=q_targets,
        quality_loss_weight=1.0,
        compatibility_targets=c_targets,
        compatibility_loss_weight=1.0,
    )

    compatibility_score = outputs["alignment_outputs"]["compatibility_score"]
    expected_bce = nn.functional.binary_cross_entropy(
        compatibility_score,
        c_targets.to(compatibility_score),
    )

    assert torch.allclose(
        outputs["compatibility_loss"],
        expected_bce,
    )
    expected_total = (
        outputs["classification_loss"]
        + outputs["quality_loss"]
        + outputs["compatibility_loss"]
    )
    assert torch.allclose(outputs["loss"], expected_total)


def test_compatibility_targets_rejected_outside_m4qc():
    torch.manual_seed(3)
    alignment = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_supervised=True,
        quality_dropout=0.0,
    )
    assert alignment.gated_interaction_fusion is not None
    alignment.gated_interaction_fusion.gated_fusion.load_state_dict(
        alignment.fusion.state_dict()
    )
    classifier = HierarchicalInformationIntegrityClassifier(
        input_dim=4,
        hidden_dim=4,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(
        list(alignment.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    trainer = FakedditAblationTrainer(
        alignment_model=alignment,
        classification_model=classifier,
        optimizer=optimizer,
        device=torch.device("cpu"),
        alignment_loss_weight=0.0,
        classification_loss_weight=1.0,
        gradient_clip_norm=1.0,
        mode="multimodal",
        fusion_architecture="quality_supervised",
    )

    generator = torch.Generator().manual_seed(4)
    batch = _Batch(
        torch.randn(4, 8, generator=generator),
        torch.randn(4, 6, generator=generator),
        torch.arange(4) % 2,
    )

    with pytest.raises(
        ValueError,
        match="compatibility_targets are valid only for M4qc",
    ):
        trainer.forward_batch(
            batch,
            compute_alignment_loss=False,
            quality_targets=(torch.ones(4, 1), torch.ones(4, 1)),
            quality_loss_weight=1.0,
            compatibility_targets=torch.ones(4, 1),
            compatibility_loss_weight=1.0,
        )

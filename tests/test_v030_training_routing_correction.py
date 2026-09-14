"""AEGIS v0.30 Step5A training-routing correction tests."""

from __future__ import annotations

import inspect

import pytest
import torch

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from aegis.training import BinaryIntegrityBatch
from scripts import run_fakeddit_ablation as runner


ARCHITECTURES = {
    "quality_compatibility_selective_intervention_weights": "weights_only",
    "quality_compatibility_selective_intervention_transition": "transition_only",
    "quality_compatibility_selective_intervention": "combined",
}


def _make_batch(n=8):
    torch.manual_seed(50301)
    return BinaryIntegrityBatch(
        text_embeddings=torch.randn(n, 768),
        vision_embeddings=torch.randn(n, 512),
        integrity_targets=torch.tensor([0, 1] * (n // 2), dtype=torch.long),
        sample_ids=[f"v030-routing-{i}" for i in range(n)],
    )


def _make_trainer(architecture):
    torch.manual_seed(50300)
    model = CrossModalAlignmentModel(
        shared_dim=16,
        dropout=0.0,
        quality_compatibility_selective_intervention=ARCHITECTURES[architecture],
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    classifier = HierarchicalInformationIntegrityClassifier(
        input_dim=16,
        hidden_dim=8,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    return runner.FakedditAblationTrainer(
        alignment_model=model,
        classification_model=classifier,
        optimizer=optimizer,
        device=torch.device("cpu"),
        alignment_loss_weight=0.5,
        classification_loss_weight=1.0,
        gradient_clip_norm=1.0,
        mode="multimodal",
        fusion_architecture=architecture,
    )


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_forward_batch_accepts_v030_quality_and_compatibility_targets(architecture):
    trainer = _make_trainer(architecture)
    batch = _make_batch()

    (
        prepared,
        quality_targets,
        compatibility_targets,
        quality_sample_weights,
        _,
    ) = runner.prepare_m4qcs_training_batch(
        batch,
        text_feature_std=torch.ones(768),
        vision_feature_std=torch.ones(512),
        seed=27_000_042,
    )

    out = trainer.forward_batch(
        prepared,
        compute_alignment_loss=True,
        quality_targets=quality_targets,
        quality_loss_weight=1.0,
        quality_sample_weights=quality_sample_weights,
        compatibility_targets=compatibility_targets,
        compatibility_loss_weight=1.0,
        transition_reference_batch=None,
        transition_loss_weight=0.0,
    )

    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["quality_loss"])
    assert torch.isfinite(out["compatibility_loss"])
    assert float(out["transition_loss"].detach().cpu()) == 0.0


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_train_one_epoch_uses_v030_auxiliary_supervision(architecture):
    trainer = _make_trainer(architecture)

    metrics = runner.train_one_epoch(
        trainer=trainer,
        full_batch=_make_batch(),
        batch_size=8,
        epoch=1,
        seed=42,
        quality_loss_weight=1.0,
        compatibility_loss_weight=1.0,
        transition_loss_weight=0.0,
        text_feature_std=torch.ones(768),
        vision_feature_std=torch.ones(512),
    )

    assert metrics["sample_count"] == 8
    assert metrics["batch_count"] == 1
    assert metrics["quality_loss"] > 0.0
    assert metrics["compatibility_loss"] > 0.0
    assert metrics["transition_loss"] == 0.0
    assert metrics["corruption_counts"]


def test_v030_routing_present_in_all_critical_regions():
    forward = inspect.getsource(runner.FakedditAblationTrainer.forward_batch)
    epoch = inspect.getsource(runner.train_one_epoch)
    main = inspect.getsource(runner.main)

    for architecture in ARCHITECTURES:
        assert architecture in forward
        assert architecture in epoch
        assert architecture in main


def test_v029_transition_objective_remains_isolated():
    epoch = inspect.getsource(runner.train_one_epoch)
    anchor = "transition_reference_batch=("
    assert anchor in epoch
    window = epoch[epoch.index(anchor):][:2200]

    assert "quality_compatibility_transition_control" in window
    assert "quality_compatibility_graded_transition_fusion" in window

    for architecture in ARCHITECTURES:
        assert architecture not in window
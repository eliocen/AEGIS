"""AEGIS v0.30 Step3 runner/ablation integration tests."""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from scripts import run_fakeddit_ablation as runner


ARCHITECTURES = {
    "quality_compatibility_selective_intervention_weights": "weights_only",
    "quality_compatibility_selective_intervention_transition": "transition_only",
    "quality_compatibility_selective_intervention": "combined",
}


def test_v030_runner_mapping_is_exact():
    assert runner.V030_SELECTIVE_INTERVENTION_ARCHITECTURE_MODES == ARCHITECTURES
    assert runner.V030_SELECTIVE_INTERVENTION_ARCHITECTURES == frozenset(
        ARCHITECTURES
    )
    for architecture, mode in ARCHITECTURES.items():
        assert runner.v030_selective_intervention_mode(architecture) == mode
        assert runner.is_v030_selective_intervention_architecture(architecture)
        assert runner.requires_quality_feature_statistics(architecture)
    assert runner.v030_selective_intervention_mode("legacy") is None
    assert not runner.is_v030_selective_intervention_architecture("legacy")


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_v030_runner_cli_accepts_exact_architectures(
    architecture,
    monkeypatch,
    tmp_path,
):
    train_cache = tmp_path / "train"
    validation_cache = tmp_path / "validation"
    train_cache.mkdir()
    validation_cache.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fakeddit_ablation.py",
            "--mode",
            "multimodal",
            "--fusion-architecture",
            architecture,
            "--train-cache",
            str(train_cache),
            "--validation-cache",
            str(validation_cache),
        ],
    )
    args = runner.parse_args()
    runner.validate_args(args)
    assert args.fusion_architecture == architecture


@pytest.mark.parametrize("architecture,mode", sorted(ARCHITECTURES.items()))
def test_v030_runner_model_and_trainer_wiring_is_exact(architecture, mode):
    torch.manual_seed(300)
    alignment_model = CrossModalAlignmentModel(
        shared_dim=16,
        dropout=0.0,
        quality_compatibility_selective_intervention=(
            runner.v030_selective_intervention_mode(architecture)
        ),
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
        list(alignment_model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    trainer = runner.FakedditAblationTrainer(
        alignment_model=alignment_model,
        classification_model=classifier,
        optimizer=optimizer,
        device=torch.device("cpu"),
        alignment_loss_weight=0.5,
        classification_loss_weight=1.0,
        gradient_clip_norm=1.0,
        mode="multimodal",
        fusion_architecture=architecture,
    )
    assert trainer.fusion_architecture == architecture
    assert (
        alignment_model.quality_compatibility_selective_intervention
        == mode
    )
    assert alignment_model.selective_intervention_controller is not None
    assert alignment_model.selective_intervention_controller.parameter_count == 0


@pytest.mark.parametrize("architecture,mode", sorted(ARCHITECTURES.items()))
def test_v030_runner_synthetic_forward_smoke(architecture, mode):
    torch.manual_seed(301)
    model = CrossModalAlignmentModel(
        shared_dim=16,
        dropout=0.0,
        quality_compatibility_selective_intervention=mode,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    ).eval()

    out = model(
        torch.randn(3, 768),
        torch.randn(3, 512),
        compute_loss=False,
    )

    assert out["fused_embedding"].shape == (3, 16)
    assert out["intervention_gate"].shape == (3, 1)
    assert out["transition_risk"].shape == (3, 1)
    assert out["weight_intervention_magnitude"].shape == (3, 1)
    assert out["transition_intervention_magnitude"].shape == (3, 1)
    assert torch.isfinite(out["fused_embedding"]).all()

    description = runner.mode_description("multimodal", architecture)
    assert "M4qesri" in description


def test_v030_runner_does_not_activate_v029_transition_objective():
    source = inspect.getsource(runner.train_one_epoch)
    anchor = "transition_reference_batch=("
    assert anchor in source
    transition_window = source[source.index(anchor):]
    transition_window = transition_window[:1800]

    assert "quality_compatibility_transition_control" in transition_window
    assert "quality_compatibility_graded_transition_fusion" in transition_window
    for architecture in ARCHITECTURES:
        assert architecture not in transition_window

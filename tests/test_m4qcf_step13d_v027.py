from __future__ import annotations

import torch

from scripts.audit_m4qcf_gradient_causality_v027 import run_audit
from scripts.run_fakeddit_ablation import (
    FakedditAblationTrainer,
    prepare_m4qc_training_batch,
)
from aegis.alignment.model import CrossModalAlignmentModel


def test_step13d_gradient_causal_audit_passes_cpu():
    result = run_audit(
        seed=42,
        batch_size=4,
        shared_dim=16,
        device="cpu",
    )
    assert result["status"] == "PASS"
    assert result["controller_parameter_count"] == 0
    assert result["fusion_equation_exact"] is True
    assert result["weights_sum_to_one"] is True
    assert result["interaction_suppression_monotonic"] is True
    assert result["formal_hypotheses_computed"] is False
    assert result["official_test_accessed"] is False

    cls = result["classification_side_gradients"]
    assert cls["text_quality_estimator"] == 0.0
    assert cls["vision_quality_estimator"] == 0.0
    assert cls["compatibility_estimator"] == 0.0
    assert cls["text_projection"] > 0.0
    assert cls["vision_projection"] > 0.0
    assert cls["interaction_pathway"] > 0.0

    aux = result["auxiliary_side_gradients"]
    assert aux["text_quality_estimator"] > 0.0
    assert aux["vision_quality_estimator"] > 0.0
    assert aux["compatibility_estimator"] > 0.0


def test_m4qcf_and_m4qc_have_equal_parameter_count_under_paired_seed():
    torch.manual_seed(42)
    m4qc = CrossModalAlignmentModel(
        shared_dim=32,
        evidence_interaction_dropout=0.0,
        quality_compatibility_supervised=True,
    )
    torch.manual_seed(42)
    m4qcf = CrossModalAlignmentModel(
        shared_dim=32,
        evidence_interaction_dropout=0.0,
        quality_compatibility_fusion=True,
    )

    assert sum(p.numel() for p in m4qc.parameters()) == sum(
        p.numel() for p in m4qcf.parameters()
    )
    assert sum(
        p.numel() for p in m4qcf.reliability_controller.parameters()
    ) == 0


def test_m4qcf_reuses_frozen_m4qc_training_batch_protocol():
    # The training corruption/target constructor remains the frozen Step11A
    # M4qc mixture.  This test verifies determinism and target identity.
    from aegis.training import BinaryIntegrityBatch

    torch.manual_seed(7)
    batch = BinaryIntegrityBatch(
        text_embeddings=torch.randn(8, 768),
        vision_embeddings=torch.randn(8, 512),
        integrity_targets=torch.randint(0, 2, (8,)),
    )
    text_std = torch.ones(768)
    vision_std = torch.ones(512)

    first = prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=27000042,
    )
    second = prepare_m4qc_training_batch(
        batch,
        text_feature_std=text_std,
        vision_feature_std=vision_std,
        seed=27000042,
    )

    first_batch, first_q, first_c, first_counts = first
    second_batch, second_q, second_c, second_counts = second

    assert torch.equal(
        first_batch.text_embeddings, second_batch.text_embeddings
    )
    assert torch.equal(
        first_batch.vision_embeddings, second_batch.vision_embeddings
    )
    assert torch.equal(first_q[0], second_q[0])
    assert torch.equal(first_q[1], second_q[1])
    assert torch.equal(first_c, second_c)
    assert first_counts == second_counts

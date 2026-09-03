"""
Tests for AEGIS v0.25.0 evidence-aware alignment integration.

These tests verify that:

1. legacy v0.24 behavior remains the default;
2. evidence-aware integration is explicitly opt-in;
3. the stable downstream alignment-model contract is preserved;
4. evidence-aware diagnostics are correctly exposed;
5. adaptive modality weights form a valid distribution;
6. the contrastive alignment objective remains available;
7. gradients propagate through the new evidence subsystem.
"""

import pytest
import torch

from aegis.alignment import (
    CrossModalAlignmentModel,
)


class TestEvidenceAwareAlignmentIntegration:

    def test_legacy_mode_is_default(self):
        model = CrossModalAlignmentModel(
            dropout=0.0,
        )

        assert model.evidence_aware is False
        assert model.evidence_integration is None

    def test_explicit_legacy_mode_matches_default_mode(self):
        default_model = CrossModalAlignmentModel(
            dropout=0.0,
        )

        explicit_legacy_model = (
            CrossModalAlignmentModel(
                dropout=0.0,
                evidence_aware=False,
            )
        )

        explicit_legacy_model.load_state_dict(
            default_model.state_dict()
        )

        default_model.eval()
        explicit_legacy_model.eval()

        text = torch.randn(4, 768)
        vision = torch.randn(4, 512)

        with torch.no_grad():
            default_output = default_model(
                text,
                vision,
                compute_loss=True,
            )

            explicit_output = (
                explicit_legacy_model(
                    text,
                    vision,
                    compute_loss=True,
                )
            )

        assert set(default_output.keys()) == {
            "aligned_text",
            "aligned_vision",
            "fused_embedding",
            "alignment_loss",
        }

        assert (
            set(explicit_output.keys())
            == set(default_output.keys())
        )

        for key in (
            "aligned_text",
            "aligned_vision",
            "fused_embedding",
            "alignment_loss",
        ):
            assert torch.equal(
                default_output[key],
                explicit_output[key],
            )

    def test_evidence_aware_mode_is_initialized(self):
        model = CrossModalAlignmentModel(
            dropout=0.0,
            evidence_aware=True,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        assert model.evidence_aware is True
        assert model.evidence_integration is not None

    def test_evidence_aware_output_preserves_stable_contract(self):
        model = CrossModalAlignmentModel(
            dropout=0.0,
            evidence_aware=True,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        output = model(
            torch.randn(3, 768),
            torch.randn(3, 512),
            compute_loss=True,
        )

        for key in (
            "aligned_text",
            "aligned_vision",
            "fused_embedding",
            "alignment_loss",
        ):
            assert key in output

        assert output["aligned_text"].shape == (
            3,
            512,
        )

        assert output["aligned_vision"].shape == (
            3,
            512,
        )

        assert output["fused_embedding"].shape == (
            3,
            512,
        )

        assert output["alignment_loss"].ndim == 0

        assert torch.isfinite(
            output["alignment_loss"]
        )

    def test_evidence_aware_diagnostics_are_exposed(self):
        model = CrossModalAlignmentModel(
            shared_dim=8,
            dropout=0.0,
            evidence_aware=True,
            evidence_reliability_hidden_dim=4,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        output = model(
            torch.randn(5, 768),
            torch.randn(5, 512),
        )

        expected_diagnostic_keys = {
            "evidence_difference",
            "evidence_product",
            "cosine_similarity",
            "interaction_embedding",
            "text_reliability",
            "vision_reliability",
            "text_weight",
            "vision_weight",
            "evidence_weights",
        }

        assert expected_diagnostic_keys.issubset(
            output.keys()
        )

        assert output[
            "evidence_difference"
        ].shape == (5, 8)

        assert output[
            "evidence_product"
        ].shape == (5, 8)

        assert output[
            "cosine_similarity"
        ].shape == (5, 1)

        assert output[
            "interaction_embedding"
        ].shape == (5, 8)

        assert output[
            "text_reliability"
        ].shape == (5, 1)

        assert output[
            "vision_reliability"
        ].shape == (5, 1)

        assert output[
            "text_weight"
        ].shape == (5, 1)

        assert output[
            "vision_weight"
        ].shape == (5, 1)

        assert output[
            "evidence_weights"
        ].shape == (5, 2)

    def test_evidence_weights_sum_to_one(self):
        model = CrossModalAlignmentModel(
            shared_dim=8,
            dropout=0.0,
            evidence_aware=True,
            evidence_reliability_hidden_dim=4,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        output = model(
            torch.randn(6, 768),
            torch.randn(6, 512),
        )

        weight_sums = (
            output["evidence_weights"]
            .sum(dim=-1)
        )

        assert torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-6,
        )

    def test_reliability_values_are_probabilities(self):
        model = CrossModalAlignmentModel(
            shared_dim=8,
            dropout=0.0,
            evidence_aware=True,
            evidence_reliability_hidden_dim=4,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        output = model(
            torch.randn(4, 768),
            torch.randn(4, 512),
        )

        for key in (
            "text_reliability",
            "vision_reliability",
        ):
            assert torch.all(
                output[key] >= 0.0
            )

            assert torch.all(
                output[key] <= 1.0
            )

    def test_alignment_loss_remains_optional(self):
        model = CrossModalAlignmentModel(
            dropout=0.0,
            evidence_aware=True,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        output_without_loss = model(
            torch.randn(3, 768),
            torch.randn(3, 512),
            compute_loss=False,
        )

        assert (
            "alignment_loss"
            not in output_without_loss
        )

        output_with_loss = model(
            torch.randn(3, 768),
            torch.randn(3, 512),
            compute_loss=True,
        )

        assert "alignment_loss" in output_with_loss

    def test_evidence_aware_gradients_reach_new_subsystem(self):
        model = CrossModalAlignmentModel(
            shared_dim=8,
            dropout=0.0,
            evidence_aware=True,
            evidence_reliability_hidden_dim=4,
            evidence_interaction_dropout=0.0,
            evidence_reliability_dropout=0.0,
        )

        text = torch.randn(
            4,
            768,
            requires_grad=True,
        )

        vision = torch.randn(
            4,
            512,
            requires_grad=True,
        )

        output = model(
            text,
            vision,
            compute_loss=True,
        )

        loss = (
            output["fused_embedding"]
            .pow(2)
            .mean()
            + output["alignment_loss"]
        )

        loss.backward()

        assert text.grad is not None
        assert vision.grad is not None

        evidence_parameters = [
            parameter
            for parameter in (
                model.evidence_integration
                .parameters()
            )
            if parameter.requires_grad
        ]

        assert evidence_parameters

        assert all(
            parameter.grad is not None
            for parameter in evidence_parameters
        )

    def test_invalid_evidence_aware_type_fails(self):
        with pytest.raises(TypeError):
            CrossModalAlignmentModel(
                evidence_aware="yes",
            )
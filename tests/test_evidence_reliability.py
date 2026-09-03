import pytest
import torch

from aegis.alignment.reliability import (
    EvidenceReliabilityEstimator,
)


class TestEvidenceReliabilityEstimator:

    def test_output_shape(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
            hidden_dim=4,
            dropout=0.0,
        )

        modality = torch.randn(5, 8)
        interaction = torch.randn(5, 8)
        cosine = torch.randn(5, 1)

        reliability = model(
            modality,
            interaction,
            cosine,
        )

        assert reliability.shape == (5, 1)

    def test_output_is_bounded_probability(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
            hidden_dim=4,
            dropout=0.0,
        )

        modality = torch.randn(10, 8)
        interaction = torch.randn(10, 8)
        cosine = torch.randn(10, 1)

        reliability = model(
            modality,
            interaction,
            cosine,
        )

        assert torch.all(reliability >= 0.0)
        assert torch.all(reliability <= 1.0)

    def test_batch_size_one(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
            hidden_dim=4,
            dropout=0.0,
        )

        reliability = model(
            torch.randn(1, 8),
            torch.randn(1, 8),
            torch.randn(1, 1),
        )

        assert reliability.shape == (1, 1)

    def test_modality_rank_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(8),
                torch.randn(1, 8),
                torch.randn(1, 1),
            )

    def test_interaction_rank_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(1, 8),
                torch.randn(8),
                torch.randn(1, 1),
            )

    def test_embedding_dimension_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 7),
                torch.randn(2, 7),
                torch.randn(2, 1),
            )

    def test_embedding_shape_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(3, 8),
                torch.randn(2, 1),
            )

    def test_cosine_rank_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.randn(2),
            )

    def test_cosine_feature_dimension_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.randn(2, 2),
            )

    def test_cosine_batch_mismatch_fails(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.randn(3, 1),
            )

    def test_invalid_dimension_fails(self):
        with pytest.raises(ValueError):
            EvidenceReliabilityEstimator(
                dimension=0,
            )

    def test_invalid_hidden_dimension_fails(self):
        with pytest.raises(ValueError):
            EvidenceReliabilityEstimator(
                dimension=8,
                hidden_dim=0,
            )

    def test_invalid_dropout_fails(self):
        with pytest.raises(ValueError):
            EvidenceReliabilityEstimator(
                dimension=8,
                dropout=1.0,
            )

    def test_gradients_flow_through_estimator(self):
        model = EvidenceReliabilityEstimator(
            dimension=8,
            hidden_dim=4,
            dropout=0.0,
        )

        modality = torch.randn(
            4,
            8,
            requires_grad=True,
        )

        interaction = torch.randn(
            4,
            8,
            requires_grad=True,
        )

        cosine = torch.randn(
            4,
            1,
            requires_grad=True,
        )

        reliability = model(
            modality,
            interaction,
            cosine,
        )

        loss = reliability.mean()
        loss.backward()

        assert modality.grad is not None
        assert interaction.grad is not None
        assert cosine.grad is not None

        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        assert all(
            gradient is not None
            for gradient in trainable_gradients
        )

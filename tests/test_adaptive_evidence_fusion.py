import pytest
import torch

from aegis.alignment.adaptive_fusion import (
    ReliabilityAwareAdaptiveFusion,
)


class TestReliabilityAwareAdaptiveFusion:

    def test_output_shapes(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        first = torch.randn(4, 8)
        second = torch.randn(4, 8)

        first_reliability = torch.rand(4, 1)
        second_reliability = torch.rand(4, 1)

        output = model(
            first,
            second,
            first_reliability,
            second_reliability,
        )

        assert output["fused_embedding"].shape == (4, 8)
        assert output["first_weight"].shape == (4, 1)
        assert output["second_weight"].shape == (4, 1)
        assert output["weights"].shape == (4, 2)

    def test_weights_sum_to_one(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        output = model(
            torch.randn(5, 8),
            torch.randn(5, 8),
            torch.rand(5, 1),
            torch.rand(5, 1),
        )

        weight_sums = output["weights"].sum(
            dim=-1
        )

        assert torch.allclose(
            weight_sums,
            torch.ones_like(weight_sums),
            atol=1e-6,
        )

    def test_equal_reliability_produces_equal_weights(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=4,
            temperature=1.0,
        )

        reliability = torch.full(
            (3, 1),
            0.5,
        )

        output = model(
            torch.randn(3, 4),
            torch.randn(3, 4),
            reliability,
            reliability,
        )

        expected = torch.full(
            (3, 2),
            0.5,
        )

        assert torch.allclose(
            output["weights"],
            expected,
            atol=1e-6,
        )

    def test_higher_reliability_receives_higher_weight(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=4,
        )

        first_reliability = torch.tensor(
            [[0.9]]
        )

        second_reliability = torch.tensor(
            [[0.1]]
        )

        output = model(
            torch.randn(1, 4),
            torch.randn(1, 4),
            first_reliability,
            second_reliability,
        )

        assert (
            output["first_weight"].item()
            > output["second_weight"].item()
        )

    def test_fusion_matches_weighted_sum(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=2,
        )

        first = torch.tensor(
            [[1.0, 2.0]]
        )

        second = torch.tensor(
            [[3.0, 4.0]]
        )

        output = model(
            first,
            second,
            torch.tensor([[0.8]]),
            torch.tensor([[0.2]]),
        )

        expected = (
            output["first_weight"] * first
            + output["second_weight"] * second
        )

        assert torch.allclose(
            output["fused_embedding"],
            expected,
            atol=1e-6,
        )

    def test_zero_reliability_scores_still_form_valid_distribution(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=4,
        )

        zeros = torch.zeros(2, 1)

        output = model(
            torch.randn(2, 4),
            torch.randn(2, 4),
            zeros,
            zeros,
        )

        expected = torch.full(
            (2, 2),
            0.5,
        )

        assert torch.allclose(
            output["weights"],
            expected,
            atol=1e-6,
        )

    def test_embedding_shape_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(3, 8),
                torch.rand(2, 1),
                torch.rand(3, 1),
            )

    def test_embedding_dimension_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 7),
                torch.randn(2, 7),
                torch.rand(2, 1),
                torch.rand(2, 1),
            )

    def test_embedding_rank_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(8),
                torch.randn(8),
                torch.rand(1, 1),
                torch.rand(1, 1),
            )

    def test_reliability_rank_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.rand(2),
                torch.rand(2, 1),
            )

    def test_reliability_feature_dimension_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.rand(2, 2),
                torch.rand(2, 1),
            )

    def test_reliability_batch_mismatch_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(2, 8),
                torch.randn(2, 8),
                torch.rand(3, 1),
                torch.rand(2, 1),
            )

    def test_negative_reliability_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=4,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(1, 4),
                torch.randn(1, 4),
                torch.tensor([[-0.1]]),
                torch.tensor([[0.5]]),
            )

    def test_reliability_above_one_fails(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=4,
        )

        with pytest.raises(ValueError):
            model(
                torch.randn(1, 4),
                torch.randn(1, 4),
                torch.tensor([[1.1]]),
                torch.tensor([[0.5]]),
            )

    def test_invalid_dimension_fails(self):
        with pytest.raises(ValueError):
            ReliabilityAwareAdaptiveFusion(
                dimension=0,
            )

    def test_invalid_temperature_fails(self):
        with pytest.raises(ValueError):
            ReliabilityAwareAdaptiveFusion(
                dimension=8,
                temperature=0.0,
            )

    def test_gradients_flow_through_embeddings_and_reliabilities(self):
        model = ReliabilityAwareAdaptiveFusion(
            dimension=8,
        )

        first = torch.randn(
            4,
            8,
            requires_grad=True,
        )

        second = torch.randn(
            4,
            8,
            requires_grad=True,
        )

        first_logits = torch.randn(
            4,
            1,
            requires_grad=True,
        )

        second_logits = torch.randn(
            4,
            1,
            requires_grad=True,
        )

        first_reliability = torch.sigmoid(
            first_logits
        )

        second_reliability = torch.sigmoid(
            second_logits
        )

        output = model(
            first,
            second,
            first_reliability,
            second_reliability,
        )

        loss = (
            output["fused_embedding"]
            .pow(2)
            .mean()
        )

        loss.backward()

        assert first.grad is not None
        assert second.grad is not None
        assert first_logits.grad is not None
        assert second_logits.grad is not None
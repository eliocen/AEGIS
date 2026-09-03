import pytest
import torch

from aegis.alignment.interaction import (
    CrossModalEvidenceInteraction,
)


class TestCrossModalEvidenceInteraction:

    def test_output_shapes(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
            dropout=0.0,
        )

        first = torch.randn(4, 8)
        second = torch.randn(4, 8)

        output = model(
            first,
            second,
        )

        assert output["difference"].shape == (4, 8)
        assert output["product"].shape == (4, 8)
        assert output["cosine_similarity"].shape == (4, 1)
        assert output["interaction_embedding"].shape == (4, 8)

    def test_identical_embeddings_have_zero_difference(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
            dropout=0.0,
        )

        embedding = torch.randn(3, 8)

        output = model(
            embedding,
            embedding,
        )

        assert torch.allclose(
            output["difference"],
            torch.zeros_like(embedding),
        )

    def test_identical_embeddings_have_unit_cosine_similarity(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
            dropout=0.0,
        )

        embedding = torch.randn(3, 8)

        output = model(
            embedding,
            embedding,
        )

        expected = torch.ones(
            3,
            1,
        )

        assert torch.allclose(
            output["cosine_similarity"],
            expected,
            atol=1e-6,
        )

    def test_product_is_elementwise(self):
        model = CrossModalEvidenceInteraction(
            dimension=4,
            dropout=0.0,
        )

        first = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]]
        )

        second = torch.tensor(
            [[2.0, 3.0, 4.0, 5.0]]
        )

        output = model(
            first,
            second,
        )

        expected = torch.tensor(
            [[2.0, 6.0, 12.0, 20.0]]
        )

        assert torch.equal(
            output["product"],
            expected,
        )

    def test_shape_mismatch_fails(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
        )

        first = torch.randn(2, 8)
        second = torch.randn(3, 8)

        with pytest.raises(ValueError):
            model(
                first,
                second,
            )

    def test_dimension_mismatch_fails(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
        )

        first = torch.randn(2, 7)
        second = torch.randn(2, 7)

        with pytest.raises(ValueError):
            model(
                first,
                second,
            )

    def test_rank_mismatch_fails(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
        )

        first = torch.randn(8)
        second = torch.randn(8)

        with pytest.raises(ValueError):
            model(
                first,
                second,
            )

    def test_invalid_dimension_fails(self):
        with pytest.raises(ValueError):
            CrossModalEvidenceInteraction(
                dimension=0,
            )

    def test_invalid_dropout_fails(self):
        with pytest.raises(ValueError):
            CrossModalEvidenceInteraction(
                dimension=8,
                dropout=1.0,
            )

    def test_gradients_flow_through_interaction_projection(self):
        model = CrossModalEvidenceInteraction(
            dimension=8,
            dropout=0.0,
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

        output = model(
            first,
            second,
        )

        loss = (
            output["interaction_embedding"]
            .pow(2)
            .mean()
        )

        loss.backward()

        assert first.grad is not None
        assert second.grad is not None

        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        assert all(
                gradient is not None
                for gradient in trainable_gradients
            )

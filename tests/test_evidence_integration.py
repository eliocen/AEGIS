import torch

from aegis.alignment.evidence_integration import (
    AEGISEvidenceIntegrationBlock,
)


class TestAEGISEvidenceIntegrationBlock:

    def test_complete_evidence_state_is_returned(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        output = model(
            torch.randn(3, 8),
            torch.randn(3, 8),
        )

        expected_keys = {
            "first_embedding",
            "second_embedding",
            "difference",
            "product",
            "cosine_similarity",
            "interaction_embedding",
            "first_reliability",
            "second_reliability",
            "first_weight",
            "second_weight",
            "weights",
            "fused_embedding",
        }

        assert set(output.keys()) == expected_keys

    def test_evidence_state_shapes(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        output = model(
            torch.randn(5, 8),
            torch.randn(5, 8),
        )

        assert output["first_embedding"].shape == (5, 8)
        assert output["second_embedding"].shape == (5, 8)
        assert output["difference"].shape == (5, 8)
        assert output["product"].shape == (5, 8)
        assert output["cosine_similarity"].shape == (5, 1)
        assert output["interaction_embedding"].shape == (5, 8)

        assert output["first_reliability"].shape == (5, 1)
        assert output["second_reliability"].shape == (5, 1)

        assert output["first_weight"].shape == (5, 1)
        assert output["second_weight"].shape == (5, 1)
        assert output["weights"].shape == (5, 2)

        assert output["fused_embedding"].shape == (5, 8)

    def test_reliabilities_are_probabilities(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        output = model(
            torch.randn(6, 8),
            torch.randn(6, 8),
        )

        for key in (
            "first_reliability",
            "second_reliability",
        ):
            assert torch.all(output[key] >= 0.0)
            assert torch.all(output[key] <= 1.0)

    def test_adaptive_weights_sum_to_one(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        output = model(
            torch.randn(7, 8),
            torch.randn(7, 8),
        )

        sums = output["weights"].sum(
            dim=-1
        )

        assert torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=1e-6,
        )

    def test_fused_embedding_matches_reported_weights(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        first = torch.randn(4, 8)
        second = torch.randn(4, 8)

        output = model(
            first,
            second,
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

    def test_interaction_difference_is_preserved(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=4,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        first = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]]
        )

        second = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0]]
        )

        output = model(
            first,
            second,
        )

        expected = torch.tensor(
            [[3.0, 1.0, 1.0, 3.0]]
        )

        assert torch.equal(
            output["difference"],
            expected,
        )

    def test_identical_embeddings_report_unit_cosine(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
        )

        embedding = torch.randn(4, 8)

        output = model(
            embedding,
            embedding,
        )

        assert torch.allclose(
            output["cosine_similarity"],
            torch.ones(4, 1),
            atol=1e-6,
        )

    def test_separate_reliability_estimators_are_used(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
        )

        assert (
            model.first_reliability_estimator
            is not model.second_reliability_estimator
        )

        first_parameters = {
            id(parameter)
            for parameter in (
                model.first_reliability_estimator
                .parameters()
            )
        }

        second_parameters = {
            id(parameter)
            for parameter in (
                model.second_reliability_estimator
                .parameters()
            )
        }

        assert first_parameters.isdisjoint(
            second_parameters
        )

    def test_gradients_flow_through_integrated_architecture(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
            reliability_hidden_dim=4,
            interaction_dropout=0.0,
            reliability_dropout=0.0,
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
            output["fused_embedding"]
            .pow(2)
            .mean()
            + output["interaction_embedding"]
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

    def test_invalid_input_shape_propagates_validation(self):
        model = AEGISEvidenceIntegrationBlock(
            dimension=8,
        )

        try:
            model(
                torch.randn(2, 8),
                torch.randn(3, 8),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError for mismatched "
                "modality shapes."
            )
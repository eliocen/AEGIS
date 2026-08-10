"""
Tests for AEGIS v0.11.0:
Hierarchical Information Integrity Classification.
"""

import unittest

import torch

from aegis.alignment import (
    MultimodalRepresentation,
)

from aegis.classification import (
    CognitiveThreatType,
    HierarchicalClassificationLayer,
    HierarchicalInformationIntegrityClassifier,
    IntegrityStatus,
    WeightedHierarchicalLoss,
)

from aegis.context import (
    ContextualizedRepresentation,
)


class TestHierarchicalClassifier(
    unittest.TestCase
):

    def test_output_shapes(self):

        model = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=512,
                hidden_dim=256,
                dropout=0.0,
            )
        )

        x = torch.randn(
            4,
            512,
        )

        outputs = model(x)

        self.assertEqual(
            outputs[
                "integrity_logits"
            ].shape,
            (4, 2),
        )

        self.assertEqual(
            outputs[
                "threat_logits"
            ].shape,
            (4, 4),
        )

        self.assertEqual(
            outputs[
                "features"
            ].shape,
            (4, 256),
        )


class TestWeightedHierarchicalLoss(
    unittest.TestCase
):

    def test_mixed_batch(self):

        loss_fn = (
            WeightedHierarchicalLoss()
        )

        integrity_logits = torch.randn(
            5,
            2,
            requires_grad=True,
        )

        threat_logits = torch.randn(
            5,
            4,
            requires_grad=True,
        )

        integrity_targets = torch.tensor(
            [0, 1, 1, 0, 1]
        )

        threat_targets = torch.tensor(
            [-1, 0, 1, -1, 3]
        )

        result = loss_fn(
            integrity_logits,
            threat_logits,
            integrity_targets,
            threat_targets,
        )

        self.assertTrue(
            torch.isfinite(
                result["loss"]
            )
        )

        self.assertEqual(
            result["harmful_count"],
            3,
        )

    def test_true_only_batch(self):

        loss_fn = (
            WeightedHierarchicalLoss()
        )

        integrity_logits = torch.randn(
            4,
            2,
            requires_grad=True,
        )

        threat_logits = torch.randn(
            4,
            4,
            requires_grad=True,
        )

        integrity_targets = torch.tensor(
            [0, 0, 0, 0]
        )

        threat_targets = torch.tensor(
            [-1, -1, -1, -1]
        )

        result = loss_fn(
            integrity_logits,
            threat_logits,
            integrity_targets,
            threat_targets,
        )

        self.assertEqual(
            result["harmful_count"],
            0,
        )

        self.assertEqual(
            float(
                result[
                    "threat_loss"
                ].item()
            ),
            0.0,
        )

        self.assertTrue(
            torch.isfinite(
                result["loss"]
            )
        )


class TestClassificationLayer(
    unittest.TestCase
):

    def test_inference_output(self):

        torch.manual_seed(42)

        model = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=512,
                hidden_dim=256,
                dropout=0.0,
            )
        )

        layer = (
            HierarchicalClassificationLayer(
                model=model,
                device="cpu",
            )
        )

        representation = (
            MultimodalRepresentation(
                sample_id="CCT-001",
                fused_embedding=torch.randn(512),
                shared_dimension=512,
                text_available=True,
                vision_available=True,
                is_multimodal=True,
            )
        )

        result = layer.process(
            representation
        )

        self.assertEqual(
            result.sample_id,
            "CCT-001",
        )

        self.assertIn(
            result.integrity_status,
            [
                IntegrityStatus.TRUE,
                IntegrityStatus.HARMFUL,
            ],
        )

        self.assertGreaterEqual(
            result.integrity_confidence,
            0.0,
        )

        self.assertLessEqual(
            result.integrity_confidence,
            1.0,
        )

        self.assertEqual(
            result.integrity_probabilities.shape,
            (2,),
        )

        self.assertEqual(
            result.threat_probabilities.shape,
            (4,),
        )

    def test_true_output_has_no_threat_type(self):

        model = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=512,
                hidden_dim=256,
                dropout=0.0,
            )
        )

        with torch.no_grad():

            model.integrity_head.weight.zero_()
            model.integrity_head.bias.zero_()

            model.integrity_head.bias[0] = 10.0
            model.integrity_head.bias[1] = -10.0

        layer = (
            HierarchicalClassificationLayer(
                model=model,
                device="cpu",
            )
        )

        representation = (
            MultimodalRepresentation(
                sample_id="TRUE-001",
                fused_embedding=torch.randn(512),
                shared_dimension=512,
                text_available=True,
                vision_available=True,
                is_multimodal=True,
            )
        )

        result = layer.process(
            representation
        )

        self.assertEqual(
            result.integrity_status,
            IntegrityStatus.TRUE,
        )

        self.assertIsNone(
            result.threat_type
        )

        self.assertIsNone(
            result.threat_confidence
        )

    def test_harmful_output_has_threat_type(self):

        model = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=512,
                hidden_dim=256,
                dropout=0.0,
            )
        )

        with torch.no_grad():

            model.integrity_head.weight.zero_()
            model.integrity_head.bias.zero_()

            model.integrity_head.bias[0] = -10.0
            model.integrity_head.bias[1] = 10.0

            model.threat_head.weight.zero_()
            model.threat_head.bias.zero_()

            model.threat_head.bias[1] = 10.0

        layer = (
            HierarchicalClassificationLayer(
                model=model,
                device="cpu",
            )
        )

        representation = (
            MultimodalRepresentation(
                sample_id="CCT-002",
                fused_embedding=torch.randn(512),
                shared_dimension=512,
                text_available=True,
                vision_available=True,
                is_multimodal=True,
            )
        )

        result = layer.process(
            representation
        )

        self.assertEqual(
            result.integrity_status,
            IntegrityStatus.HARMFUL,
        )

        self.assertEqual(
            result.threat_type,
            CognitiveThreatType.DISINFORMATION,
        )

        self.assertIsNotNone(
            result.threat_confidence
        )

    def test_contextualized_representation_input(self):

        model = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=512,
                hidden_dim=256,
                dropout=0.0,
            )
        )

        layer = (
            HierarchicalClassificationLayer(
                model=model,
                device="cpu",
            )
        )

        representation = (
            ContextualizedRepresentation(
                sample_id="CTX-CLASS-001",
                content_embedding=torch.randn(512),
                context_embedding=torch.randn(128),
                fused_embedding=torch.randn(512),
                content_dimension=512,
                context_dimension=128,
                fused_dimension=512,
            )
        )

        result = layer.process(
            representation
        )

        self.assertEqual(
            result.sample_id,
            "CTX-CLASS-001",
        )

        self.assertEqual(
            result.metadata[
                "input_source"
            ],
            "contextualized",
        )

        self.assertIn(
            result.integrity_status,
            [
                IntegrityStatus.TRUE,
                IntegrityStatus.HARMFUL,
            ],
        )


if __name__ == "__main__":
    unittest.main()
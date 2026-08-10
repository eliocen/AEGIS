"""
Tests for AEGIS v0.9.0:
Cross-Modal Semantic Alignment.
"""

import unittest

import torch

from aegis.alignment import (
    CrossModalAlignmentLayer,
    CrossModalAlignmentModel,
    GatedMultimodalFusion,
    ProjectionHead,
    SymmetricContrastiveLoss,
)

from aegis.representation import (
    TextRepresentation,
    VisionRepresentation,
)


class TestProjectionHead(unittest.TestCase):

    def test_projection_shape(self):
        head = ProjectionHead(
            input_dim=768,
            output_dim=512,
            dropout=0.0,
        )

        x = torch.randn(
            4,
            768,
        )

        output = head(x)

        self.assertEqual(
            output.shape,
            (4, 512),
        )


class TestContrastiveLoss(unittest.TestCase):

    def test_loss_is_scalar(self):
        loss_function = (
            SymmetricContrastiveLoss()
        )

        text = torch.randn(
            4,
            512,
        )

        vision = torch.randn(
            4,
            512,
        )

        loss = loss_function(
            text,
            vision,
        )

        self.assertEqual(
            loss.ndim,
            0,
        )

        self.assertTrue(
            torch.isfinite(loss)
        )


class TestFusion(unittest.TestCase):

    def test_fusion_shape(self):
        fusion = GatedMultimodalFusion(
            dimension=512
        )

        text = torch.randn(
            4,
            512,
        )

        vision = torch.randn(
            4,
            512,
        )

        result = fusion(
            text,
            vision,
        )

        self.assertEqual(
            result.shape,
            (4, 512),
        )


class TestAlignmentModel(unittest.TestCase):

    def test_alignment_forward(self):
        model = CrossModalAlignmentModel(
            dropout=0.0
        )

        text = torch.randn(
            3,
            768,
        )

        vision = torch.randn(
            3,
            512,
        )

        result = model(
            text,
            vision,
            compute_loss=True,
        )

        self.assertEqual(
            result[
                "aligned_text"
            ].shape,
            (3, 512),
        )

        self.assertEqual(
            result[
                "aligned_vision"
            ].shape,
            (3, 512),
        )

        self.assertEqual(
            result[
                "fused_embedding"
            ].shape,
            (3, 512),
        )

        self.assertTrue(
            torch.isfinite(
                result[
                    "alignment_loss"
                ]
            )
        )


class TestAlignmentLayer(unittest.TestCase):

    def test_inference_layer(self):
        model = CrossModalAlignmentModel(
            dropout=0.0
        )

        layer = CrossModalAlignmentLayer(
            model=model,
            device="cpu",
        )

        text = TextRepresentation(
            sample_id="MM1",
            embedding=torch.randn(768),
            language="en",
            model_name="mock-text",
            dimension=768,
        )

        vision = VisionRepresentation(
            sample_id="MM1",
            embedding=torch.randn(512),
            model_name="mock-vision",
            dimension=512,
        )

        result = layer.process(
            (text, vision)
        )

        self.assertEqual(
            result.sample_id,
            "MM1",
        )

        self.assertEqual(
            result.shared_dimension,
            512,
        )

        self.assertEqual(
            result.fused_embedding.shape,
            (512,),
        )

        self.assertTrue(
            result.is_multimodal
        )


if __name__ == "__main__":
    unittest.main()

class TestParallelAlignmentIntegration(
    unittest.TestCase
):

    def test_bundle_to_alignment(self):

        from aegis.representation import (
            ParallelRepresentationBundle,
        )

        text = TextRepresentation(
            sample_id="MMX",
            embedding=torch.randn(768),
            language="en",
            model_name="mock-text",
            dimension=768,
        )

        vision = VisionRepresentation(
            sample_id="MMX",
            embedding=torch.randn(512),
            model_name="mock-vision",
            dimension=512,
        )

        bundle = ParallelRepresentationBundle(
            sample_id="MMX",
            text=text,
            vision=vision,
            text_available=True,
            vision_available=True,
        )

        model = CrossModalAlignmentModel(
            dropout=0.0
        )

        layer = CrossModalAlignmentLayer(
            model=model,
            device="cpu",
        )

        result = layer.process(
            bundle
        )

        self.assertEqual(
            result.sample_id,
            "MMX",
        )

        self.assertEqual(
            result.fused_embedding.shape,
            (512,),
        )
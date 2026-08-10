"""
Tests for AEGIS v0.11.0:
Context Intelligence Layer.
"""

import unittest

import torch

from aegis.alignment import (
    MultimodalRepresentation,
)

from aegis.context import (
    ContextAwareFusion,
    ContextEncoder,
    ContextInput,
    ContextIntelligenceLayer,
    OperationalDomain,
)


class TestContextEncoder(
    unittest.TestCase
):

    def test_context_embedding_shape(self):

        encoder = ContextEncoder(
            output_dim=128
        )

        context = ContextInput(
            sample_id="CTX1",
            domain=(
                OperationalDomain.ELECTION
            ),
            country="Uganda",
            platform="X",
            language="en",
            event="General Election",
        )

        result = encoder(
            context
        )

        self.assertEqual(
            result.dimension,
            128,
        )

        self.assertEqual(
            result.embedding.shape,
            (128,),
        )

        self.assertEqual(
            result.metadata["domain"],
            "election",
        )


class TestContextAwareFusion(
    unittest.TestCase
):

    def test_fusion_shape(self):

        fusion = ContextAwareFusion(
            content_dim=512,
            context_dim=128,
            output_dim=512,
        )

        content = torch.randn(
            4,
            512,
        )

        context = torch.randn(
            4,
            128,
        )

        result = fusion(
            content,
            context,
        )

        self.assertEqual(
            result.shape,
            (4, 512),
        )


class TestContextIntelligenceLayer(
    unittest.TestCase
):

    def test_contextualized_output(self):

        encoder = ContextEncoder(
            output_dim=128
        )

        fusion = ContextAwareFusion(
            content_dim=512,
            context_dim=128,
            output_dim=512,
        )

        layer = ContextIntelligenceLayer(
            encoder=encoder,
            fusion=fusion,
            device="cpu",
        )

        representation = (
            MultimodalRepresentation(
                sample_id="CTX-001",
                fused_embedding=(
                    torch.randn(512)
                ),
                shared_dimension=512,
                text_available=True,
                vision_available=True,
                is_multimodal=True,
            )
        )

        context = ContextInput(
            sample_id="CTX-001",
            domain=(
                OperationalDomain.CONFLICT
            ),
            country="Uganda",
            platform="Social Media",
            language="en",
            event="Regional Security Crisis",
        )

        result = layer.process(
            (
                representation,
                context,
            )
        )

        self.assertEqual(
            result.sample_id,
            "CTX-001",
        )

        self.assertEqual(
            result.context_dimension,
            128,
        )

        self.assertEqual(
            result.fused_embedding.shape,
            (512,),
        )

        self.assertEqual(
            result.metadata["domain"],
            "conflict",
        )

        self.assertTrue(
            result.metadata[
                "context_aware"
            ]
        )

    def test_sample_id_mismatch_fails(self):

        encoder = ContextEncoder()

        fusion = ContextAwareFusion()

        layer = ContextIntelligenceLayer(
            encoder=encoder,
            fusion=fusion,
        )

        representation = (
            MultimodalRepresentation(
                sample_id="A",
                fused_embedding=(
                    torch.randn(512)
                ),
                shared_dimension=512,
            )
        )

        context = ContextInput(
            sample_id="B"
        )

        with self.assertRaises(
            ValueError
        ):
            layer.process(
                (
                    representation,
                    context,
                )
            )

    def test_different_domains_produce_different_context(self):

        torch.manual_seed(42)

        encoder = ContextEncoder()

        fusion = ContextAwareFusion()

        layer = ContextIntelligenceLayer(
            encoder=encoder,
            fusion=fusion,
            device="cpu",
        )

        representation = (
            MultimodalRepresentation(
                sample_id="CTX-DIFF",
                fused_embedding=(
                    torch.randn(512)
                ),
                shared_dimension=512,
            )
        )

        election_context = ContextInput(
            sample_id="CTX-DIFF",
            domain=(
                OperationalDomain.ELECTION
            ),
            country="Uganda",
            platform="X",
            language="en",
        )

        conflict_context = ContextInput(
            sample_id="CTX-DIFF",
            domain=(
                OperationalDomain.CONFLICT
            ),
            country="Uganda",
            platform="X",
            language="en",
        )

        election_result = layer.process(
            (
                representation,
                election_context,
            )
        )

        conflict_result = layer.process(
            (
                representation,
                conflict_context,
            )
        )

        self.assertFalse(
            torch.allclose(
                election_result.fused_embedding,
                conflict_result.fused_embedding,
            )
        )


if __name__ == "__main__":
    unittest.main()
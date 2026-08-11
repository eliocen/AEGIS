"""
Tests for AEGIS configuration and pipeline infrastructure.
"""

import unittest

from aegis import AEGIS
from aegis.config import AEGISConfig
from aegis.pipeline import AEGISLayer, AEGISPipeline


class AddOneLayer(AEGISLayer):
    layer_name = "add_one"

    def process(self, data):
        return data + 1


class MultiplyByTwoLayer(AEGISLayer):
    layer_name = "multiply_by_two"

    def process(self, data):
        return data * 2


class TestAEGISConfig(unittest.TestCase):

    def test_default_configuration(self):
        config = AEGISConfig()

        self.assertEqual(
            config.get("framework.version"),
            "0.20.0",
        )

        self.assertEqual(
            config.get("framework.seed"),
            42,
        )

    def test_override_configuration(self):
        config = AEGISConfig(
            {
                "framework": {
                    "device": "cpu",
                }
            }
        )

        self.assertEqual(
            config.get("framework.device"),
            "cpu",
        )


class TestAEGISPipeline(unittest.TestCase):

    def test_layer_registration(self):
        pipeline = AEGISPipeline()

        pipeline.add_layer(AddOneLayer())

        self.assertEqual(
            pipeline.describe(),
            ["add_one"],
        )

    def test_pipeline_execution(self):
        pipeline = AEGISPipeline(
            [
                AddOneLayer(),
                MultiplyByTwoLayer(),
            ]
        )

        result = pipeline.run(3)

        self.assertEqual(result, 8)

    def test_aegis_orchestration(self):
        framework = AEGIS()

        framework.add_layer(AddOneLayer())

        result = framework.analyze(10)

        self.assertEqual(result, 11)


if __name__ == "__main__":
    unittest.main()
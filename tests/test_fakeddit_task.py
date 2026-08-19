"""
Tests for the AEGIS Fakeddit empirical task contract.
"""

import unittest

from aegis.data.tasks import (
    AEGISIntegrityTarget,
    FAKEDDIT_BINARY_TASK_NAME,
    FakedditBinaryLabel,
    build_fakeddit_binary_target,
    build_fakeddit_binary_target_from_labels,
    map_fakeddit_binary_to_integrity,
    normalize_fakeddit_binary_label,
)


class TestFakedditNativeBinaryLabels(
    unittest.TestCase
):

    def test_fake_label(self):
        label = (
            normalize_fakeddit_binary_label(
                0
            )
        )

        self.assertEqual(
            label,
            FakedditBinaryLabel.FAKE,
        )

    def test_true_label(self):
        label = (
            normalize_fakeddit_binary_label(
                1
            )
        )

        self.assertEqual(
            label,
            FakedditBinaryLabel.TRUE,
        )

    def test_string_label(self):
        label = (
            normalize_fakeddit_binary_label(
                "1"
            )
        )

        self.assertEqual(
            label,
            FakedditBinaryLabel.TRUE,
        )

    def test_unknown_label_fails(self):
        with self.assertRaises(
            ValueError
        ):
            normalize_fakeddit_binary_label(
                2
            )

    def test_non_numeric_label_fails(self):
        with self.assertRaises(
            ValueError
        ):
            normalize_fakeddit_binary_label(
                "fake"
            )


class TestFakedditIntegrityMapping(
    unittest.TestCase
):

    def test_true_maps_to_true_integrity(self):
        target = (
            map_fakeddit_binary_to_integrity(
                1
            )
        )

        self.assertEqual(
            target,
            AEGISIntegrityTarget.TRUE,
        )

    def test_fake_maps_to_harmful_integrity(self):
        target = (
            map_fakeddit_binary_to_integrity(
                0
            )
        )

        self.assertEqual(
            target,
            AEGISIntegrityTarget.HARMFUL,
        )


class TestFakedditBinaryTarget(
    unittest.TestCase
):

    def test_true_target(self):
        target = (
            build_fakeddit_binary_target(
                sample_id="sample-true",
                native_label=1,
            )
        )

        self.assertEqual(
            target.native_label,
            1,
        )

        self.assertEqual(
            target.native_name,
            "true",
        )

        self.assertEqual(
            target.integrity_target,
            0,
        )

        self.assertEqual(
            target.integrity_name,
            "true",
        )

        self.assertIsNone(
            target.threat_target
        )

    def test_fake_target(self):
        target = (
            build_fakeddit_binary_target(
                sample_id="sample-fake",
                native_label=0,
            )
        )

        self.assertEqual(
            target.native_label,
            0,
        )

        self.assertEqual(
            target.native_name,
            "fake",
        )

        self.assertEqual(
            target.integrity_target,
            1,
        )

        self.assertEqual(
            target.integrity_name,
            "harmful",
        )

        self.assertIsNone(
            target.threat_target
        )

    def test_task_name(self):
        target = (
            build_fakeddit_binary_target(
                sample_id="sample-1",
                native_label=1,
            )
        )

        self.assertEqual(
            target.task_name,
            FAKEDDIT_BINARY_TASK_NAME,
        )

    def test_empty_sample_id_fails(self):
        with self.assertRaises(
            ValueError
        ):
            build_fakeddit_binary_target(
                sample_id="",
                native_label=1,
            )

    def test_mapping_from_native_labels(self):
        target = (
            build_fakeddit_binary_target_from_labels(
                sample_id="sample-2",
                native_labels={
                    "2_way_label": 0,
                    "3_way_label": 2,
                    "6_way_label": 4,
                },
            )
        )

        self.assertEqual(
            target.native_label,
            0,
        )

        self.assertEqual(
            target.integrity_target,
            1,
        )

        self.assertIsNone(
            target.threat_target
        )

    def test_missing_native_binary_label_fails(
        self
    ):
        with self.assertRaises(
            KeyError
        ):
            build_fakeddit_binary_target_from_labels(
                sample_id="sample-3",
                native_labels={
                    "6_way_label": 0,
                },
            )

    def test_as_dict(self):
        target = (
            build_fakeddit_binary_target(
                sample_id="sample-4",
                native_label=1,
            )
        )

        payload = target.as_dict()

        self.assertEqual(
            payload["sample_id"],
            "sample-4",
        )

        self.assertEqual(
            payload["source_dataset"],
            "Fakeddit",
        )

        self.assertEqual(
            payload["native_label"],
            1,
        )

        self.assertEqual(
            payload["integrity_target"],
            0,
        )

        self.assertIsNone(
            payload["threat_target"]
        )


if __name__ == "__main__":
    unittest.main()
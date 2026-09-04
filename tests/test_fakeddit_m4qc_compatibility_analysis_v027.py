from __future__ import annotations

import copy

import pytest

from scripts.analyze_fakeddit_m4qc_compatibility_v027 import (
    EXPECTED_ROWS_PER_SEED,
    compute_formal_decisions,
    roc_auc_binary,
    validate_mapping_payload,
    validate_seed_rows,
)


def _mapping_payload():
    mapping = []
    for index in range(EXPECTED_ROWS_PER_SEED):
        # Two 500-row classes; cyclic shift within class.
        class_id = 0 if index < 500 else 1
        start = 0 if class_id == 0 else 500
        local = index - start
        donor = start + ((local + 1) % 500)
        mapping.append(
            {
                "validation_row_index": index,
                "text_sample_id": f"s{index}",
                "stage1_label": class_id,
                "donor_vision_index": donor,
                "donor_vision_sample_id": f"s{donor}",
                "donor_stage1_label": class_id,
                "self_pair": False,
            }
        )

    return {
        "protocol_version": "0.27.0-step11a",
        "validation_sample_count": EXPECTED_ROWS_PER_SEED,
        "official_test_accessed": False,
        "invariants": {
            "no_self_pairs": True,
            "class_preserving": True,
            "one_mismatch_partner_per_text": True,
            "each_vision_used_exactly_once_within_class": True,
            "mapping_shared_across_model_seeds": True,
            "mapping_independent_of_training_rng": True,
        },
        "mapping": mapping,
    }


def _diagnostic_rows(seed=42):
    mapping = _mapping_payload()["mapping"]
    rows = []
    for item in mapping:
        index = item["validation_row_index"]
        donor = item["donor_vision_index"]
        label = item["stage1_label"]
        rows.append(
            {
                "protocol_version": "0.27.0-step11a",
                "seed": seed,
                "checkpoint_epoch": 5,
                "validation_row_index": index,
                "stage1_label": label,
                "text_source_index": index,
                "text_source_sample_id": f"s{index}",
                "matched": {
                    "vision_source_index": index,
                    "vision_source_sample_id": f"s{index}",
                    "text_quality": 0.90,
                    "vision_quality": 0.85,
                    "compatibility": 0.90,
                    "compatibility_target": 1,
                },
                "mismatched": {
                    "vision_source_index": donor,
                    "vision_source_sample_id": f"s{donor}",
                    "vision_source_stage1_label": label,
                    "text_quality": 0.89,
                    "vision_quality": 0.84,
                    "compatibility": 0.10,
                    "compatibility_target": 0,
                },
                "official_test_accessed": False,
            }
        )
    return rows


def test_roc_auc_perfect_separation():
    assert roc_auc_binary([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0)


def test_roc_auc_reverse_separation():
    assert roc_auc_binary([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)


def test_roc_auc_ties_are_half_credit():
    assert roc_auc_binary([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_mapping_validation_accepts_frozen_invariants():
    mapping = validate_mapping_payload(_mapping_payload())
    assert len(mapping) == EXPECTED_ROWS_PER_SEED
    assert mapping[0]["donor_vision_index"] == 1
    assert mapping[499]["donor_vision_index"] == 0
    assert mapping[999]["donor_vision_index"] == 500


def test_mapping_validation_rejects_self_pair():
    payload = _mapping_payload()
    payload["mapping"][0]["donor_vision_index"] = 0
    payload["mapping"][0]["self_pair"] = True
    with pytest.raises(RuntimeError, match="self-pair"):
        validate_mapping_payload(payload)


def test_seed_rows_validate_identity_contract():
    mapping = validate_mapping_payload(_mapping_payload())
    rows = validate_seed_rows(
        _diagnostic_rows(seed=42),
        seed=42,
        mapping_by_index=mapping,
    )
    assert len(rows) == EXPECTED_ROWS_PER_SEED
    assert rows[0]["mismatched"]["vision_source_index"] == 1


def test_seed_rows_reject_duplicate_index():
    mapping = validate_mapping_payload(_mapping_payload())
    rows = _diagnostic_rows(seed=42)
    rows[-1]["validation_row_index"] = 0
    with pytest.raises(RuntimeError, match="duplicate diagnostic row index"):
        validate_seed_rows(rows, seed=42, mapping_by_index=mapping)


def test_seed_rows_reject_out_of_range_score():
    mapping = validate_mapping_payload(_mapping_payload())
    rows = _diagnostic_rows(seed=42)
    rows[0]["matched"]["compatibility"] = 1.2
    with pytest.raises(RuntimeError, match=r"outside \[0,1\]"):
        validate_seed_rows(rows, seed=42, mapping_by_index=mapping)


def test_formal_decisions_apply_frozen_strict_h5_rules():
    seed_results = []
    for seed, auc in ((42, 0.82), (43, 0.83), (44, 0.84)):
        seed_results.append(
            {
                "seed": seed,
                "h4_c": {"roc_auc": auc},
                "h5_c": {
                    "mean_abs_text_quality_change": 0.02,
                    "mean_abs_vision_quality_change": 0.03,
                    "compatibility_drop": 0.40,
                },
            }
        )

    classification_gate = {
        "reference_model": "M1b",
        "reference_macro_f1": 0.8736,
        "m4qc_mean_macro_f1": 0.864839357609,
        "delta_vs_m1b": -0.008760642391,
        "frozen_margin": -0.01,
        "criterion": "delta_vs_m1b >= -0.01",
        "interpretation": "descriptive",
    }

    decisions = compute_formal_decisions(seed_results, classification_gate)
    assert decisions["H4-C"]["supported"] is True
    assert decisions["H5-C"]["supported"] is True
    assert decisions["classification_preservation"]["passed"] is True


def test_h5_text_threshold_is_strict():
    seed_results = []
    for seed in (42, 43, 44):
        seed_results.append(
            {
                "seed": seed,
                "h4_c": {"roc_auc": 0.90},
                "h5_c": {
                    "mean_abs_text_quality_change": 0.10,
                    "mean_abs_vision_quality_change": 0.01,
                    "compatibility_drop": 0.2,
                },
            }
        )

    gate = {
        "reference_model": "M1b",
        "reference_macro_f1": 0.8736,
        "m4qc_mean_macro_f1": 0.864,
        "delta_vs_m1b": -0.0096,
        "frozen_margin": -0.01,
        "criterion": "delta_vs_m1b >= -0.01",
        "interpretation": "descriptive",
    }

    decisions = compute_formal_decisions(seed_results, gate)
    assert decisions["H5-C"]["supported"] is False
    assert (
        decisions["H5-C"]["component_pass"]["text_quality_stability"]
        is False
    )


def test_h5_compatibility_drop_must_be_strictly_positive():
    seed_results = []
    for seed in (42, 43, 44):
        seed_results.append(
            {
                "seed": seed,
                "h4_c": {"roc_auc": 0.90},
                "h5_c": {
                    "mean_abs_text_quality_change": 0.01,
                    "mean_abs_vision_quality_change": 0.01,
                    "compatibility_drop": 0.0,
                },
            }
        )

    gate = {
        "reference_model": "M1b",
        "reference_macro_f1": 0.8736,
        "m4qc_mean_macro_f1": 0.864,
        "delta_vs_m1b": -0.0096,
        "frozen_margin": -0.01,
        "criterion": "delta_vs_m1b >= -0.01",
        "interpretation": "descriptive",
    }

    decisions = compute_formal_decisions(seed_results, gate)
    assert decisions["H5-C"]["supported"] is False
    assert (
        decisions["H5-C"]["component_pass"]["positive_compatibility_drop"]
        is False
    )

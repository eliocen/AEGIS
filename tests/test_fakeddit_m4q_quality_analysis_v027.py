from __future__ import annotations

import copy
import math

import pytest

from scripts.analyze_fakeddit_m4q_quality_v027 import (
    CONTINUOUS_FAMILIES,
    EXPECTED_CONDITIONS_PER_SEED,
    EXPECTED_PROTOCOL_VERSION,
    EXPECTED_SEEDS,
    MODALITIES,
    analyze,
    analyze_h1_q,
    analyze_h2_q,
    analyze_h3_q,
    analyze_h6_cls,
    condition_key,
    expected_condition_keys,
    extract_step9_rows,
    pearson_correlation,
    rankdata_average,
    spearman_correlation,
    validate_protocol,
    validate_step9_rows,
)


def make_protocol():
    return {
        "protocol": {
            "version": EXPECTED_PROTOCOL_VERSION,
            "status": "FROZEN_BEFORE_FORMAL_STEP10_HYPOTHESIS_COMPUTATION",
        },
        "inputs": {
            "model_seeds": [42, 43, 44],
            "conditions_per_seed": 19,
            "total_condition_evaluations": 57,
            "official_test_access_permitted": False,
            "training_permitted": False,
        },
        "h1_q": {
            "continuous_corruption_families": ["gaussian_noise", "attenuation"],
            "corrupted_modalities": ["text", "vision"],
            "severities": [0.0, 0.25, 0.5, 0.75, 1.0],
            "curve_count": 12,
            "minimum_negative_curve_count": 9,
            "minimum_negative_curve_proportion": 0.75,
        },
        "h2_q": {
            "denominator": 54,
            "minimum_positive_count": 41,
            "minimum_positive_proportion": 0.75,
            "zero_dropout_included": True,
        },
        "h3_q": {
            "condition_count": 6,
            "maximum_mean_zeroed_quality": 0.2,
            "minimum_mean_intact_quality": 0.8,
            "require_every_individual_condition_to_pass": False,
        },
        "h6_cls": {
            "metric": "Macro-F1",
            "m4q_seed_values": {
                "42": 0.872978533,
                "43": 0.877741301,
                "44": 0.866970068,
            },
            "m4q_mean": 0.8725633007431841,
            "m1b_reference_mean": 0.8736,
            "delta": -0.0010366992568159317,
            "minimum_delta": -0.01,
            "formal_noninferiority_test": False,
        },
    }


def make_rows(
    *,
    gaussian_drop=0.02,
    attenuation_values=(0.85, 0.65, 0.35, 0.05),
    zero_value=0.02,
    intact_shift=0.0,
):
    rows = []

    for seed in EXPECTED_SEEDS:
        clean_text = 0.97
        clean_vision = 0.95

        def add(family, modality, severity, text_q, vision_q):
            rows.append(
                {
                    "aegis_version": "0.27.4-dev",
                    "model_variant": "M4q",
                    "fusion_architecture": "quality_supervised",
                    "model_seed": seed,
                    "corruption_type": family,
                    "corrupted_modality": modality,
                    "severity": severity,
                    "sample_count": 1000,
                    "text_quality_mean": text_q,
                    "vision_quality_mean": vision_q,
                    "test_split_accessed": False,
                    "parameters_unchanged": True,
                }
            )

        add("clean", "none", None, clean_text, clean_vision)

        for family in CONTINUOUS_FAMILIES:
            for modality in MODALITIES:
                if family == "gaussian_noise":
                    drops = [
                        gaussian_drop * 0.25,
                        gaussian_drop * 0.50,
                        gaussian_drop * 0.75,
                        gaussian_drop,
                    ]
                    values = None
                else:
                    drops = None
                    values = attenuation_values

                for idx, severity in enumerate((0.25, 0.5, 0.75, 1.0)):
                    if modality == "text":
                        corrupted = (
                            clean_text - drops[idx]
                            if drops is not None
                            else values[idx]
                        )
                        add(
                            family,
                            modality,
                            severity,
                            corrupted,
                            clean_vision + intact_shift,
                        )
                    else:
                        corrupted = (
                            clean_vision - drops[idx]
                            if drops is not None
                            else values[idx]
                        )
                        add(
                            family,
                            modality,
                            severity,
                            clean_text + intact_shift,
                            corrupted,
                        )

        add("zero_dropout", "text", 1.0, zero_value, clean_vision + intact_shift)
        add("zero_dropout", "vision", 1.0, clean_text + intact_shift, zero_value)

    assert len(rows) == 57
    return rows


def make_step9_payload(rows=None):
    if rows is None:
        rows = make_rows()
    return {
        "number_of_condition_evaluations": 57,
        "model_seeds": [42, 43, 44],
        "conditions_per_checkpoint": 19,
        "official_test_split_accessed": False,
        "rows": rows,
    }


def indexed(rows):
    return validate_step9_rows(rows)


# ---------------------------------------------------------------------------
# Generic statistics
# ---------------------------------------------------------------------------


def test_rankdata_average_no_ties():
    assert rankdata_average([10, 20, 30]) == [1.0, 2.0, 3.0]


def test_rankdata_average_with_ties():
    assert rankdata_average([10, 10, 30]) == [1.5, 1.5, 3.0]


def test_pearson_perfect_positive():
    assert pearson_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert spearman_correlation(
        [0, 0.25, 0.5, 0.75, 1.0],
        [1.0, 0.8, 0.6, 0.4, 0.2],
    ) == pytest.approx(-1.0)


def test_spearman_constant_prediction_returns_zero():
    assert spearman_correlation([0, 1, 2], [0.5, 0.5, 0.5]) == 0.0


# ---------------------------------------------------------------------------
# Frozen protocol validation
# ---------------------------------------------------------------------------


def test_validate_protocol_accepts_frozen_contract():
    validate_protocol(make_protocol())


@pytest.mark.parametrize(
    "section,key,bad",
    [
        ("inputs", "conditions_per_seed", 18),
        ("inputs", "total_condition_evaluations", 56),
        ("h1_q", "curve_count", 11),
        ("h1_q", "minimum_negative_curve_count", 8),
        ("h2_q", "denominator", 53),
        ("h2_q", "minimum_positive_count", 40),
        ("h3_q", "condition_count", 5),
        ("h3_q", "maximum_mean_zeroed_quality", 0.3),
        ("h3_q", "minimum_mean_intact_quality", 0.7),
        ("h6_cls", "minimum_delta", -0.02),
    ],
)
def test_validate_protocol_rejects_changed_frozen_values(section, key, bad):
    protocol = make_protocol()
    protocol[section][key] = bad
    with pytest.raises(RuntimeError):
        validate_protocol(protocol)


def test_validate_protocol_rejects_test_access_permission():
    protocol = make_protocol()
    protocol["inputs"]["official_test_access_permitted"] = True
    with pytest.raises(RuntimeError):
        validate_protocol(protocol)


def test_validate_protocol_rejects_training_permission():
    protocol = make_protocol()
    protocol["inputs"]["training_permitted"] = True
    with pytest.raises(RuntimeError):
        validate_protocol(protocol)


# ---------------------------------------------------------------------------
# Step 9 integrity validation
# ---------------------------------------------------------------------------


def test_expected_condition_key_count_is_19():
    assert len(expected_condition_keys()) == 19


def test_validate_step9_rows_accepts_exact_57():
    by_seed = validate_step9_rows(make_rows())
    assert set(by_seed) == set(EXPECTED_SEEDS)
    assert all(len(by_seed[seed]) == EXPECTED_CONDITIONS_PER_SEED for seed in EXPECTED_SEEDS)


def test_validate_step9_rows_rejects_missing_condition():
    rows = make_rows()[:-1]
    with pytest.raises(RuntimeError):
        validate_step9_rows(rows)


def test_validate_step9_rows_rejects_duplicate_condition():
    rows = make_rows()
    rows[-1] = copy.deepcopy(rows[-2])
    with pytest.raises(RuntimeError):
        validate_step9_rows(rows)


def test_validate_step9_rows_rejects_test_access():
    rows = make_rows()
    rows[3]["test_split_accessed"] = True
    with pytest.raises(RuntimeError):
        validate_step9_rows(rows)


def test_validate_step9_rows_rejects_parameter_change():
    rows = make_rows()
    rows[3]["parameters_unchanged"] = False
    with pytest.raises(RuntimeError):
        validate_step9_rows(rows)


def test_extract_step9_rows_rejects_wrong_total():
    payload = make_step9_payload()
    payload["number_of_condition_evaluations"] = 56
    with pytest.raises(RuntimeError):
        extract_step9_rows(payload)


def test_extract_step9_rows_rejects_unsealed_test():
    payload = make_step9_payload()
    payload["official_test_split_accessed"] = True
    with pytest.raises(RuntimeError):
        extract_step9_rows(payload)


# ---------------------------------------------------------------------------
# H1-Q
# ---------------------------------------------------------------------------


def test_h1_has_exactly_12_curves():
    result = analyze_h1_q(indexed(make_rows()), make_protocol())
    assert result["curve_count"] == 12


def test_h1_supports_monotonic_degradation():
    result = analyze_h1_q(indexed(make_rows()), make_protocol())
    assert result["negative_curve_count"] == 12
    assert result["negative_curve_proportion"] == pytest.approx(1.0)
    assert result["mean_spearman_rho"] < 0.0
    assert result["supported"] is True


def test_h1_zero_rho_does_not_count_negative():
    rows = make_rows()
    for row in rows:
        if row["corruption_type"] == "gaussian_noise":
            if row["corrupted_modality"] == "text":
                row["text_quality_mean"] = 0.97
            else:
                row["vision_quality_mean"] = 0.95
    result = analyze_h1_q(indexed(rows), make_protocol())
    gaussian = [
        row for row in result["curves"]
        if row["corruption_family"] == "gaussian_noise"
    ]
    assert all(row["spearman_rho"] == 0.0 for row in gaussian)
    assert all(row["expected_negative"] is False for row in gaussian)


# ---------------------------------------------------------------------------
# H2-Q
# ---------------------------------------------------------------------------


def test_h2_denominator_is_exactly_54():
    result = analyze_h2_q(indexed(make_rows()), make_protocol())
    assert result["denominator"] == 54


def test_h2_includes_six_zero_dropout_conditions():
    result = analyze_h2_q(indexed(make_rows()), make_protocol())
    zero = [
        row for row in result["conditions"]
        if row["corruption_family"] == "zero_dropout"
    ]
    assert len(zero) == 6


def test_h2_positive_selectivity_supports_when_all_positive():
    result = analyze_h2_q(indexed(make_rows()), make_protocol())
    assert result["positive_selectivity_count"] == 54
    assert result["supported"] is True


def test_h2_zero_selectivity_is_not_positive():
    rows = make_rows(
        gaussian_drop=0.0,
        attenuation_values=(0.97, 0.97, 0.97, 0.97),
        zero_value=0.97,
        intact_shift=0.0,
    )
    # Vision clean is 0.95, so normalize every vision corruption back to its
    # own clean value to make all selectivities exactly zero.
    for row in rows:
        if row["corruption_type"] != "clean":
            if row["corrupted_modality"] == "text":
                row["text_quality_mean"] = 0.97
                row["vision_quality_mean"] = 0.95
            else:
                row["text_quality_mean"] = 0.97
                row["vision_quality_mean"] = 0.95
    result = analyze_h2_q(indexed(rows), make_protocol())
    assert result["positive_selectivity_count"] == 0
    assert result["supported"] is False


def test_h2_threshold_is_41_of_54():
    protocol = make_protocol()
    assert protocol["h2_q"]["minimum_positive_count"] == 41
    assert 40 / 54 < 0.75 <= 41 / 54


# ---------------------------------------------------------------------------
# H3-Q
# ---------------------------------------------------------------------------


def test_h3_uses_exactly_six_zero_dropout_conditions():
    result = analyze_h3_q(indexed(make_rows()), make_protocol())
    assert result["condition_count"] == 6


def test_h3_supports_low_zeroed_high_intact():
    result = analyze_h3_q(indexed(make_rows(zero_value=0.02)), make_protocol())
    assert result["aggregate_mean_zeroed_quality"] <= 0.2
    assert result["aggregate_mean_intact_quality"] >= 0.8
    assert result["supported"] is True


def test_h3_fails_high_zeroed_quality():
    result = analyze_h3_q(indexed(make_rows(zero_value=0.3)), make_protocol())
    assert result["aggregate_mean_zeroed_quality"] > 0.2
    assert result["supported"] is False


# ---------------------------------------------------------------------------
# H6-CLS
# ---------------------------------------------------------------------------


def test_h6_uses_frozen_step8_values_and_supports_margin():
    result = analyze_h6_cls(make_protocol())
    assert result["delta"] == pytest.approx(-0.0010366992568159317)
    assert result["minimum_delta"] == pytest.approx(-0.01)
    assert result["supported"] is True
    assert result["formal_noninferiority_test"] is False


def test_h6_rejects_inconsistent_recorded_delta():
    protocol = make_protocol()
    protocol["h6_cls"]["delta"] = -0.02
    with pytest.raises(RuntimeError):
        analyze_h6_cls(protocol)


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------


def test_end_to_end_analysis_preserves_sealed_test_and_no_training():
    result = analyze(
        protocol=make_protocol(),
        step9_payload=make_step9_payload(),
    )
    assert result["step9_condition_evaluations_validated"] == 57
    assert result["model_seeds_validated"] == [42, 43, 44]
    assert result["official_test_samples_accessed"] == 0
    assert result["training_performed"] is False
    assert result["thresholds_modified"] is False
    assert set(result) >= {"h1_q", "h2_q", "h3_q", "h6_cls"}

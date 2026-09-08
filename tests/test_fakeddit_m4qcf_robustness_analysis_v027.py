from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "analyze_fakeddit_m4qcf_robustness_v027.py"
)

spec = importlib.util.spec_from_file_location(
    "analyze_fakeddit_m4qcf_robustness_v027",
    SCRIPT,
)
assert spec is not None
assert spec.loader is not None

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_h8_observations(values):
    return [
        {
            "seed": 42 + (index // 18),
            "condition": f"condition_{index}",
            "corrupted_modality": (
                "text" if index % 2 == 0 else "vision"
            ),
            "corruption_type": "gaussian_noise",
            "severity": 0.25,
            "clean_weight": 0.5,
            "corrupted_weight": 0.5 - value,
            "down_weighting": value,
            "strict_positive": value > 0.0,
        }
        for index, value in enumerate(values)
    ]


def make_h9_observations(values):
    return [
        {
            "seed": 42 + (index // 18),
            "condition": f"condition_{index}",
            "corrupted_modality": (
                "text" if index % 2 == 0 else "vision"
            ),
            "corruption_type": "attenuation",
            "severity": 0.5,
            "m1b_macro_f1": 0.7,
            "m4qc_macro_f1": 0.72,
            "m4qcf_macro_f1": 0.7 + value,
            "m4qcf_minus_m1b": value,
            "strict_positive": value > 0.0,
            "m4qcf_minus_m4qc_descriptive": value - 0.02,
        }
        for index, value in enumerate(values)
    ]


def make_h10_seed_results(
    d_i_values=(0.2, 0.3, 0.4),
    g_values=(0.01, 0.02, 0.03),
):
    rows = []

    for seed, d_i, g in zip(
        (42, 43, 44),
        d_i_values,
        g_values,
    ):
        rows.append(
            {
                "seed": seed,
                "interaction_multiplier_matched_mean": 0.8,
                "interaction_multiplier_mismatched_mean": (
                    0.8 - d_i
                ),
                "interaction_suppression_D_I": d_i,
                "interaction_suppression_strict_positive": (
                    d_i > 0.0
                ),
                "m1b_matched_macro_f1": 0.85,
                "m1b_mismatched_macro_f1": 0.75,
                "m1b_mismatch_drop": 0.10,
                "m4qcf_matched_macro_f1": 0.85,
                "m4qcf_mismatched_macro_f1": 0.75 + g,
                "m4qcf_mismatch_drop": 0.10 - g,
                "G": g,
                "m4qc_matched_macro_f1_descriptive": 0.84,
                "m4qc_mismatched_macro_f1_descriptive": 0.76,
                "m4qc_mismatch_drop_descriptive": 0.08,
            }
        )

    return rows


def test_analyzer_version_and_frozen_commit():
    assert mod.ANALYZER_VERSION == "0.27.0-step14b"
    assert mod.INPUT_PROTOCOL_VERSION == "0.27.0-step14a"
    assert mod.OPERATIONALIZATION_COMMIT == "8ac69d5"


def test_h8_supported_with_all_positive_observations():
    values = [0.05] * 54

    result = mod.decide_h8(
        make_h8_observations(values)
    )

    assert result["decision"] == "SUPPORTED"
    assert result["strict_positive_count"] == 54
    assert result["strict_positive_proportion"] == 1.0
    assert math.isclose(
        result["mean_down_weighting"],
        0.05,
        abs_tol=1e-12,
    )


def test_h8_exact_minimum_positive_count_44_can_support():
    values = [0.10] * 44 + [-0.01] * 10

    result = mod.decide_h8(
        make_h8_observations(values)
    )

    assert result["strict_positive_count"] == 44
    assert result["strict_positive_proportion"] >= 0.80
    assert result["mean_down_weighting"] > 0.0
    assert result["decision"] == "SUPPORTED"


def test_h8_43_positive_observations_fail_80_percent_gate():
    values = [0.10] * 43 + [-0.01] * 11

    result = mod.decide_h8(
        make_h8_observations(values)
    )

    assert result["strict_positive_count"] == 43
    assert result["strict_positive_proportion"] < 0.80
    assert result["decision"] == "NOT_SUPPORTED"


def test_h8_zero_is_not_strictly_positive():
    values = [0.10] * 43 + [0.0] * 11

    result = mod.decide_h8(
        make_h8_observations(values)
    )

    assert result["strict_positive_count"] == 43
    assert result["decision"] == "NOT_SUPPORTED"


def test_h8_negative_mean_fails_even_if_positive_proportion_passes():
    values = [0.01] * 44 + [-0.10] * 10

    result = mod.decide_h8(
        make_h8_observations(values)
    )

    assert result["strict_positive_count"] == 44
    assert result["strict_positive_proportion"] >= 0.80
    assert result["mean_down_weighting"] < 0.0
    assert result["decision"] == "NOT_SUPPORTED"


def test_h8_rejects_wrong_observation_count():
    with pytest.raises(mod.FormalAnalysisError):
        mod.decide_h8(
            make_h8_observations([0.1] * 53)
        )


def test_h9_supported_when_mean_and_positive_rate_pass():
    values = [0.02] * 54

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert result["decision"] == "SUPPORTED"
    assert result["strict_positive_count"] == 54
    assert result["strict_positive_proportion"] == 1.0
    assert math.isclose(
        result["mean_macro_f1_improvement"],
        0.02,
        abs_tol=1e-12,
    )


def test_h9_mean_exactly_threshold_is_allowed():
    values = [0.01] * 54

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert math.isclose(
        result["mean_macro_f1_improvement"],
        0.01,
        abs_tol=1e-12,
    )
    assert result["decision"] == "SUPPORTED"


def test_h9_38_positive_observations_satisfy_70_percent_gate():
    values = [0.03] * 38 + [0.0] * 16

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert result["strict_positive_count"] == 38
    assert result["strict_positive_proportion"] >= 0.70
    assert result["mean_macro_f1_improvement"] >= 0.01
    assert result["decision"] == "SUPPORTED"


def test_h9_37_positive_observations_fail_positive_rate_gate():
    values = [0.03] * 37 + [0.0] * 17

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert result["strict_positive_count"] == 37
    assert result["strict_positive_proportion"] < 0.70
    assert result["decision"] == "NOT_SUPPORTED"


def test_h9_positive_rate_can_pass_while_mean_fails():
    values = [0.005] * 54

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert result["strict_positive_proportion"] == 1.0
    assert result["mean_macro_f1_improvement"] < 0.01
    assert result["decision"] == "NOT_SUPPORTED"


def test_h9_zero_is_not_positive():
    values = [0.02] * 37 + [0.0] * 17

    result = mod.decide_h9(
        make_h9_observations(values)
    )

    assert result["strict_positive_count"] == 37
    assert result["decision"] == "NOT_SUPPORTED"


def test_h9_rejects_wrong_observation_count():
    with pytest.raises(mod.FormalAnalysisError):
        mod.decide_h9(
            make_h9_observations([0.02] * 55)
        )


def test_h10_supported_when_all_interaction_drops_positive_and_mean_g_positive():
    result = mod.decide_h10(
        make_h10_seed_results()
    )

    assert result["decision"] == "SUPPORTED"
    assert result[
        "interaction_suppression_positive_seed_count"
    ] == 3
    assert result[
        "all_three_interaction_suppression_positive"
    ] is True
    assert result["mean_G"] > 0.0


def test_h10_fails_when_one_seed_interaction_drop_is_zero():
    result = mod.decide_h10(
        make_h10_seed_results(
            d_i_values=(0.2, 0.0, 0.4),
            g_values=(0.01, 0.02, 0.03),
        )
    )

    assert result[
        "interaction_suppression_positive_seed_count"
    ] == 2
    assert result["decision"] == "NOT_SUPPORTED"


def test_h10_fails_when_mean_g_is_zero():
    result = mod.decide_h10(
        make_h10_seed_results(
            d_i_values=(0.2, 0.3, 0.4),
            g_values=(0.01, -0.01, 0.0),
        )
    )

    assert math.isclose(
        result["mean_G"],
        0.0,
        abs_tol=1e-12,
    )
    assert result["decision"] == "NOT_SUPPORTED"


def test_h10_fails_when_mean_g_negative():
    result = mod.decide_h10(
        make_h10_seed_results(
            d_i_values=(0.2, 0.3, 0.4),
            g_values=(-0.01, -0.02, 0.01),
        )
    )

    assert result["mean_G"] < 0.0
    assert result["decision"] == "NOT_SUPPORTED"


def test_h10_rejects_wrong_seed_result_count():
    with pytest.raises(mod.FormalAnalysisError):
        mod.decide_h10(
            make_h10_seed_results()[:2]
        )


def test_probability_accepts_valid_boundaries():
    assert mod.probability(0.0, "x") == 0.0
    assert mod.probability(1.0, "x") == 1.0


def test_probability_rejects_out_of_range():
    with pytest.raises(mod.FormalAnalysisError):
        mod.probability(-0.001, "x")

    with pytest.raises(mod.FormalAnalysisError):
        mod.probability(1.001, "x")


def test_finite_number_rejects_nan_and_inf():
    with pytest.raises(mod.FormalAnalysisError):
        mod.finite_number(float("nan"), "x")

    with pytest.raises(mod.FormalAnalysisError):
        mod.finite_number(float("inf"), "x")


def test_not_computed_contract_is_exact():
    mod.validate_not_computed(
        {
            "H8_F": "NOT_COMPUTED",
            "H9_F": "NOT_COMPUTED",
            "H10_F": "NOT_COMPUTED",
        },
        "test",
    )

    with pytest.raises(mod.FormalAnalysisError):
        mod.validate_not_computed(
            {
                "H8_F": "SUPPORTED",
                "H9_F": "NOT_COMPUTED",
                "H10_F": "NOT_COMPUTED",
            },
            "test",
        )


def test_mean_diag_reads_persisted_summary_mean():
    obj = {
        "diagnostics": {
            "text_weight": {
                "mean": 0.375,
            }
        }
    }

    assert mod.mean_diag(
        obj,
        "text_weight",
        "synthetic",
    ) == 0.375


def test_macro_f1_reads_persisted_metric():
    obj = {
        "metrics": {
            "macro_f1": 0.8125,
        }
    }

    assert mod.macro_f1(
        obj,
        "synthetic",
    ) == 0.8125


def test_output_decisions_use_supported_or_not_supported_only():
    h8 = mod.decide_h8(
        make_h8_observations([0.05] * 54)
    )
    h9 = mod.decide_h9(
        make_h9_observations([0.02] * 54)
    )
    h10 = mod.decide_h10(
        make_h10_seed_results()
    )

    assert h8["decision"] in {
        "SUPPORTED",
        "NOT_SUPPORTED",
    }
    assert h9["decision"] in {
        "SUPPORTED",
        "NOT_SUPPORTED",
    }
    assert h10["decision"] in {
        "SUPPORTED",
        "NOT_SUPPORTED",
    }

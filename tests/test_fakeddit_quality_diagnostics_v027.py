"""
Dedicated tests for AEGIS v0.27 Step 9 M4q quality diagnostics.

These tests validate the scientific invariants of the diagnostic runner
without accessing Fakeddit experiment artifacts or the official test split.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from scripts.run_fakeddit_quality_diagnostics_v027 import (
    NONZERO_SEVERITIES,
    VALIDATION_CORRUPTION_BASE_SEED,
    add_clean_relative_quality_deltas,
    apply_condition,
    assert_safe_input_path,
    build_conditions,
    condition_slug,
    validate_m4q_checkpoint_metadata,
    validation_corruption_seed,
)


def test_full_protocol_has_exactly_19_conditions():
    conditions = build_conditions(smoke=False)

    assert len(conditions) == 19
    assert conditions[0]["corruption_type"] == "clean"
    assert conditions[0]["corrupted_modality"] == "none"


def test_full_protocol_condition_slugs_are_unique():
    conditions = build_conditions(smoke=False)
    slugs = [condition_slug(condition) for condition in conditions]

    assert len(slugs) == 19
    assert len(set(slugs)) == 19


def test_continuous_conditions_exclude_redundant_zero_severity():
    conditions = build_conditions(smoke=False)

    continuous = [
        condition
        for condition in conditions
        if condition["corruption_type"]
        in {"gaussian_noise", "attenuation"}
    ]

    assert len(continuous) == 16
    assert all(
        condition["severity"] in NONZERO_SEVERITIES
        for condition in continuous
    )
    assert all(condition["severity"] != 0.0 for condition in continuous)


def test_zero_dropout_is_discrete_for_both_modalities():
    conditions = build_conditions(smoke=False)

    zero_rows = [
        condition
        for condition in conditions
        if condition["corruption_type"] == "zero_dropout"
    ]

    assert len(zero_rows) == 2
    assert {
        row["corrupted_modality"]
        for row in zero_rows
    } == {"text", "vision"}
    assert all(row["severity"] == 1.0 for row in zero_rows)


def test_smoke_protocol_is_clean_plus_one_gaussian_condition():
    conditions = build_conditions(smoke=True)

    assert len(conditions) == 2
    assert condition_slug(conditions[0]) == "clean"
    assert condition_slug(conditions[1]) == "text_gaussian_noise_s0p25"


def test_validation_corruption_seed_signature_has_no_model_seed():
    signature = inspect.signature(validation_corruption_seed)

    assert "model_seed" not in signature.parameters
    assert "seed" not in signature.parameters


def test_validation_corruption_seed_is_deterministic():
    first = validation_corruption_seed(
        "gaussian_noise",
        "text",
        0.25,
    )
    second = validation_corruption_seed(
        "gaussian_noise",
        "text",
        0.25,
    )

    assert first == second
    assert first > VALIDATION_CORRUPTION_BASE_SEED


def test_validation_corruption_seed_changes_by_condition():
    seeds = {
        validation_corruption_seed("gaussian_noise", "text", 0.25),
        validation_corruption_seed("gaussian_noise", "text", 0.50),
        validation_corruption_seed("gaussian_noise", "vision", 0.25),
        validation_corruption_seed("attenuation", "text", 0.25),
        validation_corruption_seed("zero_dropout", "text", 1.0),
    }

    assert len(seeds) == 5


@pytest.mark.parametrize(
    "corruption_type, modality, severity",
    [
        ("unknown", "text", 0.25),
        ("gaussian_noise", "audio", 0.25),
        ("gaussian_noise", "text", -0.1),
        ("gaussian_noise", "text", 1.1),
    ],
)
def test_validation_corruption_seed_rejects_invalid_inputs(
    corruption_type,
    modality,
    severity,
):
    with pytest.raises(ValueError):
        validation_corruption_seed(
            corruption_type,
            modality,
            severity,
        )


def test_clean_condition_clones_without_modification():
    text = torch.randn(5, 4)
    vision = torch.randn(5, 3)

    out_text, out_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=torch.ones(4),
        vision_feature_std=torch.ones(3),
        condition={
            "corruption_type": "clean",
            "corrupted_modality": "none",
            "severity": None,
            "corruption_seed": None,
        },
    )

    assert torch.equal(out_text, text)
    assert torch.equal(out_vision, vision)
    assert out_text.data_ptr() != text.data_ptr()
    assert out_vision.data_ptr() != vision.data_ptr()


def test_text_attenuation_changes_only_text():
    text = torch.ones(4, 3)
    vision = torch.randn(4, 2)

    out_text, out_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=torch.ones(3),
        vision_feature_std=torch.ones(2),
        condition={
            "corruption_type": "attenuation",
            "corrupted_modality": "text",
            "severity": 0.5,
            "corruption_seed": validation_corruption_seed(
                "attenuation", "text", 0.5
            ),
        },
    )

    assert torch.allclose(out_text, torch.full_like(text, 0.5))
    assert torch.equal(out_vision, vision)


def test_vision_attenuation_changes_only_vision():
    text = torch.randn(4, 3)
    vision = torch.ones(4, 2)

    out_text, out_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=torch.ones(3),
        vision_feature_std=torch.ones(2),
        condition={
            "corruption_type": "attenuation",
            "corrupted_modality": "vision",
            "severity": 0.75,
            "corruption_seed": validation_corruption_seed(
                "attenuation", "vision", 0.75
            ),
        },
    )

    assert torch.equal(out_text, text)
    assert torch.allclose(out_vision, torch.full_like(vision, 0.25))


def test_zero_dropout_changes_only_selected_modality():
    text = torch.randn(4, 3)
    vision = torch.randn(4, 2)

    out_text, out_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=torch.ones(3),
        vision_feature_std=torch.ones(2),
        condition={
            "corruption_type": "zero_dropout",
            "corrupted_modality": "text",
            "severity": 1.0,
            "corruption_seed": validation_corruption_seed(
                "zero_dropout", "text", 1.0
            ),
        },
    )

    assert torch.equal(out_text, torch.zeros_like(text))
    assert torch.equal(out_vision, vision)


def test_gaussian_corruption_is_deterministic():
    text = torch.randn(8, 5)
    vision = torch.randn(8, 3)
    text_std = torch.linspace(0.1, 0.5, 5)

    condition = {
        "corruption_type": "gaussian_noise",
        "corrupted_modality": "text",
        "severity": 0.5,
        "corruption_seed": validation_corruption_seed(
            "gaussian_noise", "text", 0.5
        ),
    }

    first_text, first_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=text_std,
        vision_feature_std=torch.ones(3),
        condition=condition,
    )
    second_text, second_vision = apply_condition(
        text_embeddings=text,
        vision_embeddings=vision,
        text_feature_std=text_std,
        vision_feature_std=torch.ones(3),
        condition=condition,
    )

    assert torch.equal(first_text, second_text)
    assert torch.equal(first_vision, second_vision)
    assert torch.equal(first_vision, vision)


def test_quality_selectivity_sign_is_positive_when_corrupted_drops_more():
    summaries = [
        {
            "corruption_type": "clean",
            "corrupted_modality": "none",
            "text_quality_mean": 0.90,
            "vision_quality_mean": 0.90,
        },
        {
            "corruption_type": "attenuation",
            "corrupted_modality": "text",
            "text_quality_mean": 0.40,
            "vision_quality_mean": 0.85,
        },
    ]

    add_clean_relative_quality_deltas(summaries)

    corrupted = summaries[1]
    assert corrupted["delta_text_quality_from_clean"] == pytest.approx(-0.50)
    assert corrupted["delta_vision_quality_from_clean"] == pytest.approx(-0.05)
    assert corrupted["quality_selectivity"] == pytest.approx(0.45)
    assert corrupted["quality_selectivity"] > 0.0


def test_quality_selectivity_sign_is_symmetric_for_vision_corruption():
    summaries = [
        {
            "corruption_type": "clean",
            "corrupted_modality": "none",
            "text_quality_mean": 0.88,
            "vision_quality_mean": 0.92,
        },
        {
            "corruption_type": "gaussian_noise",
            "corrupted_modality": "vision",
            "text_quality_mean": 0.84,
            "vision_quality_mean": 0.50,
        },
    ]

    add_clean_relative_quality_deltas(summaries)

    corrupted = summaries[1]
    assert corrupted["corrupted_modality_quality"] == pytest.approx(0.50)
    assert corrupted["intact_modality_quality"] == pytest.approx(0.84)
    assert corrupted["quality_selectivity"] == pytest.approx(0.38)


def test_clean_summary_has_no_selectivity():
    summaries = [
        {
            "corruption_type": "clean",
            "corrupted_modality": "none",
            "text_quality_mean": 0.8,
            "vision_quality_mean": 0.9,
        }
    ]

    add_clean_relative_quality_deltas(summaries)

    assert summaries[0]["quality_selectivity"] is None
    assert summaries[0]["delta_text_quality_from_clean"] == pytest.approx(0.0)
    assert summaries[0]["delta_vision_quality_from_clean"] == pytest.approx(0.0)


def test_add_clean_relative_quality_deltas_requires_exactly_one_clean():
    with pytest.raises(RuntimeError):
        add_clean_relative_quality_deltas(
            [
                {
                    "corruption_type": "attenuation",
                    "corrupted_modality": "text",
                    "text_quality_mean": 0.5,
                    "vision_quality_mean": 0.9,
                }
            ]
        )


def _valid_checkpoint(seed=42):
    return {
        "epoch": 3,
        "ablation_mode": "multimodal",
        "fusion_architecture": "quality_supervised",
        "alignment_model_state_dict": {},
        "classification_model_state_dict": {},
        "configuration": {
            "seed": seed,
            "fusion_architecture": "quality_supervised",
            "quality_supervised": True,
            "shared_dim": 512,
            "dropout": 0.1,
            "temperature": 0.07,
        },
    }


def test_checkpoint_metadata_accepts_m4q():
    configuration = validate_m4q_checkpoint_metadata(
        _valid_checkpoint(42),
        expected_seed=42,
    )

    assert configuration["quality_supervised"] is True
    assert configuration["seed"] == 42


@pytest.mark.parametrize(
    "field, value",
    [
        ("ablation_mode", "text_only"),
        ("fusion_architecture", "gated_interaction"),
    ],
)
def test_checkpoint_metadata_rejects_wrong_top_level_contract(field, value):
    checkpoint = _valid_checkpoint()
    checkpoint[field] = value

    with pytest.raises(RuntimeError):
        validate_m4q_checkpoint_metadata(checkpoint, expected_seed=42)


def test_checkpoint_metadata_rejects_quality_flag_false():
    checkpoint = _valid_checkpoint()
    checkpoint["configuration"]["quality_supervised"] = False

    with pytest.raises(RuntimeError):
        validate_m4q_checkpoint_metadata(checkpoint, expected_seed=42)


def test_checkpoint_metadata_rejects_seed_mismatch():
    checkpoint = _valid_checkpoint(seed=43)

    with pytest.raises(RuntimeError):
        validate_m4q_checkpoint_metadata(checkpoint, expected_seed=42)


def test_checkpoint_metadata_requires_alignment_state():
    checkpoint = _valid_checkpoint()
    checkpoint.pop("alignment_model_state_dict")

    with pytest.raises(RuntimeError):
        validate_m4q_checkpoint_metadata(checkpoint, expected_seed=42)


@pytest.mark.parametrize(
    "path",
    [
        "data/processed/fakeddit/test",
        "data/processed/fakeddit/testing/cache",
        "data/processed/fakeddit/official_test/cache",
        "data/processed/fakeddit/test_split/cache",
    ],
)
def test_safe_input_path_rejects_test_components(path):
    with pytest.raises(RuntimeError):
        assert_safe_input_path(torch_path(path), "fixture")


def torch_path(value):
    from pathlib import Path

    return Path(value)


def test_safe_input_path_accepts_frozen_validation_path():
    assert_safe_input_path(
        torch_path(
            "data/processed/fakeddit/frozen_embeddings/"
            "validation_n1000_seed42"
        ),
        "validation cache",
    )

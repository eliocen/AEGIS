"""Tests for AEGIS v0.27 Step14A M4qcf robustness diagnostics."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


MODULE = importlib.import_module(
    "scripts.run_fakeddit_m4qcf_robustness_diagnostics_v027"
)


def test_protocol_version_is_frozen():
    assert MODULE.VERSION == "0.27.0-step14a"
    assert MODULE.PARENT_PROTOCOL_VERSION == "0.27.0-step13a"


def test_architecture_set_is_exact():
    assert MODULE.ARCHITECTURES == ("M1b", "M4qc", "M4qcf")


def test_architecture_fusion_identities():
    assert (
        MODULE.ARCHITECTURE_SPECS["M1b"]["fusion_architecture"]
        == "gated_interaction"
    )
    assert (
        MODULE.ARCHITECTURE_SPECS["M4qc"]["fusion_architecture"]
        == "quality_compatibility_supervised"
    )
    assert (
        MODULE.ARCHITECTURE_SPECS["M4qcf"]["fusion_architecture"]
        == "quality_compatibility_fusion"
    )


def test_formal_seed_set_is_exact():
    assert MODULE.DEFAULT_SEEDS == (42, 43, 44)


def test_checkpoint_paths_are_frozen():
    root = Path("experiments/fakeddit")

    assert MODULE.checkpoint_path(root, "M1b", 42) == (
        root
        / "v025_m1b_gated_interaction_seed42"
        / "best_model.pt"
    )
    assert MODULE.checkpoint_path(root, "M4qc", 43) == (
        root / "v027_m4qc_seed43" / "best_model.pt"
    )
    assert MODULE.checkpoint_path(root, "M4qcf", 44) == (
        root / "v027_m4qcf_seed44" / "best_model.pt"
    )


def test_unknown_architecture_fails():
    with pytest.raises(ValueError):
        MODULE.checkpoint_path(
            Path("experiments/fakeddit"),
            "unknown",
            42,
        )


def test_binary_metrics_perfect():
    metrics = MODULE.binary_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 0, 1, 1],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_binary_metrics_balanced_failure():
    metrics = MODULE.binary_metrics(
        targets=[0, 0, 1, 1],
        predictions=[1, 1, 0, 0],
    )

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(0.0)


def test_binary_metrics_reject_nonbinary_targets():
    with pytest.raises(RuntimeError):
        MODULE.binary_metrics(
            targets=[0, 2],
            predictions=[0, 1],
        )


def test_binary_metrics_reject_length_mismatch():
    with pytest.raises(RuntimeError):
        MODULE.binary_metrics(
            targets=[0, 1],
            predictions=[0],
        )


def test_jsonable_hash_is_deterministic():
    first = MODULE.sha256_jsonable(
        {"a": 1, "b": [2, 3]}
    )
    second = MODULE.sha256_jsonable(
        {"b": [2, 3], "a": 1}
    )

    assert first == second
    assert len(first) == 64


def test_optional_scalar_accepts_valid_tensor():
    outputs = {
        "text_quality": torch.tensor([[0.2], [0.8]])
    }

    result = MODULE.optional_scalar(
        outputs,
        "text_quality",
        2,
    )

    assert result is not None
    assert result.shape == (2,)
    assert result.tolist() == pytest.approx([0.2, 0.8])


def test_optional_scalar_returns_none_when_absent():
    assert (
        MODULE.optional_scalar(
            {},
            "text_quality",
            2,
        )
        is None
    )


def test_optional_scalar_rejects_wrong_count():
    with pytest.raises(RuntimeError):
        MODULE.optional_scalar(
            {"text_quality": torch.tensor([0.1])},
            "text_quality",
            2,
        )


def test_optional_scalar_rejects_nonfinite():
    with pytest.raises(RuntimeError):
        MODULE.optional_scalar(
            {
                "text_quality": torch.tensor(
                    [0.1, float("nan")]
                )
            },
            "text_quality",
            2,
        )


def test_bounded_unit_interval_accepts_boundaries():
    MODULE.bounded_unit_interval(
        torch.tensor([0.0, 0.5, 1.0]),
        "score",
    )


def test_bounded_unit_interval_rejects_negative():
    with pytest.raises(RuntimeError):
        MODULE.bounded_unit_interval(
            torch.tensor([-0.01, 0.5]),
            "score",
        )


def test_bounded_unit_interval_rejects_above_one():
    with pytest.raises(RuntimeError):
        MODULE.bounded_unit_interval(
            torch.tensor([0.5, 1.01]),
            "score",
        )


def test_population_stats():
    stats = MODULE.population_stats([1.0, 2.0, 3.0])

    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)


def test_mapping_records_preserve_identity_fields():
    mapping = torch.tensor([1, 0, 3, 2])
    targets = torch.tensor([0, 0, 1, 1])
    sample_ids = ["a", "b", "c", "d"]

    rows = MODULE.mapping_records(
        mapping,
        sample_ids=sample_ids,
        targets=targets,
    )

    assert len(rows) == 4
    assert rows[0]["receiver_sample_id"] == "a"
    assert rows[0]["vision_donor_sample_id"] == "b"
    assert rows[0]["receiver_class"] == 0
    assert rows[0]["donor_class"] == 0
    assert rows[0]["self_pair"] is False
    assert rows[0]["class_preserved"] is True


def test_mapping_records_expose_class_change():
    mapping = torch.tensor([2, 0, 3, 1])
    targets = torch.tensor([0, 0, 1, 1])
    sample_ids = ["a", "b", "c", "d"]

    rows = MODULE.mapping_records(
        mapping,
        sample_ids=sample_ids,
        targets=targets,
    )

    assert rows[0]["class_preserved"] is False


def test_formal_condition_count_is_19():
    conditions = MODULE.build_conditions(smoke=False)
    assert len(conditions) == 19


def test_smoke_condition_count_is_2():
    conditions = MODULE.build_conditions(smoke=True)
    assert len(conditions) == 2


def test_formal_condition_matrix_is_unique():
    conditions = MODULE.build_conditions(smoke=False)
    slugs = [MODULE.condition_slug(c) for c in conditions]

    assert len(slugs) == 19
    assert len(set(slugs)) == 19
    assert "clean" in slugs


def test_formal_condition_matrix_has_expected_nonclean_count():
    conditions = MODULE.build_conditions(smoke=False)

    nonclean = [
        condition
        for condition in conditions
        if condition["corruption_type"] != "clean"
    ]

    assert len(nonclean) == 18


def test_formal_matrix_contains_ten_text_and_vision_conditions_total():
    conditions = MODULE.build_conditions(smoke=False)

    text = [
        c for c in conditions
        if c["corrupted_modality"] == "text"
    ]
    vision = [
        c for c in conditions
        if c["corrupted_modality"] == "vision"
    ]

    assert len(text) == 9
    assert len(vision) == 9


def test_corruption_seeds_do_not_depend_on_model_seed():
    conditions_a = MODULE.build_conditions(smoke=False)
    conditions_b = MODULE.build_conditions(smoke=False)

    assert conditions_a == conditions_b


def test_finite_float_rejects_nan():
    with pytest.raises(RuntimeError):
        MODULE.finite_float(float("nan"), "value")


def test_finite_float_rejects_inf():
    with pytest.raises(RuntimeError):
        MODULE.finite_float(float("inf"), "value")


def test_require_fails_loudly():
    with pytest.raises(RuntimeError):
        MODULE.require(False, "expected failure")


def test_require_accepts_true():
    MODULE.require(True, "must not fail")


def test_output_root_is_v027_step14a():
    assert MODULE.DEFAULT_OUTPUT_ROOT == Path(
        "experiments/fakeddit/"
        "v027_m4qcf_robustness_diagnostics"
    )


def test_expected_sample_counts_are_frozen():
    assert MODULE.EXPECTED_TRAIN_SAMPLES == 5000
    assert MODULE.EXPECTED_VALIDATION_SAMPLES == 1000


def test_m4qc_protocol_version_is_frozen():
    assert MODULE.M4QC_PROTOCOL_VERSION == "0.27.0-step11a"


def test_m4qcf_protocol_version_is_frozen():
    assert MODULE.M4QCF_PROTOCOL_VERSION == "0.27.0-step13a"


def test_weight_sum_tolerance_is_small():
    assert MODULE.WEIGHT_SUM_TOLERANCE <= 1e-5


def test_no_hypothesis_decision_function_is_exposed():
    forbidden = {
        "decide_h8",
        "decide_h9",
        "decide_h10",
        "evaluate_h8",
        "evaluate_h9",
        "evaluate_h10",
    }

    assert forbidden.isdisjoint(set(dir(MODULE)))


def test_source_does_not_reference_official_test_cache():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8-sig")

    assert "test_n" not in source
    assert "official_test_accessed\": True" not in source
    assert "official_test_samples_accessed\": 1000" not in source


def test_source_has_no_optimizer_creation():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8-sig")

    assert "torch.optim" not in source
    assert "optimizer =" not in source


def test_source_has_no_backward_call():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8-sig")

    assert ".backward(" not in source


def test_source_reuses_step9_corruption_runner():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8-sig")

    assert (
        "scripts.run_fakeddit_quality_diagnostics_v027"
        in source
    )
    assert "precompute_validation_conditions" in source


def test_source_reuses_step12b_derangement():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8-sig")

    assert (
        "scripts.run_fakeddit_m4qc_compatibility_diagnostics_v027"
        in source
    )
    assert "build_class_preserving_derangement" in source


def test_formal_evaluation_count_is_171():
    condition_count = len(MODULE.build_conditions(smoke=False))
    total = (
        condition_count
        * len(MODULE.ARCHITECTURES)
        * len(MODULE.DEFAULT_SEEDS)
    )

    assert total == 171


def test_resolve_runtime_device_cpu():
    device = MODULE.resolve_runtime_device("cpu")

    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_resolve_runtime_device_auto_returns_torch_device():
    device = MODULE.resolve_runtime_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_resolve_runtime_device_none_returns_torch_device():
    device = MODULE.resolve_runtime_device(None)

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_resolve_runtime_device_explicit_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this test environment.")

    device = MODULE.resolve_runtime_device("cuda")

    assert isinstance(device, torch.device)
    assert device.type == "cuda"


def test_build_frozen_mismatch_mapping_consumes_step12b_tuple_contract():
    targets = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )

    mapping = MODULE.build_frozen_mismatch_mapping(targets)

    assert isinstance(mapping, torch.Tensor)
    assert mapping.dtype == torch.long
    assert mapping.shape == targets.shape

    indices = torch.arange(targets.numel(), dtype=torch.long)

    # Frozen primary mismatch must contain no self-pairs.
    assert torch.all(mapping != indices)

    # Every donor must preserve the receiver's Stage-1 class.
    assert torch.equal(
        targets,
        targets.index_select(0, mapping),
    )

    # Each vision representation must occur exactly once within its class.
    for class_id in torch.unique(targets).tolist():
        receivers = torch.nonzero(
            targets == int(class_id),
            as_tuple=False,
        ).reshape(-1)

        donors = mapping.index_select(0, receivers)

        assert torch.equal(
            torch.sort(donors).values,
            torch.sort(receivers).values,
        )

    # Construction must be deterministic.
    repeated = MODULE.build_frozen_mismatch_mapping(targets)
    assert torch.equal(mapping, repeated)


def test_step14a_uses_exact_frozen_step12b_validator_contract(monkeypatch):
    targets = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )

    calls = []

    original = MODULE.validate_class_preserving_derangement

    def exact_validator(*, targets, donor_indices):
        calls.append((targets.clone(), donor_indices.clone()))
        return original(
            targets=targets,
            donor_indices=donor_indices,
        )

    monkeypatch.setattr(
        MODULE,
        "validate_class_preserving_derangement",
        exact_validator,
    )

    mapping = MODULE.build_frozen_mismatch_mapping(targets)

    assert len(calls) == 1
    assert torch.equal(calls[0][0], targets)
    assert torch.equal(calls[0][1], mapping)


def test_step14a_source_distinguishes_smoke_and_formal_summary_status():
    source = Path(
        "scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py"
    ).read_text(encoding="utf-8")

    assert '"STEP14A_SMOKE_COMPLETE"' in source
    assert '"STEP14A_DIAGNOSTIC_EVIDENCE_COMPLETE"' in source
    assert "if args.smoke" in source

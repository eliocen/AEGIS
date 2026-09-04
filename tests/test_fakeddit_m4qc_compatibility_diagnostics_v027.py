from __future__ import annotations

import pytest
import torch

from scripts.run_fakeddit_m4qc_compatibility_diagnostics_v027 import (
    DERANGEMENT_SEED_BASE,
    build_class_preserving_derangement,
    validate_class_preserving_derangement,
)


def _balanced_targets() -> torch.Tensor:
    return torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)


def test_class_preserving_derangement_is_deterministic():
    targets = _balanced_targets()

    donors_a, metadata_a = build_class_preserving_derangement(targets)
    donors_b, metadata_b = build_class_preserving_derangement(targets)

    assert torch.equal(donors_a, donors_b)
    assert metadata_a == metadata_b


def test_class_preserving_derangement_has_no_self_pairs():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)

    rows = torch.arange(targets.numel(), dtype=torch.long)
    assert torch.all(donors != rows)


def test_class_preserving_derangement_preserves_stage1_class():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)

    assert torch.equal(targets[donors], targets)


def test_each_vision_is_used_exactly_once_within_each_class():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)

    for class_id in [0, 1]:
        indices = torch.nonzero(
            targets == class_id,
            as_tuple=False,
        ).reshape(-1)
        observed = torch.sort(donors[indices]).values
        expected = torch.sort(indices).values
        assert torch.equal(observed, expected)


def test_metadata_uses_frozen_class_seed_rule():
    targets = _balanced_targets()
    _, metadata = build_class_preserving_derangement(targets)

    assert metadata[0]["rng_seed"] == DERANGEMENT_SEED_BASE
    assert metadata[1]["rng_seed"] == DERANGEMENT_SEED_BASE + 1
    assert metadata[0]["rng_engine"] == "torch.Generator(cpu)+torch.randperm"
    assert metadata[1]["rng_engine"] == "torch.Generator(cpu)+torch.randperm"


def test_mapping_is_independent_of_model_seed_by_construction():
    targets = _balanced_targets()

    # The mapping function intentionally has no model-seed argument.
    donors_42, _ = build_class_preserving_derangement(targets)
    donors_43, _ = build_class_preserving_derangement(targets)
    donors_44, _ = build_class_preserving_derangement(targets)

    assert torch.equal(donors_42, donors_43)
    assert torch.equal(donors_43, donors_44)


def test_class_with_fewer_than_two_samples_fails_loudly():
    targets = torch.tensor([0, 0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="at least two"):
        build_class_preserving_derangement(targets)


def test_non_1d_targets_fail_loudly():
    targets = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="1-D"):
        build_class_preserving_derangement(targets)


def test_empty_targets_fail_loudly():
    targets = torch.empty(0, dtype=torch.long)

    with pytest.raises(ValueError, match="at least one"):
        build_class_preserving_derangement(targets)


def test_validator_rejects_self_pair():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)
    donors = donors.clone()
    donors[0] = 0

    with pytest.raises(ValueError, match="self-pairs"):
        validate_class_preserving_derangement(targets, donors)


def test_validator_rejects_class_change():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)
    donors = donors.clone()
    donors[0] = 10

    with pytest.raises(ValueError, match="class labels"):
        validate_class_preserving_derangement(targets, donors)


def test_validator_rejects_duplicate_or_missing_vision_within_class():
    targets = _balanced_targets()
    donors, _ = build_class_preserving_derangement(targets)
    donors = donors.clone()

    # Keep the replacement within class and avoid a self-pair, but duplicate
    # one donor so the one-to-one vision-use invariant is violated.
    donors[0] = donors[1]
    if donors[0].item() == 0:
        donors[0] = donors[2]

    with pytest.raises(ValueError, match="exactly once within class"):
        validate_class_preserving_derangement(targets, donors)

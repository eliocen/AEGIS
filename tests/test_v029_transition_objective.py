"""Tests for the prospectively frozen v0.29 transition objective."""

import math
import sys

import pytest
import torch

from scripts.run_fakeddit_ablation import (
    harmful_transition_loss,
    parse_args,
)


def test_harmful_transition_loss_is_one_sided_and_target_aware():
    matched = torch.tensor([
        [-2.0, 2.0],
        [2.0, -2.0],
    ])

    mismatched = torch.tensor([
        [0.0, 0.0],
        [3.0, -3.0],
    ])

    targets = torch.tensor([1, 0])
    mask = torch.tensor([True, True])

    loss = harmful_transition_loss(
        matched,
        mismatched,
        targets,
        mask,
    )

    matched_probability = torch.softmax(matched, dim=-1)
    mismatched_probability = torch.softmax(mismatched, dim=-1)

    expected_first = (
        matched_probability[0, 1]
        - mismatched_probability[0, 1]
    )

    expected_second = torch.tensor(0.0)

    expected = (
        expected_first + expected_second
    ) / 2.0

    assert torch.allclose(loss, expected)


def test_beneficial_transition_has_exactly_zero_penalty():
    loss = harmful_transition_loss(
        matched_logits=torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        mismatched_logits=torch.tensor([
            [-2.0, 2.0],
            [2.0, -2.0],
        ]),
        integrity_targets=torch.tensor([1, 0]),
        mismatch_mask=torch.tensor([True, True]),
    )

    assert loss.item() == 0.0


def test_non_mismatch_rows_do_not_enter_the_objective():
    matched = torch.tensor([
        [-3.0, 3.0],
        [-3.0, 3.0],
    ])

    mismatched = torch.tensor([
        [3.0, -3.0],
        [3.0, -3.0],
    ])

    loss = harmful_transition_loss(
        matched_logits=matched,
        mismatched_logits=mismatched,
        integrity_targets=torch.tensor([1, 1]),
        mismatch_mask=torch.tensor([False, True]),
    )

    matched_probability = torch.softmax(
        matched,
        dim=-1,
    )[1, 1]

    mismatched_probability = torch.softmax(
        mismatched,
        dim=-1,
    )[1, 1]

    expected = (
        matched_probability
        - mismatched_probability
    )

    assert torch.allclose(loss, expected)


def test_matched_reference_is_detached_but_mismatch_receives_gradient():
    matched = torch.tensor(
        [[-2.0, 2.0]],
        requires_grad=True,
    )

    mismatched = torch.tensor(
        [[0.0, 0.0]],
        requires_grad=True,
    )

    loss = harmful_transition_loss(
        matched,
        mismatched,
        torch.tensor([1]),
        torch.tensor([True]),
    )

    loss.backward()

    assert matched.grad is None
    assert mismatched.grad is not None

    assert mismatched.grad[0, 1].item() < 0.0


def test_empty_mismatch_mask_returns_differentiable_zero():
    mismatched = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -1.0],
        ],
        requires_grad=True,
    )

    loss = harmful_transition_loss(
        torch.tensor([
            [1.0, -1.0],
            [2.0, -2.0],
        ]),
        mismatched,
        torch.tensor([0, 0]),
        torch.tensor([False, False]),
    )

    assert loss.item() == 0.0

    loss.backward()

    assert torch.equal(
        mismatched.grad,
        torch.zeros_like(mismatched),
    )


def test_column_mismatch_mask_is_accepted_and_normalized():
    common = dict(
        matched_logits=torch.tensor([
            [-2.0, 2.0],
            [-2.0, 2.0],
        ]),
        mismatched_logits=torch.tensor([
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        integrity_targets=torch.tensor([1, 1]),
    )

    loss_vector = harmful_transition_loss(
        **common,
        mismatch_mask=torch.tensor(
            [True, False]
        ),
    )

    loss_column = harmful_transition_loss(
        **common,
        mismatch_mask=torch.tensor(
            [[True], [False]]
        ),
    )

    assert torch.allclose(
        loss_vector,
        loss_column,
    )


@pytest.mark.parametrize(
    "matched,mismatched,targets,mask",
    [
        (
            torch.zeros(2, 2),
            torch.zeros(3, 2),
            torch.zeros(3, dtype=torch.long),
            torch.ones(3, dtype=torch.bool),
        ),
        (
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
        ),
        (
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, 1, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
        ),
        (
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(3, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
        ),
        (
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
            torch.ones(3, dtype=torch.bool),
        ),
    ],
)
def test_transition_objective_rejects_invalid_tensor_contract(
    matched,
    mismatched,
    targets,
    mask,
):
    with pytest.raises(ValueError):
        harmful_transition_loss(
            matched,
            mismatched,
            targets,
            mask,
        )


def test_transition_objective_rejects_nonbinary_targets():
    with pytest.raises(ValueError):
        harmful_transition_loss(
            matched_logits=torch.zeros(2, 2),
            mismatched_logits=torch.zeros(2, 2),
            integrity_targets=torch.tensor([0, 2]),
            mismatch_mask=torch.tensor([True, True]),
        )


def test_runner_exposes_frozen_transition_objective_default(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fakeddit_ablation",
            "--mode",
            "multimodal",
        ],
    )

    args = parse_args()

    assert math.isclose(
        args.transition_objective_weight,
        0.25,
    )


def test_runner_accepts_explicit_frozen_transition_weight(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fakeddit_ablation",
            "--mode",
            "multimodal",
            "--transition-objective-weight",
            "0.25",
        ],
    )

    args = parse_args()

    assert args.transition_objective_weight == 0.25

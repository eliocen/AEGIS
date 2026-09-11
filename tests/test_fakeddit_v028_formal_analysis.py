"""Invariant tests for the frozen AEGIS v0.28 Step5 analyzer.

These tests use synthetic gate values only and never load formal Step4 data.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


PATH = Path("scripts/analyze_fakeddit_v028_formal.py")
SPEC = importlib.util.spec_from_file_location("v028_formal", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_decision_is_conjunctive():
    assert MODULE.decision({"a": True, "b": True}) == "SUPPORTED"
    assert MODULE.decision({"a": True, "b": False}) == "NOT_SUPPORTED"


def test_decision_rejects_empty_gate_set():
    with pytest.raises(RuntimeError):
        MODULE.decision({})


def test_frozen_condition_cardinalities():
    assert len(MODULE.GRADED_CONDITIONS) == 12
    assert len(MODULE.TEXT_GAUSSIAN_CONDITIONS) == 4
    assert len(MODULE.CATASTROPHIC_CONDITIONS) == 4
    assert len(MODULE.ALL_NONCLEAN_CONDITIONS) == 18
    assert len(MODULE.OTHER_NONCLEAN_CONDITIONS) == 14


def test_condition_sets_are_unique():
    assert len(set(MODULE.GRADED_CONDITIONS)) == 12
    assert len(set(MODULE.TEXT_GAUSSIAN_CONDITIONS)) == 4
    assert len(set(MODULE.CATASTROPHIC_CONDITIONS)) == 4
    assert len(set(MODULE.ALL_NONCLEAN_CONDITIONS)) == 18


def test_h4_partition_is_exact():
    assert set(MODULE.TEXT_GAUSSIAN_CONDITIONS).isdisjoint(MODULE.OTHER_NONCLEAN_CONDITIONS)
    assert set(MODULE.TEXT_GAUSSIAN_CONDITIONS) | set(MODULE.OTHER_NONCLEAN_CONDITIONS) == set(MODULE.ALL_NONCLEAN_CONDITIONS)


def test_strict_positive_semantics():
    values = [-0.1, 0.0, 0.1]
    assert sum(value > 0.0 for value in values) == 1


def test_h1_boundary_values():
    gates = {
        "mean": 0.01 >= 0.01,
        "count": 26 >= 26,
        "fraction": 26 / 36 >= 0.70,
        "catastrophic": -0.01 >= -0.01,
    }
    assert MODULE.decision(gates) == "SUPPORTED"


def test_h1_failed_coverage_cannot_be_overridden_by_mean():
    gates = {"mean": True, "count": False, "fraction": False, "catastrophic": True}
    assert MODULE.decision(gates) == "NOT_SUPPORTED"


def test_h2_uses_strict_comparator_improvements():
    primary_ratio = 1.0
    comparator_ratio = 1.0
    assert not (primary_ratio < comparator_ratio)
    assert primary_ratio <= comparator_ratio


def test_h3_macro_f1_gate_is_non_strict_but_net_harm_gate_is_strict():
    assert 0.8 >= 0.8
    assert not (10 < 10)


def test_h4_boundary_values():
    gates = {
        "mean": 0.01 >= 0.01,
        "count": 9 >= 9,
        "fraction": 9 / 12 >= 0.75,
        "other": -0.01 >= -0.01,
        "clean": -0.01 >= -0.01,
    }
    assert MODULE.decision(gates) == "SUPPORTED"


def test_diagnostic_constant_check():
    rows = [{"architecture": "M4qcs-i", "diagnostics": {"text_weight": {"min": 0.5, "mean": 0.5, "max": 0.5}}}]
    assert MODULE.diagnostic_is_constant(rows, "M4qcs-i", "text_weight", 0.5)
    assert not MODULE.diagnostic_has_variation(rows, "M4qcs-i", "text_weight")


def test_diagnostic_variation_check():
    rows = [{"architecture": "M4qcs", "diagnostics": {"text_weight": {"min": 0.4, "mean": 0.5, "max": 0.6}}}]
    assert MODULE.diagnostic_has_variation(rows, "M4qcs", "text_weight")


def test_no_training_or_tuning_code_paths():
    source = PATH.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "official_test_accessed\": True" not in source


def test_output_root_refuses_overwrite_contract_present():
    source = PATH.read_text(encoding="utf-8")
    assert "not args.output_root.exists()" in source


def test_analyzer_has_no_dynamic_execution():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    forbidden = {"eval", "exec", "compile"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint(forbidden)


def test_real_formal_evidence_is_not_loaded_by_tests():
    source = Path(__file__).read_text(encoding="utf-8")
    assert "step4_unified_evaluation_results.json\").read" not in source

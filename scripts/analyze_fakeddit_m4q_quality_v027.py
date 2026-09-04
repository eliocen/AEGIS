"""
AEGIS v0.27 - Step 10B Formal M4q Quality Analysis

Purpose
-------
Implement the frozen Step 10A operationalization for H1-Q, H2-Q, H3-Q,
and H6-CLS using only existing Step 8/Step 9 validation artifacts.

This analyzer:
- does not train or load models;
- does not access the official Fakeddit test split;
- validates the expected three seeds and 57 Step 9 condition summaries;
- computes the primary hypothesis decisions exactly as frozen in Step 10A;
- writes machine-readable and human-readable analysis artifacts.

It MUST NOT be used to change the Step 10A denominators, thresholds, or
aggregation rules after inspecting results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


VERSION = "0.27.5-dev"
EXPERIMENT_NAME = "fakeddit_m4q_step10_quality_analysis_v027"

DEFAULT_PROTOCOL_PATH = Path(
    "docs/experiments/v027/step10_analysis_protocol.json"
)
DEFAULT_STEP9_PATH = Path(
    "experiments/fakeddit/v027_m4q_quality_diagnostics/"
    "aggregate/all_condition_summaries.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v027_m4q_step10_analysis"
)

EXPECTED_PROTOCOL_VERSION = "0.27.0-step10a"
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_CONDITIONS_PER_SEED = 19
EXPECTED_TOTAL_CONDITIONS = 57
CONTINUOUS_FAMILIES = ("gaussian_noise", "attenuation")
MODALITIES = ("text", "vision")
NONZERO_SEVERITIES = (0.25, 0.5, 0.75, 1.0)


# ============================================================================
# CLI / I/O
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.27 Step 10B formal M4q quality analysis under the "
            "frozen Step 10A operationalization."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Frozen Step 10A protocol JSON.",
    )
    parser.add_argument(
        "--step9",
        type=Path,
        default=DEFAULT_STEP9_PATH,
        help="Step 9 aggregate all_condition_summaries.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for Step 10B analysis outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON input not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ============================================================================
# Generic validation / statistics
# ============================================================================


def _require_finite(value: Any, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} must be finite; found {value!r}")
    return numeric


def _canonical_severity(value: Any) -> Optional[float]:
    if value is None:
        return None
    numeric = _require_finite(value, "severity")
    for allowed in NONZERO_SEVERITIES:
        if math.isclose(numeric, allowed, rel_tol=0.0, abs_tol=1e-12):
            return float(allowed)
    if math.isclose(numeric, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    return numeric


def condition_key(row: Mapping[str, Any]) -> Tuple[str, str, Optional[float]]:
    return (
        str(row["corruption_type"]),
        str(row["corrupted_modality"]),
        _canonical_severity(row.get("severity")),
    )


def expected_condition_keys() -> set[Tuple[str, str, Optional[float]]]:
    keys: set[Tuple[str, str, Optional[float]]] = {
        ("clean", "none", None)
    }
    for family in CONTINUOUS_FAMILIES:
        for modality in MODALITIES:
            for severity in NONZERO_SEVERITIES:
                keys.add((family, modality, float(severity)))
    for modality in MODALITIES:
        keys.add(("zero_dropout", modality, 1.0))
    return keys


def rankdata_average(values: Sequence[float]) -> List[float]:
    """
    Average-rank implementation equivalent to standard Spearman tie handling.

    Ranks are one-based. Equal values receive the arithmetic mean of the ranks
    they occupy.
    """
    if not values:
        raise ValueError("rankdata_average requires at least one value.")

    indexed = sorted(enumerate(float(v) for v in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)

    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            original_index = indexed[k][0]
            ranks[original_index] = average_rank
        i = j

    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Pearson inputs must have equal length.")
    if len(x) < 2:
        raise ValueError("Pearson correlation requires at least two values.")

    x_values = [float(v) for v in x]
    y_values = [float(v) for v in y]
    x_mean = mean(x_values)
    y_mean = mean(y_values)

    x_centered = [v - x_mean for v in x_values]
    y_centered = [v - y_mean for v in y_values]

    numerator = sum(a * b for a, b in zip(x_centered, y_centered))
    x_ss = sum(a * a for a in x_centered)
    y_ss = sum(b * b for b in y_centered)

    denominator = math.sqrt(x_ss * y_ss)
    if denominator == 0.0:
        # A constant predicted-quality curve has no monotonic ranking signal.
        # For the frozen H1-Q decision, it must not count as negative. Returning
        # zero gives the intended decision behavior while remaining explicit.
        return 0.0

    return float(numerator / denominator)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_correlation(rankdata_average(x), rankdata_average(y))


# ============================================================================
# Frozen Step 10A protocol validation
# ============================================================================


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if not isinstance(protocol, Mapping):
        raise RuntimeError("Step 10A protocol must be a JSON object.")

    meta = protocol.get("protocol")
    if not isinstance(meta, Mapping):
        raise RuntimeError("Protocol metadata section is missing.")

    if meta.get("version") != EXPECTED_PROTOCOL_VERSION:
        raise RuntimeError(
            "Unexpected Step 10A protocol version: "
            f"{meta.get('version')!r} != {EXPECTED_PROTOCOL_VERSION!r}"
        )

    if meta.get("status") != "FROZEN_BEFORE_FORMAL_STEP10_HYPOTHESIS_COMPUTATION":
        raise RuntimeError("Step 10A protocol status is not frozen.")

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("Protocol inputs section is missing.")

    if list(inputs.get("model_seeds", [])) != list(EXPECTED_SEEDS):
        raise RuntimeError("Protocol model seeds differ from frozen 42/43/44.")
    if int(inputs.get("conditions_per_seed", -1)) != EXPECTED_CONDITIONS_PER_SEED:
        raise RuntimeError("Protocol conditions_per_seed is not 19.")
    if int(inputs.get("total_condition_evaluations", -1)) != EXPECTED_TOTAL_CONDITIONS:
        raise RuntimeError("Protocol total_condition_evaluations is not 57.")
    if bool(inputs.get("official_test_access_permitted", True)):
        raise RuntimeError("Protocol unexpectedly permits official test access.")
    if bool(inputs.get("training_permitted", True)):
        raise RuntimeError("Protocol unexpectedly permits training.")

    h1 = protocol.get("h1_q")
    h2 = protocol.get("h2_q")
    h3 = protocol.get("h3_q")
    h6 = protocol.get("h6_cls")
    if not all(isinstance(section, Mapping) for section in (h1, h2, h3, h6)):
        raise RuntimeError("One or more frozen hypothesis sections are missing.")

    # H1-Q
    if int(h1.get("curve_count", -1)) != 12:
        raise RuntimeError("Frozen H1-Q curve_count must be 12.")
    if int(h1.get("minimum_negative_curve_count", -1)) != 9:
        raise RuntimeError("Frozen H1-Q minimum negative curves must be 9.")
    if not math.isclose(
        float(h1.get("minimum_negative_curve_proportion")),
        0.75,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen H1-Q proportion must be 0.75.")
    if list(h1.get("continuous_corruption_families", [])) != list(CONTINUOUS_FAMILIES):
        raise RuntimeError("Frozen H1-Q corruption families changed.")
    if list(h1.get("corrupted_modalities", [])) != list(MODALITIES):
        raise RuntimeError("Frozen H1-Q modalities changed.")
    if [float(v) for v in h1.get("severities", [])] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("Frozen H1-Q severities changed.")

    # H2-Q
    if int(h2.get("denominator", -1)) != 54:
        raise RuntimeError("Frozen H2-Q denominator must be 54.")
    if int(h2.get("minimum_positive_count", -1)) != 41:
        raise RuntimeError("Frozen H2-Q minimum positive count must be 41.")
    if not math.isclose(
        float(h2.get("minimum_positive_proportion")),
        0.75,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen H2-Q proportion must be 0.75.")
    if h2.get("zero_dropout_included") is not True:
        raise RuntimeError("Frozen H2-Q must include zero dropout.")

    # H3-Q
    if int(h3.get("condition_count", -1)) != 6:
        raise RuntimeError("Frozen H3-Q condition_count must be 6.")
    if not math.isclose(
        float(h3.get("maximum_mean_zeroed_quality")),
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen H3-Q maximum mean zeroed quality must be 0.2.")
    if not math.isclose(
        float(h3.get("minimum_mean_intact_quality")),
        0.8,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen H3-Q minimum mean intact quality must be 0.8.")
    if h3.get("require_every_individual_condition_to_pass") is not False:
        raise RuntimeError("Frozen H3-Q must use aggregate, not per-row, passing.")

    # H6-CLS
    if not math.isclose(
        float(h6.get("minimum_delta")),
        -0.01,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen H6-CLS minimum delta must be -0.01.")
    if h6.get("formal_noninferiority_test") is not False:
        raise RuntimeError("H6-CLS must not be labeled formal noninferiority.")


# ============================================================================
# Step 9 artifact validation
# ============================================================================


def extract_step9_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Step 9 aggregate payload must be a JSON object.")

    if payload.get("official_test_split_accessed") is not False:
        raise RuntimeError("Step 9 aggregate does not confirm sealed official test.")

    if int(payload.get("number_of_condition_evaluations", -1)) != EXPECTED_TOTAL_CONDITIONS:
        raise RuntimeError("Step 9 aggregate does not contain 57 evaluations.")

    if list(payload.get("model_seeds", [])) != list(EXPECTED_SEEDS):
        raise RuntimeError("Step 9 aggregate model seeds differ from 42/43/44.")

    if int(payload.get("conditions_per_checkpoint", -1)) != EXPECTED_CONDITIONS_PER_SEED:
        raise RuntimeError("Step 9 aggregate conditions_per_checkpoint is not 19.")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Step 9 aggregate rows are missing.")
    return [dict(row) for row in rows]


def validate_step9_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[Tuple[str, str, Optional[float]], Dict[str, Any]]]:
    if len(rows) != EXPECTED_TOTAL_CONDITIONS:
        raise RuntimeError(
            f"Expected 57 Step 9 rows, found {len(rows)}."
        )

    by_seed: Dict[int, Dict[Tuple[str, str, Optional[float]], Dict[str, Any]]] = {
        seed: {} for seed in EXPECTED_SEEDS
    }
    expected_keys = expected_condition_keys()

    for index, raw in enumerate(rows):
        row = dict(raw)
        seed = int(row.get("model_seed", -1))
        if seed not in by_seed:
            raise RuntimeError(f"Unexpected model seed at row {index}: {seed}")

        if row.get("model_variant") != "M4q":
            raise RuntimeError(f"Unexpected model_variant at row {index}.")
        if row.get("fusion_architecture") != "quality_supervised":
            raise RuntimeError(f"Unexpected fusion_architecture at row {index}.")
        if row.get("test_split_accessed") is not False:
            raise RuntimeError(f"Row {index} does not confirm sealed test split.")
        if row.get("parameters_unchanged") is not True:
            raise RuntimeError(f"Row {index} does not confirm unchanged parameters.")

        text_q = _require_finite(row.get("text_quality_mean"), f"row {index} text_quality_mean")
        vision_q = _require_finite(row.get("vision_quality_mean"), f"row {index} vision_quality_mean")
        if not 0.0 <= text_q <= 1.0:
            raise RuntimeError(f"Row {index} text quality outside [0,1].")
        if not 0.0 <= vision_q <= 1.0:
            raise RuntimeError(f"Row {index} vision quality outside [0,1].")

        if int(row.get("sample_count", -1)) != 1000:
            raise RuntimeError(f"Row {index} sample_count is not 1000.")

        key = condition_key(row)
        if key not in expected_keys:
            raise RuntimeError(f"Unexpected Step 9 condition at row {index}: {key}")
        if key in by_seed[seed]:
            raise RuntimeError(f"Duplicate condition for seed {seed}: {key}")

        by_seed[seed][key] = row

    for seed in EXPECTED_SEEDS:
        keys = set(by_seed[seed].keys())
        missing = expected_keys - keys
        extra = keys - expected_keys
        if missing or extra:
            raise RuntimeError(
                f"Condition-set mismatch for seed {seed}; "
                f"missing={sorted(missing, key=str)}, extra={sorted(extra, key=str)}"
            )
        if len(by_seed[seed]) != EXPECTED_CONDITIONS_PER_SEED:
            raise RuntimeError(f"Seed {seed} does not have exactly 19 conditions.")

    return by_seed


def clean_row(
    by_seed: Mapping[int, Mapping[Tuple[str, str, Optional[float]], Mapping[str, Any]]],
    seed: int,
) -> Mapping[str, Any]:
    return by_seed[seed][("clean", "none", None)]


def quality_for_modality(row: Mapping[str, Any], modality: str) -> float:
    if modality == "text":
        return _require_finite(row["text_quality_mean"], "text quality")
    if modality == "vision":
        return _require_finite(row["vision_quality_mean"], "vision quality")
    raise ValueError(f"Unsupported modality: {modality!r}")


# ============================================================================
# H1-Q
# ============================================================================


def analyze_h1_q(
    by_seed: Mapping[int, Mapping[Tuple[str, str, Optional[float]], Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]

    for seed in EXPECTED_SEEDS:
        clean = clean_row(by_seed, seed)
        for family in CONTINUOUS_FAMILIES:
            for modality in MODALITIES:
                qualities = [quality_for_modality(clean, modality)]
                for severity in NONZERO_SEVERITIES:
                    row = by_seed[seed][(family, modality, float(severity))]
                    qualities.append(quality_for_modality(row, modality))

                rho = spearman_correlation(severities, qualities)
                rows.append(
                    {
                        "model_seed": seed,
                        "corruption_family": family,
                        "corrupted_modality": modality,
                        "severity_0_quality": qualities[0],
                        "severity_0p25_quality": qualities[1],
                        "severity_0p5_quality": qualities[2],
                        "severity_0p75_quality": qualities[3],
                        "severity_1_quality": qualities[4],
                        "spearman_rho": rho,
                        "expected_negative": rho < 0.0,
                    }
                )

    if len(rows) != 12:
        raise RuntimeError(f"H1-Q expected 12 curves, found {len(rows)}.")

    negative_count = sum(bool(row["expected_negative"]) for row in rows)
    negative_proportion = negative_count / len(rows)
    mean_rho = mean(float(row["spearman_rho"]) for row in rows)

    frozen = protocol["h1_q"]
    decision = (
        negative_count >= int(frozen["minimum_negative_curve_count"])
        and mean_rho < 0.0
    )

    family_summary = {}
    for family in CONTINUOUS_FAMILIES:
        subset = [row for row in rows if row["corruption_family"] == family]
        family_summary[family] = {
            "curve_count": len(subset),
            "negative_curve_count": sum(bool(row["expected_negative"]) for row in subset),
            "negative_curve_proportion": (
                sum(bool(row["expected_negative"]) for row in subset) / len(subset)
            ),
            "mean_spearman_rho": mean(float(row["spearman_rho"]) for row in subset),
        }

    modality_summary = {}
    for modality in MODALITIES:
        subset = [row for row in rows if row["corrupted_modality"] == modality]
        modality_summary[modality] = {
            "curve_count": len(subset),
            "negative_curve_count": sum(bool(row["expected_negative"]) for row in subset),
            "negative_curve_proportion": (
                sum(bool(row["expected_negative"]) for row in subset) / len(subset)
            ),
            "mean_spearman_rho": mean(float(row["spearman_rho"]) for row in subset),
        }

    return {
        "hypothesis": "H1-Q",
        "curve_count": len(rows),
        "negative_curve_count": negative_count,
        "negative_curve_proportion": negative_proportion,
        "mean_spearman_rho": mean_rho,
        "minimum_negative_curve_count": int(frozen["minimum_negative_curve_count"]),
        "minimum_negative_curve_proportion": float(frozen["minimum_negative_curve_proportion"]),
        "mean_rho_requirement": "mean_rho < 0",
        "supported": bool(decision),
        "curves": rows,
        "family_stratification": family_summary,
        "modality_stratification": modality_summary,
    }


# ============================================================================
# H2-Q
# ============================================================================


def analyze_h2_q(
    by_seed: Mapping[int, Mapping[Tuple[str, str, Optional[float]], Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for seed in EXPECTED_SEEDS:
        clean = clean_row(by_seed, seed)
        clean_text = quality_for_modality(clean, "text")
        clean_vision = quality_for_modality(clean, "vision")

        condition_specs: List[Tuple[str, str, float]] = []
        for family in CONTINUOUS_FAMILIES:
            for modality in MODALITIES:
                for severity in NONZERO_SEVERITIES:
                    condition_specs.append((family, modality, float(severity)))
        for modality in MODALITIES:
            condition_specs.append(("zero_dropout", modality, 1.0))

        for family, modality, severity in condition_specs:
            row = by_seed[seed][(family, modality, severity)]
            current_text = quality_for_modality(row, "text")
            current_vision = quality_for_modality(row, "vision")

            delta_text = current_text - clean_text
            delta_vision = current_vision - clean_vision

            if modality == "text":
                corrupted_quality = current_text
                intact_quality = current_vision
                corrupted_delta = delta_text
                intact_delta = delta_vision
            else:
                corrupted_quality = current_vision
                intact_quality = current_text
                corrupted_delta = delta_vision
                intact_delta = delta_text

            selectivity = intact_delta - corrupted_delta

            rows.append(
                {
                    "model_seed": seed,
                    "corruption_family": family,
                    "corrupted_modality": modality,
                    "severity": severity,
                    "corrupted_modality_quality": corrupted_quality,
                    "intact_modality_quality": intact_quality,
                    "corrupted_modality_delta_from_clean": corrupted_delta,
                    "intact_modality_delta_from_clean": intact_delta,
                    "quality_selectivity": selectivity,
                    "positive_selectivity": selectivity > 0.0,
                }
            )

    if len(rows) != 54:
        raise RuntimeError(f"H2-Q expected 54 conditions, found {len(rows)}.")

    positive_count = sum(bool(row["positive_selectivity"]) for row in rows)
    proportion = positive_count / len(rows)
    selectivities = [float(row["quality_selectivity"]) for row in rows]

    frozen = protocol["h2_q"]
    decision = positive_count >= int(frozen["minimum_positive_count"])

    family_summary = {}
    for family in ("gaussian_noise", "attenuation", "zero_dropout"):
        subset = [row for row in rows if row["corruption_family"] == family]
        positives = sum(bool(row["positive_selectivity"]) for row in subset)
        family_summary[family] = {
            "condition_count": len(subset),
            "positive_selectivity_count": positives,
            "positive_selectivity_proportion": positives / len(subset),
            "mean_selectivity": mean(float(row["quality_selectivity"]) for row in subset),
            "median_selectivity": median(float(row["quality_selectivity"]) for row in subset),
        }

    modality_summary = {}
    for modality in MODALITIES:
        subset = [row for row in rows if row["corrupted_modality"] == modality]
        positives = sum(bool(row["positive_selectivity"]) for row in subset)
        modality_summary[modality] = {
            "condition_count": len(subset),
            "positive_selectivity_count": positives,
            "positive_selectivity_proportion": positives / len(subset),
            "mean_selectivity": mean(float(row["quality_selectivity"]) for row in subset),
            "median_selectivity": median(float(row["quality_selectivity"]) for row in subset),
        }

    return {
        "hypothesis": "H2-Q",
        "denominator": len(rows),
        "positive_selectivity_count": positive_count,
        "positive_selectivity_proportion": proportion,
        "mean_selectivity": mean(selectivities),
        "median_selectivity": median(selectivities),
        "minimum_positive_count": int(frozen["minimum_positive_count"]),
        "minimum_positive_proportion": float(frozen["minimum_positive_proportion"]),
        "supported": bool(decision),
        "conditions": rows,
        "family_stratification": family_summary,
        "modality_stratification": modality_summary,
    }


# ============================================================================
# H3-Q
# ============================================================================


def analyze_h3_q(
    by_seed: Mapping[int, Mapping[Tuple[str, str, Optional[float]], Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for seed in EXPECTED_SEEDS:
        for modality in MODALITIES:
            row = by_seed[seed][("zero_dropout", modality, 1.0)]
            text_q = quality_for_modality(row, "text")
            vision_q = quality_for_modality(row, "vision")

            if modality == "text":
                zeroed_q = text_q
                intact_q = vision_q
            else:
                zeroed_q = vision_q
                intact_q = text_q

            rows.append(
                {
                    "model_seed": seed,
                    "zeroed_modality": modality,
                    "zeroed_modality_quality": zeroed_q,
                    "intact_modality_quality": intact_q,
                }
            )

    if len(rows) != 6:
        raise RuntimeError(f"H3-Q expected 6 zero-dropout conditions, found {len(rows)}.")

    mean_zeroed = mean(float(row["zeroed_modality_quality"]) for row in rows)
    mean_intact = mean(float(row["intact_modality_quality"]) for row in rows)

    frozen = protocol["h3_q"]
    max_zeroed = float(frozen["maximum_mean_zeroed_quality"])
    min_intact = float(frozen["minimum_mean_intact_quality"])
    decision = mean_zeroed <= max_zeroed and mean_intact >= min_intact

    modality_summary = {}
    for modality in MODALITIES:
        subset = [row for row in rows if row["zeroed_modality"] == modality]
        modality_summary[modality] = {
            "condition_count": len(subset),
            "mean_zeroed_quality": mean(float(row["zeroed_modality_quality"]) for row in subset),
            "mean_intact_quality": mean(float(row["intact_modality_quality"]) for row in subset),
        }

    return {
        "hypothesis": "H3-Q",
        "condition_count": len(rows),
        "aggregate_mean_zeroed_quality": mean_zeroed,
        "aggregate_mean_intact_quality": mean_intact,
        "maximum_mean_zeroed_quality": max_zeroed,
        "minimum_mean_intact_quality": min_intact,
        "supported": bool(decision),
        "conditions": rows,
        "modality_stratification": modality_summary,
    }


# ============================================================================
# H6-CLS
# ============================================================================


def analyze_h6_cls(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    frozen = protocol["h6_cls"]
    seed_values = {
        str(key): float(value)
        for key, value in dict(frozen["m4q_seed_values"]).items()
    }
    computed_m4q_mean = mean(seed_values.values())
    recorded_m4q_mean = float(frozen["m4q_mean"])

    if not math.isclose(
        computed_m4q_mean,
        recorded_m4q_mean,
        rel_tol=0.0,
        abs_tol=5e-10,
    ):
        raise RuntimeError(
            "Frozen H6-CLS M4q mean is inconsistent with seed values: "
            f"{computed_m4q_mean} vs {recorded_m4q_mean}"
        )

    reference = float(frozen["m1b_reference_mean"])
    computed_delta = recorded_m4q_mean - reference
    recorded_delta = float(frozen["delta"])
    if not math.isclose(
        computed_delta,
        recorded_delta,
        rel_tol=0.0,
        abs_tol=5e-10,
    ):
        raise RuntimeError(
            "Frozen H6-CLS delta is inconsistent with frozen means: "
            f"{computed_delta} vs {recorded_delta}"
        )

    minimum_delta = float(frozen["minimum_delta"])
    decision = recorded_delta >= minimum_delta

    return {
        "hypothesis": "H6-CLS",
        "metric": str(frozen["metric"]),
        "m4q_seed_values": seed_values,
        "m4q_mean": recorded_m4q_mean,
        "m1b_reference_mean": reference,
        "delta": recorded_delta,
        "minimum_delta": minimum_delta,
        "supported": bool(decision),
        "interpretation": "descriptive_preregistered_validation_margin",
        "formal_noninferiority_test": False,
        "official_test_result": False,
    }


# ============================================================================
# Reporting
# ============================================================================


def make_summary_markdown(results: Mapping[str, Any]) -> str:
    h1 = results["h1_q"]
    h2 = results["h2_q"]
    h3 = results["h3_q"]
    h6 = results["h6_cls"]

    def status(value: bool) -> str:
        return "SUPPORTED" if value else "NOT SUPPORTED"

    return f"""# AEGIS v0.27 Step 10B M4q Quality Analysis

**Analyzer version:** {VERSION}  
**Protocol version:** {results['protocol_version']}  
**Scientific status:** validation-only formal analysis under frozen Step 10A operationalization  
**Official Fakeddit test samples accessed:** 0

## Primary decisions

| Hypothesis | Result | Frozen criterion | Decision |
| --- | --- | --- | --- |
| H1-Q | {h1['negative_curve_count']}/{h1['curve_count']} negative curves; mean rho = {h1['mean_spearman_rho']:.6f} | >= 9/12 negative AND mean rho < 0 | {status(h1['supported'])} |
| H2-Q | {h2['positive_selectivity_count']}/{h2['denominator']} positive; proportion = {h2['positive_selectivity_proportion']:.6f} | >= 41/54 positive | {status(h2['supported'])} |
| H3-Q | mean zeroed q = {h3['aggregate_mean_zeroed_quality']:.6f}; mean intact q = {h3['aggregate_mean_intact_quality']:.6f} | zeroed <= 0.20 AND intact >= 0.80 | {status(h3['supported'])} |
| H6-CLS | Delta Macro-F1 = {h6['delta']:.9f} | Delta >= -0.01 | {status(h6['supported'])} |

## Interpretation boundary

These results concern explicitly supervised intrinsic modality-quality
estimation under the frozen controlled representation-corruption validation
protocol. They do not establish factual verification, source credibility,
human-calibrated trustworthiness, cross-modal compatibility discrimination,
or raw-world corruption generalization.

H6-CLS is a descriptive validation-margin comparison, not a formal
statistical noninferiority test.
"""


def analyze(
    *,
    protocol: Mapping[str, Any],
    step9_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_protocol(protocol)
    rows = extract_step9_rows(step9_payload)
    by_seed = validate_step9_rows(rows)

    h1 = analyze_h1_q(by_seed, protocol)
    h2 = analyze_h2_q(by_seed, protocol)
    h3 = analyze_h3_q(by_seed, protocol)
    h6 = analyze_h6_cls(protocol)

    return {
        "aegis_version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "scientific_status": (
            "validation-only formal analysis under frozen Step 10A operationalization"
        ),
        "step9_condition_evaluations_validated": EXPECTED_TOTAL_CONDITIONS,
        "model_seeds_validated": list(EXPECTED_SEEDS),
        "official_test_samples_accessed": 0,
        "training_performed": False,
        "thresholds_modified": False,
        "h1_q": h1,
        "h2_q": h2,
        "h3_q": h3,
        "h6_cls": h6,
    }


def main() -> None:
    args = parse_args()

    protocol = load_json(args.protocol)
    step9_payload = load_json(args.step9)

    results = analyze(
        protocol=protocol,
        step9_payload=step9_payload,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    save_json(args.output_root / "step10_results.json", results)
    save_csv(args.output_root / "h1_curves.csv", results["h1_q"]["curves"])
    save_csv(args.output_root / "h2_conditions.csv", results["h2_q"]["conditions"])
    save_csv(args.output_root / "h3_zero_dropout.csv", results["h3_q"]["conditions"])
    save_text(
        args.output_root / "summary.md",
        make_summary_markdown(results),
    )

    run_manifest = {
        "aegis_version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "protocol_path": str(args.protocol),
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "step9_input": str(args.step9),
        "output_root": str(args.output_root),
        "model_seeds": list(EXPECTED_SEEDS),
        "step9_condition_evaluations_validated": EXPECTED_TOTAL_CONDITIONS,
        "official_test_samples_accessed": 0,
        "training_performed": False,
        "hypotheses_evaluated": ["H1-Q", "H2-Q", "H3-Q", "H6-CLS"],
        "primary_decisions": {
            "H1-Q": bool(results["h1_q"]["supported"]),
            "H2-Q": bool(results["h2_q"]["supported"]),
            "H3-Q": bool(results["h3_q"]["supported"]),
            "H6-CLS": bool(results["h6_cls"]["supported"]),
        },
    }
    save_json(args.output_root / "run_manifest.json", run_manifest)

    print("=" * 78)
    print("AEGIS v0.27 STEP 10B - FORMAL M4q QUALITY ANALYSIS")
    print("=" * 78)
    print("Protocol:", args.protocol)
    print("Step 9 input:", args.step9)
    print("Validated Step 9 condition evaluations:", EXPECTED_TOTAL_CONDITIONS)
    print("Model seeds:", list(EXPECTED_SEEDS))
    print("Official test samples accessed: 0")
    print("Training performed: NO")
    print()
    print(
        "H1-Q:",
        "SUPPORTED" if results["h1_q"]["supported"] else "NOT SUPPORTED",
        f"({results['h1_q']['negative_curve_count']}/12 negative curves, "
        f"mean rho={results['h1_q']['mean_spearman_rho']:.6f})",
    )
    print(
        "H2-Q:",
        "SUPPORTED" if results["h2_q"]["supported"] else "NOT SUPPORTED",
        f"({results['h2_q']['positive_selectivity_count']}/54 positive, "
        f"p={results['h2_q']['positive_selectivity_proportion']:.6f})",
    )
    print(
        "H3-Q:",
        "SUPPORTED" if results["h3_q"]["supported"] else "NOT SUPPORTED",
        f"(mean zeroed q={results['h3_q']['aggregate_mean_zeroed_quality']:.6f}, "
        f"mean intact q={results['h3_q']['aggregate_mean_intact_quality']:.6f})",
    )
    print(
        "H6-CLS:",
        "SUPPORTED" if results["h6_cls"]["supported"] else "NOT SUPPORTED",
        f"(Delta Macro-F1={results['h6_cls']['delta']:.9f})",
    )
    print()
    print("Outputs:", args.output_root)
    print("=" * 78)


if __name__ == "__main__":
    main()

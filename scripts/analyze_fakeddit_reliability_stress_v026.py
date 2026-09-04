"""
AEGIS v0.26.1 — Fakeddit Reliability Stress-Test Analysis

Purpose
-------
Analyse the frozen v0.26 reliability stress-test outputs and convert the
189 condition-level evaluations into quantitative evidence for:

H1 Sensitivity:
    Increasing corruption of modality m should reduce its learned
    reliability/weight.

H2 Selectivity:
    Reliability reduction should be larger for the corrupted modality than
    for the intact modality, and fusion weight should shift toward the intact
    modality.

H3 Utility:
    Architectures whose reliability reacts appropriately should exhibit
    smaller classification degradation under corrupted evidence.

This script DOES NOT load model checkpoints, perform inference, retrain any
model, or access the official Fakeddit test split. It only analyses the
condition-summary CSV produced by:

    scripts/run_fakeddit_reliability_stress_test.py

Default input
-------------
experiments/fakeddit/reliability_stress_v026_full/aggregate/
    all_condition_summaries.csv

Default output
--------------
experiments/fakeddit/reliability_stress_v026_full/analysis_v026/

Scientific interpretation
-------------------------
The three model seeds evaluate the SAME fixed validation examples. Seed runs
are repeated trained-model evaluations, not independent dataset samples.
All statistics produced here are therefore validation diagnostics. They must
not be presented as official test-set generalization estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VERSION = "0.26.1"

DEFAULT_INPUT = Path(
    "experiments/fakeddit/reliability_stress_v026_full/"
    "aggregate/all_condition_summaries.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/fakeddit/reliability_stress_v026_full/"
    "analysis_v026"
)

EXPECTED_ARCHITECTURES = ("M1b", "M2b", "M3")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_ROWS = 189
EXPECTED_ROWS_PER_MODEL_SEED = 21

CONTINUOUS_CORRUPTIONS = ("gaussian_noise", "attenuation")
DISCRETE_CORRUPTIONS = ("permutation", "zero_dropout")
MODALITIES = ("text", "vision")

METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "mean_cosine_similarity",
    "mean_text_reliability",
    "mean_vision_reliability",
    "mean_text_weight",
    "mean_vision_weight",
    "delta_accuracy_from_clean",
    "delta_macro_f1_from_clean",
    "delta_mean_cosine_similarity_from_clean",
    "delta_mean_text_reliability_from_clean",
    "delta_mean_vision_reliability_from_clean",
    "delta_mean_text_weight_from_clean",
    "delta_mean_vision_weight_from_clean",
    "reliability_selectivity",
)


# =====================================================================
# CLI / IO
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.26 reliability stress-test quantitative analysis."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="all_condition_summaries.csv from the frozen v0.26 run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for quantitative analysis outputs.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=EXPECTED_ROWS,
        help="Hard expected condition-row count. Default: 189.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_int(value: Any) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: Dict[str, Any] = dict(raw)

            row["model_seed"] = to_int(row.get("model_seed"))
            row["severity"] = to_float(row.get("severity"))

            for field in METRIC_FIELDS:
                row[field] = to_float(row.get(field))

            if "sample_count" in row:
                row["sample_count"] = to_int(row.get("sample_count"))

            if "test_split_accessed" in row:
                row["test_split_accessed"] = to_bool(
                    row.get("test_split_accessed")
                )

            rows.append(row)

    if not rows:
        raise RuntimeError("Input CSV contained zero rows.")
    return rows


# =====================================================================
# Statistics
# =====================================================================

def population_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    clean = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std_population": None,
            "std_sample": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(clean),
        "mean": mean(clean),
        "std_population": pstdev(clean) if len(clean) > 1 else 0.0,
        "std_sample": stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def rankdata(values: Sequence[float]) -> List[float]:
    """
    Average ranks for ties. Ranks are 1-based.
    """
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i + 1
        while (
            j < len(indexed)
            and indexed[j][1] == indexed[i][1]
        ):
            j += 1

        average_rank = (
            (i + 1) + j
        ) / 2.0

        for k in range(i, j):
            original_index = indexed[k][0]
            ranks[original_index] = average_rank
        i = j

    return ranks


def pearson(
    first: Sequence[Optional[float]],
    second: Sequence[Optional[float]],
) -> Optional[float]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(first, second)
        if (
            a is not None
            and b is not None
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        )
    ]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)

    numerator = sum(
        (x - mx) * (y - my)
        for x, y in pairs
    )
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    denominator = math.sqrt(dx * dy)

    if denominator == 0.0:
        return None
    return numerator / denominator


def spearman(
    first: Sequence[Optional[float]],
    second: Sequence[Optional[float]],
) -> Optional[float]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(first, second)
        if (
            a is not None
            and b is not None
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        )
    ]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return pearson(rankdata(xs), rankdata(ys))


def trapezoid_auc(
    xs: Sequence[float],
    ys: Sequence[float],
) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    area = 0.0
    for (x1, y1), (x2, y2) in zip(pairs[:-1], pairs[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0
    return area


def sign(value: Optional[float], tolerance: float = 1e-12) -> Optional[int]:
    if value is None:
        return None
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


# =====================================================================
# Integrity checks
# =====================================================================

def validate_dataset(
    rows: Sequence[Dict[str, Any]],
    expected_rows: int,
) -> Dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} condition rows; found {len(rows)}."
        )

    architectures = sorted(
        {str(row["architecture"]) for row in rows}
    )
    expected_architectures = sorted(EXPECTED_ARCHITECTURES)
    if architectures != expected_architectures:
        raise RuntimeError(
            "Architecture set mismatch. Expected "
            f"{expected_architectures}, found {architectures}."
        )

    seeds = sorted(
        {
            int(row["model_seed"])
            for row in rows
            if row["model_seed"] is not None
        }
    )
    if seeds != list(EXPECTED_SEEDS):
        raise RuntimeError(
            f"Seed set mismatch. Expected {EXPECTED_SEEDS}, found {seeds}."
        )

    grouped: Dict[Tuple[str, int], int] = defaultdict(int)
    for row in rows:
        key = (
            str(row["architecture"]),
            int(row["model_seed"]),
        )
        grouped[key] += 1

        if row.get("test_split_accessed") is True:
            raise RuntimeError(
                "Input reports official test split access for "
                f"{key}; refusing analysis."
            )

    for key, count in grouped.items():
        if count != EXPECTED_ROWS_PER_MODEL_SEED:
            raise RuntimeError(
                f"{key} has {count} conditions; expected "
                f"{EXPECTED_ROWS_PER_MODEL_SEED}."
            )

    clean_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for row in rows:
        if row["corruption_type"] == "clean":
            clean_counts[
                (
                    str(row["architecture"]),
                    int(row["model_seed"]),
                )
            ] += 1

    for key in grouped:
        if clean_counts[key] != 1:
            raise RuntimeError(
                f"{key} has {clean_counts[key]} clean rows; expected 1."
            )

    return {
        "row_count": len(rows),
        "architectures": architectures,
        "seeds": seeds,
        "rows_per_architecture_seed": {
            f"{key[0]}_seed{key[1]}": count
            for key, count in sorted(grouped.items())
        },
        "official_test_split_accessed": False,
        "integrity_checks_passed": True,
    }


def build_clean_index(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if row["corruption_type"] != "clean":
            continue
        key = (
            str(row["architecture"]),
            int(row["model_seed"]),
        )
        result[key] = row
    return result


# =====================================================================
# H1 — Sensitivity
# =====================================================================

def corrupted_reliability_field(modality: str) -> str:
    return (
        "mean_text_reliability"
        if modality == "text"
        else "mean_vision_reliability"
    )


def intact_reliability_field(modality: str) -> str:
    return (
        "mean_vision_reliability"
        if modality == "text"
        else "mean_text_reliability"
    )


def corrupted_weight_field(modality: str) -> str:
    return (
        "mean_text_weight"
        if modality == "text"
        else "mean_vision_weight"
    )


def intact_weight_field(modality: str) -> str:
    return (
        "mean_vision_weight"
        if modality == "text"
        else "mean_text_weight"
    )


def continuous_series_with_clean(
    rows: Sequence[Dict[str, Any]],
    clean_index: Dict[Tuple[str, int], Dict[str, Any]],
    architecture: str,
    seed: int,
    corruption_type: str,
    modality: str,
) -> List[Dict[str, Any]]:
    subset = [
        row
        for row in rows
        if (
            row["architecture"] == architecture
            and row["model_seed"] == seed
            and row["corruption_type"] == corruption_type
            and row["corrupted_modality"] == modality
        )
    ]

    clean = clean_index[(architecture, seed)]
    synthetic_clean = dict(clean)
    synthetic_clean["severity"] = 0.0
    synthetic_clean["corruption_type"] = corruption_type
    synthetic_clean["corrupted_modality"] = modality

    return sorted(
        [synthetic_clean] + subset,
        key=lambda row: float(row["severity"]),
    )


def analyse_h1(
    rows: Sequence[Dict[str, Any]],
    clean_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result_rows: List[Dict[str, Any]] = []

    for architecture in ("M2b", "M3"):
        for seed in EXPECTED_SEEDS:
            for corruption_type in CONTINUOUS_CORRUPTIONS:
                for modality in MODALITIES:
                    series = continuous_series_with_clean(
                        rows,
                        clean_index,
                        architecture,
                        seed,
                        corruption_type,
                        modality,
                    )

                    severities = [
                        float(row["severity"])
                        for row in series
                    ]
                    rel_field = corrupted_reliability_field(modality)
                    weight_field = corrupted_weight_field(modality)

                    reliabilities = [
                        row.get(rel_field)
                        for row in series
                    ]
                    weights = [
                        row.get(weight_field)
                        for row in series
                    ]
                    macro_f1s = [
                        row.get("macro_f1")
                        for row in series
                    ]

                    rho_rel = spearman(severities, reliabilities)
                    rho_weight = spearman(severities, weights)
                    rho_f1 = spearman(severities, macro_f1s)

                    result_rows.append(
                        {
                            "architecture": architecture,
                            "model_seed": seed,
                            "corruption_type": corruption_type,
                            "corrupted_modality": modality,
                            "n_severity_points": len(series),
                            "spearman_severity_vs_corrupted_reliability": (
                                rho_rel
                            ),
                            "h1_reliability_direction_supported": (
                                None
                                if rho_rel is None
                                else rho_rel < 0.0
                            ),
                            "spearman_severity_vs_corrupted_weight": (
                                rho_weight
                            ),
                            "weight_down_with_severity": (
                                None
                                if rho_weight is None
                                else rho_weight < 0.0
                            ),
                            "spearman_severity_vs_macro_f1": rho_f1,
                            "performance_degrades_with_severity": (
                                None
                                if rho_f1 is None
                                else rho_f1 < 0.0
                            ),
                            "clean_corrupted_reliability": reliabilities[0],
                            "max_severity_corrupted_reliability": (
                                reliabilities[-1]
                            ),
                            "delta_corrupted_reliability_at_max": (
                                None
                                if (
                                    reliabilities[0] is None
                                    or reliabilities[-1] is None
                                )
                                else (
                                    float(reliabilities[-1])
                                    - float(reliabilities[0])
                                )
                            ),
                            "clean_corrupted_weight": weights[0],
                            "max_severity_corrupted_weight": weights[-1],
                            "delta_corrupted_weight_at_max": (
                                None
                                if (
                                    weights[0] is None
                                    or weights[-1] is None
                                )
                                else (
                                    float(weights[-1])
                                    - float(weights[0])
                                )
                            ),
                            "clean_macro_f1": macro_f1s[0],
                            "max_severity_macro_f1": macro_f1s[-1],
                            "delta_macro_f1_at_max": (
                                None
                                if (
                                    macro_f1s[0] is None
                                    or macro_f1s[-1] is None
                                )
                                else (
                                    float(macro_f1s[-1])
                                    - float(macro_f1s[0])
                                )
                            ),
                        }
                    )

    valid_rel = [
        row
        for row in result_rows
        if row[
            "h1_reliability_direction_supported"
        ] is not None
    ]
    supported = sum(
        bool(row["h1_reliability_direction_supported"])
        for row in valid_rel
    )

    summary = {
        "hypothesis": (
            "H1: increasing corruption should reduce the corrupted "
            "modality's learned reliability/weight."
        ),
        "number_of_architecture_seed_family_modality_curves": (
            len(result_rows)
        ),
        "reliability_direction_supported_count": supported,
        "reliability_direction_evaluable_count": len(valid_rel),
        "reliability_direction_supported_rate": (
            supported / len(valid_rel)
            if valid_rel
            else None
        ),
        "spearman_reliability_distribution": population_stats(
            [
                row[
                    "spearman_severity_vs_corrupted_reliability"
                ]
                for row in result_rows
            ]
        ),
        "spearman_weight_distribution": population_stats(
            [
                row[
                    "spearman_severity_vs_corrupted_weight"
                ]
                for row in result_rows
            ]
        ),
        "spearman_macro_f1_distribution": population_stats(
            [
                row[
                    "spearman_severity_vs_macro_f1"
                ]
                for row in result_rows
            ]
        ),
    }
    return result_rows, summary


# =====================================================================
# H2 — Selectivity and direction of weight shift
# =====================================================================

def analyse_h2(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detail: List[Dict[str, Any]] = []

    for row in rows:
        if row["architecture"] not in ("M2b", "M3"):
            continue
        if row["corruption_type"] == "clean":
            continue

        modality = str(row["corrupted_modality"])
        if modality not in MODALITIES:
            continue

        if modality == "text":
            delta_corrupted_rel = row.get(
                "delta_mean_text_reliability_from_clean"
            )
            delta_intact_rel = row.get(
                "delta_mean_vision_reliability_from_clean"
            )
            delta_corrupted_weight = row.get(
                "delta_mean_text_weight_from_clean"
            )
            delta_intact_weight = row.get(
                "delta_mean_vision_weight_from_clean"
            )
        else:
            delta_corrupted_rel = row.get(
                "delta_mean_vision_reliability_from_clean"
            )
            delta_intact_rel = row.get(
                "delta_mean_text_reliability_from_clean"
            )
            delta_corrupted_weight = row.get(
                "delta_mean_vision_weight_from_clean"
            )
            delta_intact_weight = row.get(
                "delta_mean_text_weight_from_clean"
            )

        selectivity = (
            None
            if (
                delta_corrupted_rel is None
                or delta_intact_rel is None
            )
            else (
                float(delta_intact_rel)
                - float(delta_corrupted_rel)
            )
        )

        corrupted_weight_down = (
            None
            if delta_corrupted_weight is None
            else float(delta_corrupted_weight) < 0.0
        )
        intact_weight_up = (
            None
            if delta_intact_weight is None
            else float(delta_intact_weight) > 0.0
        )
        correct_weight_shift = (
            None
            if (
                corrupted_weight_down is None
                or intact_weight_up is None
            )
            else (
                corrupted_weight_down
                and intact_weight_up
            )
        )

        corrupted_rel_down = (
            None
            if delta_corrupted_rel is None
            else float(delta_corrupted_rel) < 0.0
        )
        selectivity_positive = (
            None
            if selectivity is None
            else selectivity > 0.0
        )

        detail.append(
            {
                "architecture": row["architecture"],
                "model_seed": row["model_seed"],
                "corruption_type": row["corruption_type"],
                "corrupted_modality": modality,
                "severity": row["severity"],
                "delta_macro_f1_from_clean": (
                    row["delta_macro_f1_from_clean"]
                ),
                "delta_corrupted_reliability": delta_corrupted_rel,
                "delta_intact_reliability": delta_intact_rel,
                "reliability_selectivity": selectivity,
                "corrupted_reliability_decreased": corrupted_rel_down,
                "selectivity_positive": selectivity_positive,
                "delta_corrupted_weight": delta_corrupted_weight,
                "delta_intact_weight": delta_intact_weight,
                "corrupted_weight_decreased": corrupted_weight_down,
                "intact_weight_increased": intact_weight_up,
                "correct_weight_shift": correct_weight_shift,
            }
        )

    summary_by_architecture: Dict[str, Any] = {}

    for architecture in ("M2b", "M3"):
        subset = [
            row
            for row in detail
            if row["architecture"] == architecture
        ]

        evaluable_selectivity = [
            row
            for row in subset
            if row["selectivity_positive"] is not None
        ]
        positive_selectivity = sum(
            bool(row["selectivity_positive"])
            for row in evaluable_selectivity
        )

        evaluable_weight = [
            row
            for row in subset
            if row["correct_weight_shift"] is not None
        ]
        correct_weight = sum(
            bool(row["correct_weight_shift"])
            for row in evaluable_weight
        )

        evaluable_rel_down = [
            row
            for row in subset
            if row["corrupted_reliability_decreased"] is not None
        ]
        rel_down = sum(
            bool(row["corrupted_reliability_decreased"])
            for row in evaluable_rel_down
        )

        summary_by_architecture[architecture] = {
            "condition_count": len(subset),
            "positive_selectivity_count": positive_selectivity,
            "positive_selectivity_rate": (
                positive_selectivity
                / len(evaluable_selectivity)
                if evaluable_selectivity
                else None
            ),
            "corrupted_reliability_decreased_count": rel_down,
            "corrupted_reliability_decreased_rate": (
                rel_down / len(evaluable_rel_down)
                if evaluable_rel_down
                else None
            ),
            "correct_weight_shift_count": correct_weight,
            "correct_weight_shift_rate": (
                correct_weight / len(evaluable_weight)
                if evaluable_weight
                else None
            ),
            "selectivity_distribution": population_stats(
                [row["reliability_selectivity"] for row in subset]
            ),
        }

    overall = {
        "hypothesis": (
            "H2: corruption should preferentially reduce the corrupted "
            "modality's reliability and shift weight toward the intact "
            "modality."
        ),
        "by_architecture": summary_by_architecture,
    }
    return detail, overall


# =====================================================================
# H3 — Utility / robustness
# =====================================================================

def robustness_auc_rows(
    rows: Sequence[Dict[str, Any]],
    clean_index: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for architecture in EXPECTED_ARCHITECTURES:
        for seed in EXPECTED_SEEDS:
            for corruption_type in CONTINUOUS_CORRUPTIONS:
                for modality in MODALITIES:
                    series = continuous_series_with_clean(
                        rows,
                        clean_index,
                        architecture,
                        seed,
                        corruption_type,
                        modality,
                    )
                    xs = [
                        float(row["severity"])
                        for row in series
                    ]
                    f1s = [
                        float(row["macro_f1"])
                        for row in series
                        if row["macro_f1"] is not None
                    ]
                    if len(xs) != len(f1s):
                        raise RuntimeError(
                            "Missing Macro-F1 in continuous robustness curve."
                        )

                    clean_f1 = f1s[0]
                    normalized = [
                        value / clean_f1
                        if clean_f1 != 0.0
                        else 0.0
                        for value in f1s
                    ]

                    output.append(
                        {
                            "architecture": architecture,
                            "model_seed": seed,
                            "corruption_type": corruption_type,
                            "corrupted_modality": modality,
                            "clean_macro_f1": clean_f1,
                            "macro_f1_auc_0_to_1": trapezoid_auc(xs, f1s),
                            "normalized_macro_f1_auc_0_to_1": (
                                trapezoid_auc(xs, normalized)
                            ),
                            "macro_f1_at_severity_1": f1s[-1],
                            "delta_macro_f1_at_severity_1": (
                                f1s[-1] - clean_f1
                            ),
                        }
                    )
    return output


def discrete_robustness_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "architecture": row["architecture"],
            "model_seed": row["model_seed"],
            "corruption_type": row["corruption_type"],
            "corrupted_modality": row["corrupted_modality"],
            "macro_f1": row["macro_f1"],
            "delta_macro_f1_from_clean": (
                row["delta_macro_f1_from_clean"]
            ),
            "mean_text_reliability": row["mean_text_reliability"],
            "mean_vision_reliability": row["mean_vision_reliability"],
            "reliability_selectivity": row["reliability_selectivity"],
        }
        for row in rows
        if row["corruption_type"] in DISCRETE_CORRUPTIONS
    ]


def analyse_h3(
    rows: Sequence[Dict[str, Any]],
    clean_index: Dict[Tuple[str, int], Dict[str, Any]],
    h2_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    auc_detail = robustness_auc_rows(rows, clean_index)
    discrete_detail = discrete_robustness_rows(rows)

    by_architecture: Dict[str, Any] = {}

    for architecture in EXPECTED_ARCHITECTURES:
        auc_subset = [
            row
            for row in auc_detail
            if row["architecture"] == architecture
        ]
        discrete_subset = [
            row
            for row in discrete_detail
            if row["architecture"] == architecture
        ]

        by_architecture[architecture] = {
            "normalized_auc_distribution": population_stats(
                [
                    row["normalized_macro_f1_auc_0_to_1"]
                    for row in auc_subset
                ]
            ),
            "severity_1_delta_macro_f1_distribution": population_stats(
                [
                    row["delta_macro_f1_at_severity_1"]
                    for row in auc_subset
                ]
            ),
            "discrete_delta_macro_f1_distribution": population_stats(
                [
                    row["delta_macro_f1_from_clean"]
                    for row in discrete_subset
                ]
            ),
        }

    # Does more positive reliability selectivity associate with robustness?
    # Robustness is represented as delta Macro-F1 from clean, where values
    # closer to zero are better.
    h2_valid = [
        row
        for row in h2_rows
        if (
            row["reliability_selectivity"] is not None
            and row["delta_macro_f1_from_clean"] is not None
        )
    ]

    selectivity_vs_robustness = spearman(
        [row["reliability_selectivity"] for row in h2_valid],
        [row["delta_macro_f1_from_clean"] for row in h2_valid],
    )

    by_reliability_architecture: Dict[str, Any] = {}
    for architecture in ("M2b", "M3"):
        subset = [
            row
            for row in h2_valid
            if row["architecture"] == architecture
        ]
        by_reliability_architecture[architecture] = {
            "spearman_selectivity_vs_delta_macro_f1": spearman(
                [row["reliability_selectivity"] for row in subset],
                [row["delta_macro_f1_from_clean"] for row in subset],
            ),
            "n_conditions": len(subset),
        }

    summary = {
        "hypothesis": (
            "H3: appropriate reliability response should associate with "
            "smaller classification degradation."
        ),
        "by_architecture": by_architecture,
        "overall_spearman_selectivity_vs_delta_macro_f1": (
            selectivity_vs_robustness
        ),
        "by_reliability_architecture": by_reliability_architecture,
    }

    return auc_detail, discrete_detail, summary


# =====================================================================
# Architecture / modality summaries
# =====================================================================

def architecture_condition_summary(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        Tuple[str, str, str, Optional[float]],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        if row["corruption_type"] == "clean":
            continue

        key = (
            str(row["architecture"]),
            str(row["corruption_type"]),
            str(row["corrupted_modality"]),
            row["severity"],
        )
        grouped[key].append(row)

    output: List[Dict[str, Any]] = []

    for key, subset in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            -1.0 if item[0][3] is None else float(item[0][3]),
        ),
    ):
        architecture, corruption_type, modality, severity = key
        f1_delta_stats = population_stats(
            [row["delta_macro_f1_from_clean"] for row in subset]
        )

        output.append(
            {
                "architecture": architecture,
                "corruption_type": corruption_type,
                "corrupted_modality": modality,
                "severity": severity,
                "seed_count": len(subset),
                "mean_macro_f1": population_stats(
                    [row["macro_f1"] for row in subset]
                )["mean"],
                "mean_delta_macro_f1_from_clean": (
                    f1_delta_stats["mean"]
                ),
                "std_sample_delta_macro_f1_from_clean": (
                    f1_delta_stats["std_sample"]
                ),
                "mean_text_reliability": population_stats(
                    [row["mean_text_reliability"] for row in subset]
                )["mean"],
                "mean_vision_reliability": population_stats(
                    [row["mean_vision_reliability"] for row in subset]
                )["mean"],
                "mean_reliability_selectivity": population_stats(
                    [row["reliability_selectivity"] for row in subset]
                )["mean"],
            }
        )

    return output


def modality_dependency_summary(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare text vs vision corruption effects. More-negative delta Macro-F1
    means greater dependency on that corrupted modality.
    """
    result: Dict[str, Any] = {}

    for architecture in EXPECTED_ARCHITECTURES:
        architecture_rows = [
            row for row in rows
            if row["architecture"] == architecture
            and row["corruption_type"] != "clean"
        ]

        text_deltas = [
            row["delta_macro_f1_from_clean"]
            for row in architecture_rows
            if (
                row["corrupted_modality"] == "text"
                and row["delta_macro_f1_from_clean"] is not None
            )
        ]
        vision_deltas = [
            row["delta_macro_f1_from_clean"]
            for row in architecture_rows
            if (
                row["corrupted_modality"] == "vision"
                and row["delta_macro_f1_from_clean"] is not None
            )
        ]

        mean_text_delta = (
            mean(text_deltas) if text_deltas else None
        )
        mean_vision_delta = (
            mean(vision_deltas) if vision_deltas else None
        )

        result[architecture] = {
            "text_corruption_delta_macro_f1": population_stats(text_deltas),
            "vision_corruption_delta_macro_f1": population_stats(
                vision_deltas
            ),
            "mean_vision_minus_text_degradation": (
                None
                if (
                    mean_text_delta is None
                    or mean_vision_delta is None
                )
                else (
                    mean_vision_delta - mean_text_delta
                )
            ),
            "interpretation": (
                "Negative vision-minus-text degradation means vision "
                "corruption is more damaging on average."
            ),
        }

    return result


# =====================================================================
# Markdown report
# =====================================================================

def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown_report(
    path: Path,
    *,
    integrity: Dict[str, Any],
    h1: Dict[str, Any],
    h2: Dict[str, Any],
    h3: Dict[str, Any],
    modality_dependency: Dict[str, Any],
) -> None:
    lines: List[str] = []

    lines.append("# AEGIS v0.26 Reliability Stress-Test Analysis")
    lines.append("")
    lines.append(
        "Status: validation-only diagnostic; official Fakeddit test split "
        "not accessed."
    )
    lines.append("")
    lines.append("## Experiment integrity")
    lines.append("")
    lines.append(
        f"- Condition rows analysed: {integrity['row_count']}"
    )
    lines.append(
        f"- Architectures: {', '.join(integrity['architectures'])}"
    )
    lines.append(
        "- Seeds: "
        + ", ".join(str(seed) for seed in integrity["seeds"])
    )
    lines.append(
        "- Integrity checks passed: "
        + str(integrity["integrity_checks_passed"])
    )
    lines.append("")

    lines.append("## H1 — Sensitivity")
    lines.append("")
    lines.append(
        "Expected direction: increasing corruption severity should reduce "
        "the corrupted modality's learned reliability."
    )
    lines.append("")
    lines.append(
        "- Curves evaluated: "
        + str(
            h1[
                "number_of_architecture_seed_family_modality_curves"
            ]
        )
    )
    lines.append(
        "- Curves with expected negative reliability correlation: "
        f"{h1['reliability_direction_supported_count']}/"
        f"{h1['reliability_direction_evaluable_count']} "
        f"({fmt(h1['reliability_direction_supported_rate'] * 100 if h1['reliability_direction_supported_rate'] is not None else None, 1)}%)"
    )
    lines.append(
        "- Mean Spearman(severity, corrupted reliability): "
        + fmt(
            h1["spearman_reliability_distribution"]["mean"]
        )
    )
    lines.append(
        "- Mean Spearman(severity, corrupted weight): "
        + fmt(
            h1["spearman_weight_distribution"]["mean"]
        )
    )
    lines.append(
        "- Mean Spearman(severity, Macro-F1): "
        + fmt(
            h1["spearman_macro_f1_distribution"]["mean"]
        )
    )
    lines.append("")

    lines.append("## H2 — Selectivity")
    lines.append("")
    lines.append(
        "Expected direction: the corrupted modality should lose more "
        "reliability than the intact modality, with fusion weight shifting "
        "toward the intact modality."
    )
    lines.append("")

    for architecture in ("M2b", "M3"):
        payload = h2["by_architecture"][architecture]
        lines.append(f"### {architecture}")
        lines.append("")
        lines.append(
            "- Positive reliability-selectivity rate: "
            + fmt(payload["positive_selectivity_rate"] * 100, 1)
            + "%"
        )
        lines.append(
            "- Corrupted-reliability-decreased rate: "
            + fmt(
                payload[
                    "corrupted_reliability_decreased_rate"
                ] * 100,
                1,
            )
            + "%"
        )
        lines.append(
            "- Correct weight-shift rate: "
            + fmt(payload["correct_weight_shift_rate"] * 100, 1)
            + "%"
        )
        lines.append(
            "- Mean selectivity: "
            + fmt(payload["selectivity_distribution"]["mean"], 8)
        )
        lines.append("")

    lines.append("## H3 — Utility / robustness")
    lines.append("")
    lines.append(
        "Normalized AUC is the area under the Macro-F1-vs-severity curve "
        "after dividing each curve by its own clean Macro-F1. Values closer "
        "to 1 indicate greater robustness."
    )
    lines.append("")

    for architecture in EXPECTED_ARCHITECTURES:
        payload = h3["by_architecture"][architecture]
        lines.append(f"### {architecture}")
        lines.append("")
        lines.append(
            "- Mean normalized corruption AUC: "
            + fmt(
                payload[
                    "normalized_auc_distribution"
                ]["mean"]
            )
        )
        lines.append(
            "- Mean ΔMacro-F1 at severity 1.0: "
            + fmt(
                payload[
                    "severity_1_delta_macro_f1_distribution"
                ]["mean"]
            )
        )
        lines.append(
            "- Mean discrete-corruption ΔMacro-F1: "
            + fmt(
                payload[
                    "discrete_delta_macro_f1_distribution"
                ]["mean"]
            )
        )
        lines.append("")

    lines.append(
        "- Overall Spearman(reliability selectivity, ΔMacro-F1): "
        + fmt(
            h3[
                "overall_spearman_selectivity_vs_delta_macro_f1"
            ]
        )
    )
    lines.append("")

    lines.append("## Modality dependency")
    lines.append("")
    for architecture in EXPECTED_ARCHITECTURES:
        payload = modality_dependency[architecture]
        lines.append(
            f"- **{architecture}** mean vision-minus-text degradation: "
            + fmt(
                payload[
                    "mean_vision_minus_text_degradation"
                ]
            )
            + " (negative means vision corruption is more damaging)"
        )
    lines.append("")

    lines.append("## Interpretation rule")
    lines.append("")
    lines.append(
        "These outputs quantify behaviour of learned scalar modality scores "
        "under controlled representation corruption. They do not establish "
        "open-world factual reliability, provenance quality, or calibrated "
        "trustworthiness. The validation set was used during model "
        "development, so all conclusions remain diagnostic until replicated "
        "on an independent diagnostic/dev split."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Console
# =====================================================================

def print_summary(
    integrity: Dict[str, Any],
    h1: Dict[str, Any],
    h2: Dict[str, Any],
    h3: Dict[str, Any],
) -> None:
    print("=" * 78)
    print("AEGIS v0.26.1 — RELIABILITY STRESS-TEST ANALYSIS")
    print("=" * 78)
    print("Condition rows:", integrity["row_count"])
    print("Integrity checks: PASSED")
    print("Official test split: SEALED / NOT ACCESSED")
    print()

    rate = h1["reliability_direction_supported_rate"]
    print("H1 — SENSITIVITY")
    print(
        "  expected negative reliability curves:",
        f"{h1['reliability_direction_supported_count']}/"
        f"{h1['reliability_direction_evaluable_count']}",
        (
            f"({100.0 * rate:.1f}%)"
            if rate is not None
            else ""
        ),
    )
    print(
        "  mean Spearman severity↔reliability:",
        fmt(
            h1["spearman_reliability_distribution"]["mean"],
            4,
        ),
    )
    print()

    print("H2 — SELECTIVITY")
    for architecture in ("M2b", "M3"):
        payload = h2["by_architecture"][architecture]
        print(
            f"  {architecture}: "
            f"selectivity+={100.0 * payload['positive_selectivity_rate']:.1f}% "
            f"reliability↓={100.0 * payload['corrupted_reliability_decreased_rate']:.1f}% "
            f"correct weight shift={100.0 * payload['correct_weight_shift_rate']:.1f}%"
        )
    print()

    print("H3 — ROBUSTNESS")
    for architecture in EXPECTED_ARCHITECTURES:
        payload = h3["by_architecture"][architecture]
        print(
            f"  {architecture}: normalized AUC="
            f"{payload['normalized_auc_distribution']['mean']:.4f}; "
            f"mean ΔF1@1="
            f"{payload['severity_1_delta_macro_f1_distribution']['mean']:.4f}"
        )
    print(
        "  Spearman selectivity↔ΔMacro-F1:",
        fmt(
            h3[
                "overall_spearman_selectivity_vs_delta_macro_f1"
            ],
            4,
        ),
    )
    print()
    print("Analysis complete.")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input)
    integrity = validate_dataset(rows, args.expected_rows)
    clean_index = build_clean_index(rows)

    h1_rows, h1_summary = analyse_h1(
        rows,
        clean_index,
    )
    h2_rows, h2_summary = analyse_h2(rows)
    (
        h3_auc_rows,
        h3_discrete_rows,
        h3_summary,
    ) = analyse_h3(
        rows,
        clean_index,
        h2_rows,
    )

    condition_summary = architecture_condition_summary(rows)
    modality_dependency = modality_dependency_summary(rows)

    save_json(
        args.output_root / "integrity.json",
        {
            "analysis_version": VERSION,
            "input": str(args.input),
            **integrity,
        },
    )
    save_csv(
        args.output_root / "h1_sensitivity_curves.csv",
        h1_rows,
    )
    save_json(
        args.output_root / "h1_sensitivity_summary.json",
        h1_summary,
    )
    save_csv(
        args.output_root / "h2_selectivity_conditions.csv",
        h2_rows,
    )
    save_json(
        args.output_root / "h2_selectivity_summary.json",
        h2_summary,
    )
    save_csv(
        args.output_root / "h3_continuous_robustness_auc.csv",
        h3_auc_rows,
    )
    save_csv(
        args.output_root / "h3_discrete_robustness.csv",
        h3_discrete_rows,
    )
    save_json(
        args.output_root / "h3_utility_summary.json",
        h3_summary,
    )
    save_csv(
        args.output_root / "architecture_condition_seed_summary.csv",
        condition_summary,
    )
    save_json(
        args.output_root / "modality_dependency_summary.json",
        modality_dependency,
    )

    master_summary = {
        "analysis_version": VERSION,
        "input": str(args.input),
        "integrity": integrity,
        "H1_sensitivity": h1_summary,
        "H2_selectivity": h2_summary,
        "H3_utility": h3_summary,
        "modality_dependency": modality_dependency,
        "official_test_split_accessed": False,
        "scientific_status": (
            "validation-only diagnostic; seed runs evaluate the same fixed "
            "validation examples and are not independent dataset samples"
        ),
    }
    save_json(
        args.output_root / "v026_reliability_analysis_summary.json",
        master_summary,
    )

    write_markdown_report(
        args.output_root / "v026_reliability_analysis_report.md",
        integrity=integrity,
        h1=h1_summary,
        h2=h2_summary,
        h3=h3_summary,
        modality_dependency=modality_dependency,
    )

    print_summary(
        integrity,
        h1_summary,
        h2_summary,
        h3_summary,
    )
    print("Output root:", args.output_root)


if __name__ == "__main__":
    main()

"""
AEGIS v0.27 reliability and evidence-quality research utilities.

The v0.27 package is intentionally isolated from the established AEGIS
alignment backbone until each new mechanism has passed its pre-registered
behavioral tests.
"""

from .compatibility import (
    CompatibilityCondition,
    CompatibilityTargets,
    PermutationMismatchPlan,
    apply_permutation_mismatch_plan,
    make_compatibility_targets,
    make_permutation_mismatch_plan,
    scalar_compatibility_target,
)
from .corruption import (
    CorruptionFamily,
    CorruptionResult,
    attenuation,
    compute_feature_std,
    corrupt_representation,
    deterministic_derangement_indices,
    gaussian_noise,
    permutation_mismatch,
    validate_feature_std,
    zero_dropout,
)
from .quality_estimator import (
    ModalityQualityEstimator,
    QualityEstimatorConfig,
)
from .reliability_controller import (
    DeterministicReliabilityController,
    ReliabilityControllerOutput,
    SelectiveReliabilityController,
)
from .targets import (
    ModalityName,
    QualityCondition,
    QualityTargets,
    continuous_quality_target,
    make_quality_targets,
    scalar_quality_targets,
)

__all__ = [
    "CompatibilityCondition",
    "CompatibilityTargets",
    "CorruptionFamily",
    "CorruptionResult",
    "ModalityName",
    "ModalityQualityEstimator",
    "DeterministicReliabilityController",
    "PermutationMismatchPlan",
    "QualityCondition",
    "QualityEstimatorConfig",
    "QualityTargets",
    "ReliabilityControllerOutput",
    "SelectiveReliabilityController",
    "apply_permutation_mismatch_plan",
    "attenuation",
    "compute_feature_std",
    "continuous_quality_target",
    "corrupt_representation",
    "deterministic_derangement_indices",
    "gaussian_noise",
    "make_compatibility_targets",
    "make_permutation_mismatch_plan",
    "make_quality_targets",
    "permutation_mismatch",
    "scalar_compatibility_target",
    "scalar_quality_targets",
    "validate_feature_std",
    "zero_dropout",
]

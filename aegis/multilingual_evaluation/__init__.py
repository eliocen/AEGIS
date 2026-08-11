from .comparison import (
    language_performance_gap,
    summarize_cross_lingual_transfer,
)

from .evaluator import (
    MultilingualEvaluator,
)

from .language import (
    LANGUAGE_ALIASES,
    LanguagePair,
    normalize_language,
)

from .report import (
    MultilingualEvaluationReport,
)

from .result import (
    CrossLingualRunResult,
    LanguageEvaluationResult,
)

from .transfer import (
    build_transfer_matrix,
)


__all__ = [
    "LANGUAGE_ALIASES",
    "normalize_language",
    "LanguagePair",

    "LanguageEvaluationResult",
    "CrossLingualRunResult",

    "MultilingualEvaluator",

    "build_transfer_matrix",
    "summarize_cross_lingual_transfer",
    "language_performance_gap",

    "MultilingualEvaluationReport",
]
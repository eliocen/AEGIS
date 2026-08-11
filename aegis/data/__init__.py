from .collate import (
    aegis_collate_fn,
)

from .dataset import (
    AEGISResearchDataset,
)

from .labels import (
    DatasetLabel,
    label_to_hierarchical_targets,
    normalize_dataset_label,
)

from .loader import (
    create_dataloader,
)

from .record import (
    ResearchSample,
)

from .split import (
    group_aware_split,
    random_split_samples,
)

from .validation import (
    validate_research_sample,
)

from .manifest import (
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)

from .statistics import (
    compute_dataset_statistics,
)

__all__ = [
    "DatasetLabel",
    "ResearchSample",
    "AEGISResearchDataset",
    "normalize_dataset_label",
    "label_to_hierarchical_targets",
    "validate_research_sample",
    "aegis_collate_fn",
    "create_dataloader",
    "random_split_samples",
    "group_aware_split",
    "compute_dataset_statistics",
    "build_split_manifest",
    "save_split_manifest",
    "load_split_manifest",
]
"""
Dataset integrity auditing for AEGIS empirical research.
"""

from .fakeddit import (
    CrossSplitAuditResult,
    FakedditDatasetAudit,
    FakedditFullAuditor,
    SplitAuditResult,
    sha256_file,
)

__all__ = [
    "SplitAuditResult",
    "CrossSplitAuditResult",
    "FakedditDatasetAudit",
    "FakedditFullAuditor",
    "sha256_file",
]
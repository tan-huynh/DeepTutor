"""Co-Scientist agent package."""

from .pipeline import CoScientistPipeline
from .data_structures import (
    CoScientistConfig,
    EvidenceItem,
    Hypothesis,
    CoScientistResult,
    QualityMetrics,
)

__all__ = [
    "CoScientistPipeline",
    "CoScientistConfig",
    "EvidenceItem",
    "Hypothesis",
    "CoScientistResult",
    "QualityMetrics",
]

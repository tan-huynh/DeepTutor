"""Co-Scientist core data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CoScientistConfig:
    """Runtime configuration for a Co-Scientist run."""

    goal: str
    max_hypotheses: int = 3
    max_evidence: int = 8
    use_web_search: bool = True
    use_paper_search: bool = True
    use_rag: bool = True
    use_graphiti: bool = True  # Added Graphiti KG
    tournament_rounds: int = 1
    evolution_iterations: int = 2  # Added: Outer loop iterations
    temperature: float = 0.45
    language: str = "en"
    kb_name: str = "default"
    # Seed evidence injected by the bridge (from HybridRetriever)
    seed_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Conversation history for multi-turn context
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """A single piece of grounding evidence."""

    id: str                # e.g. "E1", "E2"
    source: str            # "rag", "paper_search", "web_search", "bridge_hybrid"
    title: str
    snippet: str
    url: str = ""
    score: float = 0.0
    doi: str = ""
    year: int = 0
    citations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "score": self.score,
            "doi": self.doi,
            "year": self.year,
            "citations": self.citations,
        }


@dataclass
class Hypothesis:
    """A single research hypothesis with scoring dimensions."""

    id: str                 # e.g. "H1", "H2"
    statement: str          # One-sentence hypothesis statement
    rationale: str          # Scientific rationale (2-3 sentences)
    evidence_ids: list[str] = field(default_factory=list)  # IDs of supporting EvidenceItems

    # Scoring dimensions (0.0 – 1.0)
    novelty: float = 0.0
    testability: float = 0.0
    impact: float = 0.0
    risk: float = 0.0

    # Peer-review loop outputs
    critique: str = ""      # Reflection agent critique
    refinement: str = ""    # Evolution agent refinement
    
    # Outer refinement loop lineage
    iteration: int = 0
    parent_id: str = ""

    # Tournament
    rank_score: float = 0.0
    tournament_wins: int = 0
    tournament_losses: int = 0

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "evidence_ids": self.evidence_ids,
            "novelty": self.novelty,
            "testability": self.testability,
            "impact": self.impact,
            "risk": self.risk,
            "critique": self.critique,
            "refinement": self.refinement,
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "rank_score": self.rank_score,
            "tournament_wins": self.tournament_wins,
            "tournament_losses": self.tournament_losses,
        }


@dataclass
class QualityMetrics:
    """Aggregate quality metrics for a Co-Scientist run."""

    average_score: float = 0.0
    evidence_count: int = 0
    top_hypothesis_id: str | None = None
    quality_gates: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_score": self.average_score,
            "evidence_count": self.evidence_count,
            "top_hypothesis_id": self.top_hypothesis_id,
            "quality_gates": self.quality_gates,
        }


@dataclass
class CoScientistResult:
    """Final output of a Co-Scientist pipeline run."""

    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    text: str = ""          # Final research proposal (Markdown)
    blocked: bool = False
    safety_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "evidence": [e.to_dict() for e in self.evidence],
            "quality": self.quality.to_dict(),
            "text": self.text,
            "blocked": self.blocked,
            "safety": {
                "blocked": self.blocked,
                "reason": self.safety_reason,
            } if self.blocked else None,
        }


__all__ = [
    "CoScientistConfig",
    "EvidenceItem",
    "Hypothesis",
    "CoScientistResult",
    "QualityMetrics",
]

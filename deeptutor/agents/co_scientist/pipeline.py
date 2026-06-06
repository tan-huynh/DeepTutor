"""
Co-Scientist Pipeline — the local fallback and orchestrator.

Runs: Evidence gathering -> Hypothesis generation -> Peer critique -> Tournament ranking -> Proposal.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from .data_structures import CoScientistConfig, CoScientistResult, QualityMetrics
from .evidence_agent import EvidenceAgent
from .hypothesis_agent import HypothesisAgent
from .critique_agent import CritiqueAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent
from .proposal_agent import ProposalAgent


class CoScientistPipeline:
    """Orchestrates the Co-Scientist multi-agent flow locally."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.evidence_agent = EvidenceAgent(config)
        self.hypothesis_agent = HypothesisAgent(config)
        self.critique_agent = CritiqueAgent(config)
        self.ranking_agent = RankingAgent(config)
        self.evolution_agent = EvolutionAgent(config)
        self.proposal_agent = ProposalAgent(config)

    async def run(
        self,
        cs_config: CoScientistConfig,
        call_tool: Callable[[str, str], Awaitable[str]],
        progress_cb: Callable[[str, Any], None] | None = None,
    ) -> CoScientistResult:
        """
        Executes the full pipeline.
        
        Args:
            cs_config: The run configuration.
            call_tool: Tool execution callback provided by the backend.
            progress_cb: Callback for SSE streaming (stage, msg or partial content).
        """
        result = CoScientistResult()

        def _notify(stage: str, msg: str):
            if progress_cb:
                progress_cb(stage, {"message": msg})

        # 1. Gather Evidence
        evidence = await self.evidence_agent.process(
            cs_config=cs_config,
            call_tool=call_tool,
            progress_cb=lambda s, m: _notify(s, m),
        )
        result.evidence = evidence

        if not evidence:
            _notify("error", "No evidence found. Proceeding with zero evidence.")

        # 2. Generate Initial Hypotheses
        hypotheses = await self.hypothesis_agent.process(
            cs_config=cs_config,
            evidence=evidence,
            progress_cb=lambda s, m: _notify(s, m),
        )
        if not hypotheses:
            _notify("error", "Failed to generate initial hypotheses.")
            return result

        # --- OUTER REFINEMENT LOOP ---
        iterations = max(1, min(3, cs_config.evolution_iterations))
        
        for iteration in range(1, iterations + 1):
            _notify("evolution", f"Starting refinement loop {iteration}/{iterations}...")

            # 3. Peer Critique
            hypotheses = await self.critique_agent.process(
                cs_config=cs_config,
                hypotheses=hypotheses,
                progress_cb=lambda s, m: _notify(s, m),
            )

            # 4. Tournament Ranking
            hypotheses = await self.ranking_agent.process(
                cs_config=cs_config,
                hypotheses=hypotheses,
                progress_cb=lambda s, m: _notify(s, m),
            )
            
            # If not the last iteration, evolve using targeted evidence
            if iteration < iterations and hypotheses:
                top_hyp = hypotheses[0]
                
                # Targeted Evidence Search
                new_evidence = await self.evidence_agent.process_targeted(
                    cs_config=cs_config,
                    hypothesis=top_hyp,
                    call_tool=call_tool,
                    progress_cb=lambda s, m: _notify(s, m),
                    existing_evidence_count=len(evidence),
                )
                
                if new_evidence:
                    evidence.extend(new_evidence)
                
                # Evolution
                hypotheses = await self.evolution_agent.process(
                    cs_config=cs_config,
                    hypotheses=hypotheses,
                    new_evidence=new_evidence,
                    iteration=iteration,
                    progress_cb=lambda s, m: _notify(s, m),
                )

        result.hypotheses = hypotheses

        # Calculate Final Quality Metrics
        if hypotheses:
            avg_score = sum(h.rank_score for h in hypotheses) / len(hypotheses)
            result.quality = QualityMetrics(
                average_score=avg_score,
                evidence_count=len(evidence),
                top_hypothesis_id=hypotheses[0].id,
                quality_gates={"grounded": len(evidence) > 0, "ranked": True},
            )

        # 5. Draft Final Proposal
        def _stream_proposal(stage: str, chunk: str):
            if progress_cb and stage == "proposal_stream":
                progress_cb("stream", {"delta": chunk})
            else:
                _notify(stage, chunk)

        text = await self.proposal_agent.process(
            cs_config=cs_config,
            hypotheses=hypotheses,
            evidence=evidence,
            progress_cb=_stream_proposal,
        )
        result.text = text

        _notify("complete", "Co-Scientist run finished successfully.")

        return result


__all__ = ["CoScientistPipeline"]

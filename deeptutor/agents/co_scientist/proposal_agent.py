"""
ProposalAgent — drafts the final research proposal in Markdown.

Consolidates the top-ranked hypotheses, evidence, and critiques
into a cohesive scientific report.
"""

from __future__ import annotations

from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, EvidenceItem, Hypothesis


_SYSTEM_ZH = (
    "你是一位资深的首席科学家。你的任务是基于之前生成的假设、同行评审意见和证据，"
    "撰写一份最终的、严谨的科学研究提案 (Research Proposal)。"
)

_SYSTEM_EN = (
    "You are a senior Principal Investigator. Your task is to write a final, rigorous "
    "scientific Research Proposal based on the generated hypotheses, peer critiques, and evidence."
)


class ProposalAgent(BaseAgent):
    """Drafts the final Markdown research proposal."""

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
    ) -> None:
        language = config.get("system", {}).get("language", "en")
        super().__init__(
            module_name="co_scientist",
            agent_name="proposal_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )

    async def process(  # type: ignore[override]
        self,
        *,
        cs_config: CoScientistConfig,
        hypotheses: list[Hypothesis],
        evidence: list[EvidenceItem],
        progress_cb=None,
    ) -> str:
        """Generates the final markdown proposal text."""
        if progress_cb:
            progress_cb("proposal", "Drafting final research proposal…")

        system_prompt = _SYSTEM_ZH if self.language == "zh" else _SYSTEM_EN
        user_prompt = self._build_prompt(cs_config.goal, hypotheses, evidence)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="draft_proposal",
        ):
            chunks.append(chunk)
            if progress_cb:
                progress_cb("proposal_stream", chunk)

        response = "".join(chunks)

        if progress_cb:
            progress_cb("proposal", "Final proposal complete.")

        return response

    def _build_prompt(self, goal: str, hypotheses: list[Hypothesis], evidence: list[EvidenceItem]) -> str:
        e_text = "\n\n".join(f"[{e.id}] {e.title} - {e.snippet}" for e in evidence)
        h_text = "\n\n".join(
            f"Hypothesis {h.id} (Rank Score: {h.rank_score:.2f}):\n"
            f"Statement: {h.statement}\nRationale: {h.rationale}\n"
            f"Critique: {h.critique}\nRefinement: {h.refinement}\n"
            f"Evidence used: {h.evidence_ids}"
            for h in hypotheses
        )

        return f"""
Research Goal:
{goal}

Grounding Evidence:
{e_text}

Generated & Ranked Hypotheses:
{h_text}

Task:
Write a comprehensive, professional Research Proposal in Markdown format.
Do not wrap the whole response in a JSON block. Just output the Markdown text directly.

Structure required:
1. # Executive Summary (Brief overview of the goal and the top hypothesis)
2. ## Grounding Evidence (Summary of the literature/data used)
3. ## Competing Hypotheses (Present the hypotheses, their rationales, and their relative strengths/weaknesses based on the critiques)
4. ## Proposed Experimental Design (How to test the top-ranked hypothesis)
5. ## Expected Impact & Risks

Be academic, precise, and cite evidence using the [E1] format in the text.
""".strip()


__all__ = ["ProposalAgent"]

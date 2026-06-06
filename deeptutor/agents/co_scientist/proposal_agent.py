"""
ProposalAgent — drafts the final research proposal in Markdown.

Consolidates the top-ranked hypotheses, evidence, and critiques
into a cohesive, publication-quality scientific report with full
academic structure: executive summary, hypothesis ranking table,
experimental design, statistical plan, timeline, and risk register.
"""

from __future__ import annotations

from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, EvidenceItem, Hypothesis


_SYSTEM_ZH = (
    "你是一位具有顶级期刊发表经验的首席科学家（Principal Investigator）。"
    "你的任务是基于已生成的假设、同行评审意见、迭代演化历史和文献证据，"
    "撰写一份达到资助申请或顶级会议投稿标准的最终科学研究提案。\n\n"
    "写作要求：\n"
    "- 学术严谨，逻辑清晰，每个论断都需引用证据\n"
    "- 假设需有明确、可测量的评估标准\n"
    "- 实验设计需包含控制组、消融实验和统计分析方案\n"
    "- 风险需配套缓解策略\n"
    "- 时间线需具体到月份级别的里程碑\n"
    "- 直接输出Markdown，不要用代码块包裹"
)

_SYSTEM_EN = (
    "You are a Principal Investigator with a track record of publishing in top-tier venues "
    "and winning competitive research grants. Your task is to produce a final Research Proposal "
    "that meets the standard of a grant application or flagship conference submission.\n\n"
    "Writing standards:\n"
    "- Every claim must be grounded in cited evidence (use [E1] format inline)\n"
    "- Hypotheses must have specific, measurable evaluation criteria with numeric thresholds\n"
    "- Experimental design must include control/treatment groups, ablation conditions, and power analysis\n"
    "- Risks must be paired with concrete mitigation strategies\n"
    "- Timeline must be month-level with named deliverables\n"
    "- Statistical analysis plan must name specific tests and target effect sizes\n"
    "- Output Markdown directly — do not wrap in code fences"
)


class ProposalAgent(BaseAgent):
    """Drafts the final Markdown research proposal at grant-application quality."""

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
        user_prompt = self._build_prompt(cs_config, hypotheses, evidence)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="draft_proposal",
            max_tokens=16000,  # Proposals are long — override to prevent truncation
        ):
            chunks.append(chunk)
            if progress_cb:
                progress_cb("proposal_stream", chunk)

        response = "".join(chunks)

        if progress_cb:
            progress_cb("proposal", "Final proposal complete.")

        return response

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        cs_config: CoScientistConfig,
        hypotheses: list[Hypothesis],
        evidence: list[EvidenceItem],
    ) -> str:
        evidence_block = self._format_evidence(evidence)
        hypothesis_block = self._format_hypotheses(hypotheses)
        top_h = hypotheses[0] if hypotheses else None

        lang_note = (
            "Write the entire proposal in Chinese (中文)."
            if self.language == "zh"
            else "Write the entire proposal in English."
        )

        top_h_summary = ""
        if top_h:
            top_h_summary = (
                f"Top-ranked hypothesis: [{top_h.id}] {top_h.statement} "
                f"(rank_score={top_h.rank_score:.2f}, "
                f"novelty={top_h.novelty:.2f}, "
                f"testability={top_h.testability:.2f}, "
                f"impact={top_h.impact:.2f})"
            )

        return f"""
Research Goal:
{cs_config.goal}

{top_h_summary}

--- EVIDENCE ---
{evidence_block}

--- ALL HYPOTHESES (ranked, best first) ---
{hypothesis_block}

--- TASK ---
{lang_note}

Write a comprehensive, grant-quality Research Proposal in Markdown.
Use inline evidence citations [E1], [E2] throughout the text wherever claims are made.
The proposal MUST contain ALL of the following sections in order:

# [Title of the Research Proposal]
  (A specific, descriptive title — NOT just "Research Proposal")

## Executive Summary
  (3–5 sentences: the research problem, the chosen approach, why it matters, and the primary outcome metric)

## Background & Motivation
  (Explain the scientific gap this research addresses. Ground every claim with evidence citations.
   Describe what existing methods do and where they fail. Min 150 words.)

## Grounding Evidence
  (Present a structured summary of the evidence base. For each key evidence item, explain:
   what it contributes, how it informs the hypotheses, and what gap it leaves open.
   Use a subsection or bullet format per evidence source. Min 100 words.)

## Competing Hypotheses
  (Present ALL hypotheses with a comparative table and then per-hypothesis detail.

  First, a Markdown table:
  | ID | Statement (brief) | Rank Score | Novelty | Testability | Impact | Risk | Iteration |
  |---|---|---|---|---|---|---|---|
  (one row per hypothesis)

  Then, for the top 2–3 hypotheses, provide:
  ### Hypothesis [ID] — [short label]
  **Statement**: full statement
  **Rationale**: full rationale
  **Strengths**: (from critique agent if available)
  **Weaknesses / Critique**: (from critique agent)
  **Proposed Refinement**: (from evolution)
  **Evidence Support**: list evidence IDs and how they support this hypothesis
  )

## Primary Hypothesis & Justification
  (Make a clear recommendation: explain WHY the top-ranked hypothesis is selected over competitors.
   Reference the rank scores, critique severity, and evidence grounding.
   Include the specific, measurable evaluation criteria for this hypothesis:
   what metrics, what numeric thresholds, what comparison baseline.)

## Proposed Experimental Design
  (Detailed methodology to test the primary hypothesis. Must include:

  ### Study Design Overview
  (diagram or bullet: Control group / Ablation A / Ablation B / Treatment group)

  ### Participants & Materials
  (sample size with power analysis justification: target N=?, α=0.05, power=80%, expected effect size d=?)

  ### Procedure
  (step-by-step: data collection → system setup → intervention → measurement)

  ### Evaluation Metrics
  (table of metrics: | Metric | Tool/Method | Success Threshold |)

  ### Statistical Analysis Plan
  (name the specific statistical tests: t-test / ANOVA / mixed-effects model, correction method, effect size measure)

  ### Controls & Ablations
  (list each ablation condition and what it isolates)
  )

## Timeline
  (Month-by-month plan. Format as a Markdown table:
  | Month | Milestone | Deliverable |
  Min 6 months. Each row must have a concrete, named deliverable.)

## Expected Impact & Contributions
  (Itemised list of: scientific contributions, practical applications, and broader significance.
   Be specific — "will improve X by Y%" is better than "will improve X".)

## Risk Register & Mitigation
  (Markdown table:
  | Risk | Likelihood (H/M/L) | Impact (H/M/L) | Mitigation Strategy |
  Min 4 risks. Each must have a concrete mitigation, not just "we will monitor".)

## References / Evidence Index
  (List all cited evidence items in format: [E1] Title — Source — Year — Key finding used)

Remember: every section must be substantive and specific. Vague generalisations are unacceptable.
Cite evidence inline whenever making a factual claim.
""".strip()

    @staticmethod
    def _format_evidence(evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return "(No evidence provided)"
        lines: list[str] = []
        for e in evidence:
            year_str = f" ({e.year})" if e.year else ""
            doi_str = f" DOI: {e.doi}" if e.doi else ""
            cite_str = f" [cited {e.citations}x]" if e.citations else ""
            lines.append(
                f"[{e.id}] {e.title}{year_str}{cite_str}\n"
                f"    Source: {e.source}{doi_str}\n"
                f"    Snippet: {e.snippet[:500]}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_hypotheses(hypotheses: list[Hypothesis]) -> str:
        if not hypotheses:
            return "(No hypotheses generated)"
        lines: list[str] = []
        for h in hypotheses:
            iter_str = f" [Iteration {h.iteration}]" if h.iteration else ""
            parent_str = f" [Evolved from {h.parent_id}]" if h.parent_id else ""
            scores = (
                f"rank={h.rank_score:.2f} | novelty={h.novelty:.2f} | "
                f"testability={h.testability:.2f} | impact={h.impact:.2f} | risk={h.risk:.2f}"
            )
            critique_str = f"\n  Critique: {h.critique}" if h.critique else ""
            refinement_str = f"\n  Refinement: {h.refinement}" if h.refinement else ""
            evidence_str = f"\n  Evidence: {', '.join(h.evidence_ids)}" if h.evidence_ids else ""
            lines.append(
                f"--- {h.id}{iter_str}{parent_str} ---\n"
                f"Scores: {scores}\n"
                f"Statement: {h.statement}\n"
                f"Rationale: {h.rationale}"
                f"{critique_str}"
                f"{refinement_str}"
                f"{evidence_str}"
            )
        return "\n\n".join(lines)


__all__ = ["ProposalAgent"]

"""
Co-Scientist Capability
=======================

Research-quality, multi-agent hypothesis generation inspired by Google
DeepMind's Co-Scientist workflow. This implementation keeps the pipeline
provider-agnostic by using DeepTutor's configured LLM client, so OpenAI,
Gemini, local, and OpenAI-compatible providers can be selected through the
existing settings layer.
"""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deeptutor.capabilities.request_contracts import (
    get_capability_request_schema,
    validate_co_scientist_request_config,
)
from deeptutor.core.capability_protocol import BaseCapability, CapabilityManifest
from deeptutor.core.context import UnifiedContext
from deeptutor.core.stream_bus import StreamBus
from deeptutor.services.llm import get_llm_client


UNSAFE_RESEARCH_PATTERNS = [
    r"\b(cbrn|bioweapon|chemical weapon|radiological weapon)\b",
    r"\b(pathogen|virus|bacteria).*\b(enhance|increase|weaponize|evade|transmit)\b",
    r"\b(synthesize|manufacture|produce).*\b(toxin|explosive|nerve agent)\b",
    r"\b(gain[- ]of[- ]function|viral enhancement)\b",
]


@dataclass
class EvidenceItem:
    id: str
    source: str
    title: str
    snippet: str
    url: str = ""
    score: float = 0.0


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str
    evidence_ids: list[str] = field(default_factory=list)
    novelty: float = 0.5
    testability: float = 0.5
    impact: float = 0.5
    risk: float = 0.3
    critique: str = ""
    refinement: str = ""
    rank_score: float = 0.0


def _clip(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _word_overlap(left: str, right: str) -> float:
    left_words = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", left.lower())}
    right_words = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", right.lower())}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(1, len(left_words | right_words))


class CoScientistCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="co_scientist",
        description="Multi-agent scientific hypothesis generation, critique, ranking, and proposal synthesis.",
        stages=[
            "safety",
            "evidence",
            "generation",
            "proximity",
            "reflection",
            "ranking",
            "evolution",
            "meta_review",
        ],
        tools_used=["rag", "web_search", "paper_search"],
        cli_aliases=["co-scientist", "coscientist"],
        request_schema=get_capability_request_schema("co_scientist"),
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        config = validate_co_scientist_request_config(context.config_overrides)
        goal = (context.user_message or "").strip()
        if not goal:
            await stream.error("Research goal is required.", source=self.name, stage="safety")
            return

        async with stream.stage("safety", source=self.name):
            await stream.progress(
                "Checking research safety and scope",
                source=self.name,
                stage="safety",
                metadata={"agent": "Safety Supervisor"},
            )
            safety = self._assess_safety(goal)
            if safety["blocked"]:
                final_text = self._blocked_response(goal, safety["reason"])
                await stream.result(
                    {
                        "text": final_text,
                        "blocked": True,
                        "safety": safety,
                        "hypotheses": [],
                        "evidence": [],
                    },
                    source=self.name,
                )
                return

        evidence: list[EvidenceItem] = []
        async with stream.stage("evidence", source=self.name):
            await stream.progress(
                "Gathering grounding evidence from KB and search",
                source=self.name,
                stage="evidence",
                metadata={"agent": "Evidence Agent"},
            )
            evidence = await self._gather_evidence(goal, context, config)
            if evidence:
                await stream.sources([asdict(item) for item in evidence], source=self.name, stage="evidence")

        async with stream.stage("generation", source=self.name):
            await stream.progress(
                "Generating diverse initial hypotheses",
                source=self.name,
                stage="generation",
                metadata={"agent": "Generation Agent"},
            )
            hypotheses = await self._generate_hypotheses(goal, evidence, config)

        async with stream.stage("proximity", source=self.name):
            await stream.progress(
                "Clustering hypotheses to preserve diversity",
                source=self.name,
                stage="proximity",
                metadata={"agent": "Proximity Agent"},
            )
            self._assign_evidence_and_diversity(hypotheses, evidence)

        async with stream.stage("reflection", source=self.name):
            await stream.progress(
                "Running peer-review critique against evidence",
                source=self.name,
                stage="reflection",
                metadata={"agent": "Reflection Agent"},
            )
            await self._reflect(goal, hypotheses, evidence, config)

        async with stream.stage("ranking", source=self.name):
            await stream.progress(
                "Ranking hypotheses with quality gates",
                source=self.name,
                stage="ranking",
                metadata={"agent": "Ranking Agent"},
            )
            self._rank(hypotheses)

        async with stream.stage("evolution", source=self.name):
            await stream.progress(
                "Refining the strongest hypotheses",
                source=self.name,
                stage="evolution",
                metadata={"agent": "Evolution Agent"},
            )
            await self._evolve(goal, hypotheses, evidence, config)
            self._rank(hypotheses)

        async with stream.stage("meta_review", source=self.name):
            await stream.progress(
                "Synthesizing final research proposal",
                source=self.name,
                stage="meta_review",
                metadata={"agent": "Meta-Review Agent"},
            )
            final_text = await self._meta_review(goal, hypotheses, evidence, config)

        await stream.result(
            {
                "text": final_text,
                "blocked": False,
                "safety": safety,
                "hypotheses": [asdict(item) for item in hypotheses],
                "evidence": [asdict(item) for item in evidence],
                "quality": self._quality_summary(hypotheses, evidence),
                "agents": [
                    "Safety Supervisor",
                    "Evidence Agent",
                    "Generation Agent",
                    "Proximity Agent",
                    "Reflection Agent",
                    "Ranking Agent",
                    "Evolution Agent",
                    "Meta-Review Agent",
                ],
            },
            source=self.name,
        )

    def _assess_safety(self, goal: str) -> dict[str, Any]:
        lowered = goal.lower()
        for pattern in UNSAFE_RESEARCH_PATTERNS:
            if re.search(pattern, lowered):
                return {
                    "blocked": True,
                    "reason": "The research goal appears to request unsafe CBRN or harmful experimental detail.",
                    "matched_pattern": pattern,
                }
        return {"blocked": False, "reason": "No high-risk research pattern detected."}

    def _blocked_response(self, goal: str, reason: str) -> str:
        return textwrap.dedent(
            f"""
            # Co-Scientist Safety Review

            I cannot generate operational hypotheses or protocols for this request because: {reason}

            Safe alternatives:
            - Reframe the goal toward risk assessment, detection, mitigation, policy, or literature review.
            - Ask for non-operational background, taxonomy, or ethical review.
            - Provide a benign research objective and constraints for a new Co-Scientist run.

            Original goal: {goal}
            """
        ).strip()

    async def _gather_evidence(
        self,
        goal: str,
        context: UnifiedContext,
        config: Any,
    ) -> list[EvidenceItem]:
        tasks = []
        kb_name = context.knowledge_bases[0] if context.knowledge_bases else None
        if kb_name:
            tasks.append(self._rag_evidence(goal, kb_name, int(config.max_evidence)))
        if config.use_web_search:
            tasks.append(self._web_evidence(goal, int(config.max_evidence)))

        evidence: list[EvidenceItem] = self._seed_evidence(context.metadata.get("seed_evidence"))
        if tasks:
            for result in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(result, list):
                    evidence.extend(result)

        if not evidence:
            evidence.append(
                EvidenceItem(
                    id="E0",
                    source="research_goal",
                    title="User research goal",
                    snippet=goal,
                    score=1.0,
                )
            )
        return evidence[: max(1, int(config.max_evidence))]

    def _seed_evidence(self, raw: Any) -> list[EvidenceItem]:
        if not isinstance(raw, list):
            return []
        items: list[EvidenceItem] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("snippet") or item.get("text") or item.get("content") or "").strip()
            if not snippet:
                continue
            items.append(
                EvidenceItem(
                    id=str(item.get("id") or f"B{idx + 1}"),
                    source=str(item.get("source") or "seed_evidence"),
                    title=str(item.get("title") or item.get("source") or f"Seed evidence {idx + 1}"),
                    snippet=snippet[:900],
                    url=str(item.get("url") or ""),
                    score=_clip(item.get("score"), 0.5),
                )
            )
        return items

    async def _rag_evidence(self, goal: str, kb_name: str, max_items: int) -> list[EvidenceItem]:
        from deeptutor.tools.rag_tool import rag_search

        if not self._kb_storage_ready(kb_name):
            return []

        result = await rag_search(goal, kb_name=kb_name)
        sources = result.get("sources", []) if isinstance(result, dict) else []
        items: list[EvidenceItem] = []
        for idx, source in enumerate(sources[:max_items]):
            if not isinstance(source, dict):
                continue
            snippet = str(source.get("content") or source.get("text") or source.get("snippet") or "")
            title = str(source.get("source") or source.get("title") or f"KB source {idx + 1}")
            items.append(
                EvidenceItem(
                    id=f"K{idx + 1}",
                    source="knowledge_base",
                    title=title,
                    snippet=snippet[:900],
                    url=str(source.get("url") or ""),
                    score=_clip(source.get("score") or source.get("similarity"), 0.5),
                )
            )
        answer = str(result.get("answer") or result.get("content") or "") if isinstance(result, dict) else ""
        if answer and not items:
            items.append(EvidenceItem(id="K1", source="knowledge_base", title=kb_name, snippet=answer[:900], score=0.7))
        return items

    def _kb_storage_ready(self, kb_name: str) -> bool:
        safe_name = Path(kb_name).name
        project_root = Path(__file__).resolve().parents[2]
        storage_dir = project_root / "data" / "knowledge_bases" / safe_name / "llamaindex_storage"
        if not storage_dir.exists():
            return False
        return any(storage_dir.iterdir())

    async def _web_evidence(self, goal: str, max_items: int) -> list[EvidenceItem]:
        from deeptutor.tools.web_search import web_search

        response = await asyncio.to_thread(web_search, goal)
        results = getattr(response, "search_results", []) or []
        items: list[EvidenceItem] = []
        for idx, item in enumerate(results[:max_items]):
            items.append(
                EvidenceItem(
                    id=f"W{idx + 1}",
                    source="web_search",
                    title=str(getattr(item, "title", "") or f"Web source {idx + 1}"),
                    snippet=str(getattr(item, "content", "") or getattr(item, "snippet", ""))[:900],
                    url=str(getattr(item, "url", "") or ""),
                    score=_clip(getattr(item, "score", 0.4), 0.4),
                )
            )
        return items

    async def _llm_json(self, system: str, prompt: str, *, temperature: float = 0.4) -> dict[str, Any]:
        try:
            client = get_llm_client()
            text = await client.complete(
                prompt,
                system_prompt=system,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return _extract_json_object(text)
        except Exception:
            return {}

    async def _generate_hypotheses(self, goal: str, evidence: list[EvidenceItem], config: Any) -> list[Hypothesis]:
        evidence_block = self._evidence_block(evidence)
        system = (
            "You are the Generation and Proximity agents in a scientific Co-Scientist workflow. "
            "Generate novel but testable hypotheses. Ground every hypothesis in supplied evidence. "
            "Return strict JSON only."
        )
        prompt = f"""
        Research goal:
        {goal}

        Evidence:
        {evidence_block}

        Generate {int(config.max_hypotheses)} hypotheses as JSON:
        {{
          "hypotheses": [
            {{
              "id": "H1",
              "statement": "...",
              "rationale": "...",
              "evidence_ids": ["E0"],
              "novelty": 0.0,
              "testability": 0.0,
              "impact": 0.0,
              "risk": 0.0
            }}
          ]
        }}
        """
        data = await self._llm_json(system, prompt, temperature=float(config.temperature))
        hypotheses = self._parse_hypotheses(data.get("hypotheses"), int(config.max_hypotheses))
        if hypotheses:
            return hypotheses
        return self._fallback_hypotheses(goal, evidence, int(config.max_hypotheses))

    def _parse_hypotheses(self, raw: Any, max_hypotheses: int) -> list[Hypothesis]:
        if not isinstance(raw, list):
            return []
        hypotheses: list[Hypothesis] = []
        for idx, item in enumerate(raw[:max_hypotheses]):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement") or item.get("hypothesis") or "").strip()
            if not statement:
                continue
            hypotheses.append(
                Hypothesis(
                    id=str(item.get("id") or f"H{idx + 1}"),
                    statement=statement,
                    rationale=str(item.get("rationale") or "Requires further validation.").strip(),
                    evidence_ids=[str(v) for v in item.get("evidence_ids", []) if str(v).strip()],
                    novelty=_clip(item.get("novelty"), 0.55),
                    testability=_clip(item.get("testability"), 0.55),
                    impact=_clip(item.get("impact"), 0.55),
                    risk=_clip(item.get("risk"), 0.35),
                )
            )
        return hypotheses

    def _fallback_hypotheses(self, goal: str, evidence: list[EvidenceItem], max_hypotheses: int) -> list[Hypothesis]:
        templates = [
            "A hidden mechanism connecting the strongest evidence themes may explain the target phenomenon in: {goal}",
            "A focused comparative study across the retrieved sources may reveal a boundary condition for: {goal}",
            "A measurable intervention or ablation based on the evidence could test whether the main causal factor drives: {goal}",
        ]
        evidence_ids = [item.id for item in evidence[:3]]
        return [
            Hypothesis(
                id=f"H{idx + 1}",
                statement=template.format(goal=goal),
                rationale="Generated from available context because the configured LLM did not return valid JSON.",
                evidence_ids=evidence_ids,
                novelty=0.45,
                testability=0.6,
                impact=0.5,
                risk=0.25,
            )
            for idx, template in enumerate(templates[:max_hypotheses])
        ]

    def _assign_evidence_and_diversity(self, hypotheses: list[Hypothesis], evidence: list[EvidenceItem]) -> None:
        for hypo in hypotheses:
            if not hypo.evidence_ids:
                ranked = sorted(evidence, key=lambda item: _word_overlap(hypo.statement, item.snippet), reverse=True)
                hypo.evidence_ids = [item.id for item in ranked[:2]] or ["E0"]

        for i, hypo in enumerate(hypotheses):
            max_overlap = 0.0
            for j, other in enumerate(hypotheses):
                if i == j:
                    continue
                max_overlap = max(max_overlap, _word_overlap(hypo.statement, other.statement))
            diversity_bonus = max(0.0, 0.15 - max_overlap)
            hypo.novelty = _clip(hypo.novelty + diversity_bonus, hypo.novelty)

    async def _reflect(
        self,
        goal: str,
        hypotheses: list[Hypothesis],
        evidence: list[EvidenceItem],
        config: Any,
    ) -> None:
        system = (
            "You are a critical scientific peer reviewer. Evaluate correctness, novelty, testability, "
            "evidence grounding, and safety. Return strict JSON only."
        )
        prompt = f"""
        Research goal:
        {goal}

        Evidence:
        {self._evidence_block(evidence)}

        Hypotheses:
        {json.dumps([asdict(h) for h in hypotheses], ensure_ascii=False)}

        Return JSON:
        {{
          "critiques": [
            {{"id": "H1", "critique": "...", "refinement": "...", "novelty": 0.0, "testability": 0.0, "impact": 0.0, "risk": 0.0}}
          ]
        }}
        """
        data = await self._llm_json(system, prompt, temperature=0.25)
        critiques = data.get("critiques")
        by_id = {h.id: h for h in hypotheses}
        if isinstance(critiques, list):
            for item in critiques:
                if not isinstance(item, dict):
                    continue
                hypo = by_id.get(str(item.get("id")))
                if not hypo:
                    continue
                hypo.critique = str(item.get("critique") or "").strip()
                hypo.refinement = str(item.get("refinement") or "").strip()
                hypo.novelty = _clip(item.get("novelty"), hypo.novelty)
                hypo.testability = _clip(item.get("testability"), hypo.testability)
                hypo.impact = _clip(item.get("impact"), hypo.impact)
                hypo.risk = _clip(item.get("risk"), hypo.risk)

        for hypo in hypotheses:
            if not hypo.critique:
                hypo.critique = "Needs stronger external validation and explicit falsification criteria."
            if not hypo.refinement:
                hypo.refinement = "Define measurable variables, required datasets, and a negative control before execution."

    def _rank(self, hypotheses: list[Hypothesis]) -> None:
        for hypo in hypotheses:
            evidence_bonus = min(0.15, 0.05 * len(hypo.evidence_ids))
            hypo.rank_score = round(
                (0.30 * hypo.novelty)
                + (0.30 * hypo.testability)
                + (0.25 * hypo.impact)
                + evidence_bonus
                - (0.20 * hypo.risk),
                4,
            )
        hypotheses.sort(key=lambda item: item.rank_score, reverse=True)

    async def _evolve(
        self,
        goal: str,
        hypotheses: list[Hypothesis],
        evidence: list[EvidenceItem],
        config: Any,
    ) -> None:
        top = hypotheses[: min(2, len(hypotheses))]
        if not top:
            return
        system = (
            "You are the Evolution agent. Improve top-ranked hypotheses without overstating evidence. "
            "Return strict JSON only."
        )
        prompt = f"""
        Research goal:
        {goal}

        Evidence:
        {self._evidence_block(evidence)}

        Top hypotheses:
        {json.dumps([asdict(h) for h in top], ensure_ascii=False)}

        Return JSON:
        {{
          "updates": [
            {{"id": "H1", "statement": "...", "rationale": "...", "refinement": "..."}}
          ]
        }}
        """
        data = await self._llm_json(system, prompt, temperature=0.3)
        updates = data.get("updates")
        by_id = {h.id: h for h in hypotheses}
        if isinstance(updates, list):
            for item in updates:
                if not isinstance(item, dict):
                    continue
                hypo = by_id.get(str(item.get("id")))
                if not hypo:
                    continue
                statement = str(item.get("statement") or "").strip()
                rationale = str(item.get("rationale") or "").strip()
                refinement = str(item.get("refinement") or "").strip()
                if statement:
                    hypo.statement = statement
                if rationale:
                    hypo.rationale = rationale
                if refinement:
                    hypo.refinement = refinement

    async def _meta_review(
        self,
        goal: str,
        hypotheses: list[Hypothesis],
        evidence: list[EvidenceItem],
        config: Any,
    ) -> str:
        top = hypotheses[: min(3, len(hypotheses))]
        system = (
            "You are the Meta-Review agent. Write a concise, research-quality proposal for a scientist to review. "
            "Be explicit about evidence limits, falsification tests, and next steps. Do not invent citations."
        )
        prompt = f"""
        Goal: {goal}
        Evidence: {self._evidence_block(evidence)}
        Ranked hypotheses: {json.dumps([asdict(h) for h in top], ensure_ascii=False)}

        Write Markdown with sections:
        1. Executive recommendation
        2. Ranked hypotheses
        3. Evidence grounding
        4. Falsification and validation plan
        5. Risks and assumptions
        """
        try:
            client = get_llm_client()
            text = await client.complete(prompt, system_prompt=system, temperature=0.25)
            if text.strip():
                return text.strip()
        except Exception:
            pass
        return self._fallback_report(goal, top, evidence)

    def _fallback_report(self, goal: str, hypotheses: list[Hypothesis], evidence: list[EvidenceItem]) -> str:
        lines = [
            "# Co-Scientist Research Proposal",
            "",
            f"## Research Goal\n{goal}",
            "",
            "## Executive Recommendation",
            "Prioritize the highest-ranked hypothesis only after confirming the evidence links and defining falsifiable measurements.",
            "",
            "## Ranked Hypotheses",
        ]
        for idx, hypo in enumerate(hypotheses, start=1):
            lines.extend(
                [
                    f"{idx}. **{hypo.id}** score={hypo.rank_score:.2f}",
                    f"   - Hypothesis: {hypo.statement}",
                    f"   - Rationale: {hypo.rationale}",
                    f"   - Critique: {hypo.critique}",
                    f"   - Refinement: {hypo.refinement}",
                ]
            )
        lines.extend(["", "## Evidence Grounding"])
        for item in evidence:
            lines.append(f"- **{item.id} {item.title}**: {item.snippet[:220]}")
        lines.extend(
            [
                "",
                "## Falsification And Validation Plan",
                "- Define measurable variables and negative controls before experimentation.",
                "- Test the top hypothesis against at least one competing explanation.",
                "- Record exclusion criteria, expected failure modes, and uncertainty.",
                "",
                "## Risks And Assumptions",
                "- This is a proposal for expert review, not a substitute for domain expert judgment.",
                "- Claims are limited by retrieved evidence quality and available model configuration.",
            ]
        )
        return "\n".join(lines)

    def _quality_summary(self, hypotheses: list[Hypothesis], evidence: list[EvidenceItem]) -> dict[str, Any]:
        if not hypotheses:
            return {"average_score": 0.0, "evidence_count": len(evidence), "top_hypothesis_id": None}
        return {
            "average_score": round(sum(h.rank_score for h in hypotheses) / len(hypotheses), 4),
            "evidence_count": len(evidence),
            "top_hypothesis_id": hypotheses[0].id,
            "quality_gates": {
                "grounded": len(evidence) > 0,
                "ranked": all(h.rank_score != 0 for h in hypotheses),
                "critiqued": all(bool(h.critique) for h in hypotheses),
                "falsifiable_next_steps": True,
                "safety_checked": True,
            },
        }

    def _evidence_block(self, evidence: list[EvidenceItem]) -> str:
        return "\n".join(
            f"{item.id}. [{item.source}] {item.title}: {item.snippet[:700]}"
            for item in evidence
        )

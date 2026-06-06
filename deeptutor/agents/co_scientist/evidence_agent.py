"""
EvidenceAgent — gathers grounding evidence for a research goal.

Calls RAG, paper_search, and/or web_search in parallel then deduplicates
and ranks the results. Returns a list of EvidenceItem.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Awaitable

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, EvidenceItem, Hypothesis


class EvidenceAgent(BaseAgent):
    """Gathers and ranks grounding evidence from multiple sources."""

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
            agent_name="evidence_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process(  # type: ignore[override]
        self,
        *,
        cs_config: CoScientistConfig,
        call_tool: Callable[[str, str], Awaitable[str]],
        progress_cb: Callable[[str, str], None] | None = None,
    ) -> list[EvidenceItem]:
        """
        Gather evidence from all enabled sources and return deduplicated items.

        Args:
            cs_config: Co-Scientist run configuration.
            call_tool: Async callable (tool_type, query) → raw JSON string.
            progress_cb: Optional (stage, message) callback for streaming UI.
        """
        goal = cs_config.goal
        evidence: list[EvidenceItem] = []
        evidence_counter = [0]

        def _next_id() -> str:
            evidence_counter[0] += 1
            return f"E{evidence_counter[0]}"

        # --- Seed evidence from bridge (bridge_hybrid) ---
        for item in cs_config.seed_evidence:
            evidence.append(EvidenceItem(
                id=_next_id(),
                source=item.get("source", "bridge_hybrid"),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("url", ""),
                score=float(item.get("score", 0.0)),
            ))

        if progress_cb:
            progress_cb("evidence", f"Loaded {len(evidence)} seed evidence items from knowledge base.")

        # --- Parallel external tool calls ---
        tasks: list[asyncio.Task] = []
        labels: list[str] = []

        if cs_config.use_rag:
            tasks.append(asyncio.create_task(call_tool("rag", goal)))
            labels.append("rag")
        if cs_config.use_paper_search:
            tasks.append(asyncio.create_task(call_tool("paper_search", goal)))
            labels.append("paper_search")
        if cs_config.use_web_search:
            tasks.append(asyncio.create_task(call_tool("web_search", goal)))
            labels.append("web_search")
        if getattr(cs_config, "use_graphiti", False):
            tasks.append(asyncio.create_task(call_tool("graphiti", goal)))
            labels.append("graphiti")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for label, raw in zip(labels, results):
                if isinstance(raw, Exception):
                    self.logger.warning(f"EvidenceAgent: {label} failed — {raw}")
                    continue
                items = self._parse_tool_result(label, raw, _next_id)
                evidence.extend(items)
                if progress_cb:
                    progress_cb("evidence", f"Retrieved {len(items)} items from {label}.")

        # --- Deduplicate by title similarity ---
        evidence = self._deduplicate(evidence)

        # --- Trim to max_evidence ---
        evidence = evidence[: cs_config.max_evidence]

        if progress_cb:
            progress_cb("evidence", f"Final evidence set: {len(evidence)} items.")

        return evidence

    async def process_targeted(
        self,
        *,
        cs_config: CoScientistConfig,
        hypothesis: Hypothesis,
        call_tool: Callable[[str, str], Awaitable[str]],
        progress_cb: Callable[[str, str], None] | None = None,
        existing_evidence_count: int = 0,
    ) -> list[EvidenceItem]:
        """
        Gather new evidence specifically targeting the critique of a hypothesis.
        """
        # Formulate a targeted query based on the critique
        system_prompt = "You extract short, precise search queries from a critique to find verifying or debunking scientific literature."
        user_prompt = f"Hypothesis: {hypothesis.statement}\nCritique: {hypothesis.critique}\n\nTask: Output ONLY a precise 3-6 word search query to investigate the critique. No quotes or intro text."
        
        chunks = []
        async for chunk in self.stream_llm(user_prompt, system_prompt, stage="generate_query"):
            chunks.append(chunk)
        query = "".join(chunks).strip(' "\'')
        
        if progress_cb:
            progress_cb("targeted_evidence", f"Targeted search query: '{query}'")

        evidence: list[EvidenceItem] = []
        # Local state closure to continue numbering properly
        def _next_id() -> str:
            # Note: IDs will be appended to existing evidence
            return f"E_T{len(evidence) + existing_evidence_count + 1}"

        tasks: list[asyncio.Task] = []
        labels: list[str] = []

        if cs_config.use_paper_search:
            tasks.append(asyncio.create_task(call_tool("paper_search", query)))
            labels.append("paper_search")
        if cs_config.use_web_search:
            tasks.append(asyncio.create_task(call_tool("web_search", query)))
            labels.append("web_search")
        if getattr(cs_config, "use_graphiti", False):
            tasks.append(asyncio.create_task(call_tool("graphiti", query)))
            labels.append("graphiti")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for label, raw in zip(labels, results):
                if isinstance(raw, Exception):
                    self.logger.warning(f"EvidenceAgent targeted: {label} failed — {raw}")
                    continue
                items = self._parse_tool_result(label, raw, _next_id)
                evidence.extend(items)
                if progress_cb:
                    progress_cb("targeted_evidence", f"Retrieved {len(items)} items from {label}.")

        evidence = self._deduplicate(evidence)
        return evidence[:3] # Return top 3 targeted items

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_tool_result(
        self,
        tool_type: str,
        raw: str,
        next_id: Callable[[], str],
    ) -> list[EvidenceItem]:
        """Parse raw JSON string from a tool call into EvidenceItem list."""
        try:
            data = json.loads(raw)
        except Exception:
            return []

        items: list[EvidenceItem] = []

        if tool_type == "paper_search":
            for paper in (data.get("papers") or [])[:5]:
                if not isinstance(paper, dict):
                    continue
                items.append(EvidenceItem(
                    id=next_id(),
                    source="paper_search",
                    title=str(paper.get("title") or ""),
                    snippet=str(paper.get("abstract") or "")[:600],
                    url=str(paper.get("url") or ""),
                    doi=str(paper.get("doi") or ""),
                    year=int(paper.get("year") or 0),
                    citations=int(paper.get("citations") or 0),
                    score=0.9,  # Academic papers score higher by default
                ))

        elif tool_type == "web_search":
            # Support both structured and flat responses
            results = (
                data.get("search_results")
                or data.get("results")
                or []
            )
            for item in results[:5]:
                if not isinstance(item, dict):
                    continue
                items.append(EvidenceItem(
                    id=next_id(),
                    source="web_search",
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or item.get("content") or "")[:600],
                    url=str(item.get("url") or ""),
                    score=float(item.get("score") or 0.5),
                ))

        elif tool_type == "rag":
            answer = str(data.get("answer") or data.get("content") or "")
            if answer:
                items.append(EvidenceItem(
                    id=next_id(),
                    source="rag",
                    title="Knowledge Base Context",
                    snippet=answer[:600],
                    score=0.8,
                ))
            # Also extract individual source chunks if available
            for src in (data.get("sources") or [])[:4]:
                if isinstance(src, dict) and src.get("content"):
                    items.append(EvidenceItem(
                        id=next_id(),
                        source="rag",
                        title=str(src.get("source") or src.get("title") or "KB Chunk"),
                        snippet=str(src.get("content") or "")[:500],
                        score=float(src.get("score") or 0.7),
                    ))

        elif tool_type == "graphiti":
            # Data from graphiti is expected to be a list of dicts:
            # [{"source_entity": "...", "relation": "...", "target_entity": "...", "fact": "...", "score": 0.8}]
            results = data if isinstance(data, list) else data.get("results", [])
            for res in results[:5]:
                if not isinstance(res, dict):
                    continue
                src_ent = res.get("source_entity", "Unknown")
                rel = res.get("relation", "Unknown")
                tgt_ent = res.get("target_entity", "Unknown")
                fact = res.get("fact", "")
                
                if not fact:
                    fact = f"{src_ent} {rel} {tgt_ent}"
                    
                title = f"Knowledge Graph: {src_ent} → {tgt_ent}"
                
                items.append(EvidenceItem(
                    id=next_id(),
                    source="graphiti",
                    title=title,
                    snippet=str(fact)[:600],
                    score=float(res.get("score", 0.85))
                ))

        return items

    @staticmethod
    def _normalize_title(title: str) -> str:
        return title.lower().strip()

    def _deduplicate(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Remove near-duplicate evidence items by title similarity."""
        seen: set[str] = set()
        result: list[EvidenceItem] = []
        for item in sorted(items, key=lambda x: -x.score):
            key = self._normalize_title(item.title)
            if key and key not in seen:
                seen.add(key)
                result.append(item)
        return result


__all__ = ["EvidenceAgent"]

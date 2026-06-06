"""
EvolutionAgent — evolves hypotheses into stronger versions.

Takes the current hypotheses, their peer critiques, and newly gathered
targeted evidence, and generates evolved (v2, v3...) versions of them.
"""

from __future__ import annotations

import json
from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, EvidenceItem, Hypothesis


_SYSTEM_ZH = (
    "你是一位创新的资深研究员。你的任务是基于同行的批评意见和最新收集的证据，"
    "对原有的研究假设进行迭代升级，使其更加无懈可击、科学严谨。"
)

_SYSTEM_EN = (
    "You are an innovative senior researcher. Your task is to evolve existing hypotheses "
    "into stronger, more rigorous versions based on peer critiques and newly acquired evidence."
)


class EvolutionAgent(BaseAgent):
    """Evolves hypotheses by fixing flaws pointed out in critiques."""

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
            agent_name="evolution_agent",
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
        new_evidence: list[EvidenceItem],
        iteration: int,
        progress_cb=None,
    ) -> list[Hypothesis]:
        """
        Takes the current hypotheses and evolves them into a new set of hypotheses.
        Returns the new Hypothesis objects.
        """
        if progress_cb:
            progress_cb("evolution", f"Evolving {len(hypotheses)} hypotheses (Iteration {iteration})…")

        system_prompt = _SYSTEM_ZH if self.language == "zh" else _SYSTEM_EN
        user_prompt = self._build_prompt(cs_config.goal, hypotheses, new_evidence)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="evolve_hypotheses",
        ):
            chunks.append(chunk)
        response = "".join(chunks)

        evolved = self._parse(response, hypotheses, iteration)

        if progress_cb:
            progress_cb("evolution", f"Evolution complete. Generated {len(evolved)} refined hypotheses.")

        return evolved

    def _build_prompt(self, goal: str, hypotheses: list[Hypothesis], new_evidence: list[EvidenceItem]) -> str:
        h_lines: list[str] = []
        for h in hypotheses:
            h_lines.append(
                f"ID: {h.id}\n"
                f"Statement: {h.statement}\n"
                f"Rationale: {h.rationale}\n"
                f"Critique to address: {h.critique}\n"
                f"Suggested Refinement: {h.refinement}\n"
            )
        h_text = "\n\n".join(h_lines)
        
        e_lines = []
        for e in new_evidence:
            e_lines.append(f"[{e.id}] {e.title}\n{e.snippet[:300]}")
        e_text = "\n\n".join(e_lines) if e_lines else "(No additional evidence found)"

        return f"""
Research Goal:
{goal}

Newly Acquired Evidence (use this to address critiques if relevant):
{e_text}

Current Hypotheses & Peer Critiques:
{h_text}

Task:
Evolve EACH hypothesis into a stronger, new version. Incorporate the critique, the refinement suggestion, and the new evidence (if applicable).
Each evolved hypothesis should be significantly stronger and more testable.

For EACH evolved hypothesis, output a JSON object with:
  "parent_id"    — The ID of the original hypothesis you are evolving.
  "statement"   — The newly evolved hypothesis sentence.
  "rationale"   — 2–3 sentences explaining the evolved mechanism and how it addresses the critique.
  "evidence_ids" — The evidence IDs supporting this new version (you can include old ones + new ones).
  "novelty"     — Float 0.0–1.0.
  "testability" — Float 0.0–1.0.
  "impact"      — Float 0.0–1.0.
  "risk"        — Float 0.0–1.0.

ONLY output valid JSON with this schema:
{{
  "evolved_hypotheses": [
    {{
      "parent_id": "H1",
      "statement": "...",
      "rationale": "...",
      "evidence_ids": ["E1", "E_T1"],
      "novelty": 0.85,
      "testability": 0.8,
      "impact": 0.9,
      "risk": 0.2
    }}
  ]
}}
""".strip()

    def _parse(self, response: str, original_hypotheses: list[Hypothesis], iteration: int) -> list[Hypothesis]:
        from deeptutor.agents.research.utils.json_utils import extract_json_from_text  # type: ignore

        try:
            data = extract_json_from_text(response)
        except Exception:
            data = None

        if not isinstance(data, dict):
            self.logger.warning("EvolutionAgent: failed to parse JSON response")
            return []

        raw_list = data.get("evolved_hypotheses") or []
        
        # Build mapping of original for reference
        h_map = {h.id: h for h in original_hypotheses}

        result: list[Hypothesis] = []
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                continue
                
            parent_id = str(item.get("parent_id") or "")
            new_id = f"{parent_id}_v{iteration}" if parent_id else f"H_new_{i}"
            
            # Inherit old evidence ids as fallback, but LLM should provide the merged list
            fallback_evidence = h_map[parent_id].evidence_ids if parent_id in h_map else []
            raw_ev = item.get("evidence_ids")
            ev_ids = [str(x) for x in raw_ev] if isinstance(raw_ev, list) else fallback_evidence

            h = Hypothesis(
                id=new_id,
                statement=str(item.get("statement") or ""),
                rationale=str(item.get("rationale") or ""),
                evidence_ids=ev_ids,
                novelty=self._clamp(item.get("novelty", 0.5)),
                testability=self._clamp(item.get("testability", 0.5)),
                impact=self._clamp(item.get("impact", 0.5)),
                risk=self._clamp(item.get("risk", 0.3)),
                iteration=iteration,
                parent_id=parent_id,
            )
            result.append(h)

        # If LLM failed to evolve some, just carry over the original
        evolved_parents = {h.parent_id for h in result}
        for orig in original_hypotheses:
            if orig.id not in evolved_parents:
                orig.iteration = iteration
                result.append(orig)

        return result
        
    @staticmethod
    def _clamp(val, lo: float = 0.0, hi: float = 1.0) -> float:
        try:
            return max(lo, min(hi, float(val)))
        except Exception:
            return 0.5


__all__ = ["EvolutionAgent"]

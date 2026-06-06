"""
HypothesisAgent — generates N competing research hypotheses from evidence.

Each hypothesis has a statement, rationale, supporting evidence IDs,
and four scoring dimensions: novelty, testability, impact, risk.
"""

from __future__ import annotations

import json
from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, EvidenceItem, Hypothesis


_SYSTEM_ZH = (
    "你是一位严谨的科研助理，专门负责从证据中提炼研究假设。"
    "你必须生成科学合理、可验证、且互相有差异的假设。"
)

_SYSTEM_EN = (
    "You are a rigorous AI research partner specialized in generating "
    "scientifically sound, testable, and diverse research hypotheses from evidence."
)


class HypothesisAgent(BaseAgent):
    """Generates competing research hypotheses from a curated evidence set."""

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
            agent_name="hypothesis_agent",
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
        evidence: list[EvidenceItem],
        progress_cb=None,
    ) -> list[Hypothesis]:
        """
        Generate `cs_config.max_hypotheses` hypotheses grounded in evidence.
        Returns a list of Hypothesis objects with scored dimensions.
        """
        n = cs_config.max_hypotheses
        if progress_cb:
            progress_cb("generation", f"Generating {n} competing hypotheses…")

        evidence_block = self._format_evidence(evidence)
        system_prompt = _SYSTEM_ZH if self.language == "zh" else _SYSTEM_EN
        user_prompt = self._build_prompt(cs_config.goal, evidence_block, n)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="generate_hypotheses",
        ):
            chunks.append(chunk)
        response = "".join(chunks)

        hypotheses = self._parse(response, n)

        if progress_cb:
            progress_cb("generation", f"Generated {len(hypotheses)} hypotheses.")

        return hypotheses

    # ------------------------------------------------------------------

    @staticmethod
    def _format_evidence(evidence: list[EvidenceItem]) -> str:
        lines: list[str] = []
        for e in evidence:
            lines.append(
                f"[{e.id}] {e.title} ({e.source})\n"
                f"    {e.snippet[:300]}"
            )
        return "\n\n".join(lines) if lines else "(No evidence provided)"

    def _build_prompt(self, goal: str, evidence_block: str, n: int) -> str:
        evidence_ids = [f"E{i+1}" for i in range(30)]  # placeholder list shown to LLM
        return f"""
Research Goal:
{goal}

Available Evidence (cite by ID):
{evidence_block}

Task:
Generate exactly {n} diverse, scientifically valid research hypotheses about the goal above.
Each hypothesis must differ meaningfully (cover different mechanisms, perspectives, or variables).

For EACH hypothesis output a JSON object with:
  "statement"   — A single crisp hypothesis sentence (claim + mechanism).
  "rationale"   — 2–3 sentences explaining scientific basis and linking to evidence IDs.
  "evidence_ids" — List of evidence IDs (e.g. ["E1","E3"]) that support this hypothesis.
  "novelty"     — Float 0.0–1.0: how novel/surprising vs. existing literature.
  "testability" — Float 0.0–1.0: how feasible to test experimentally.
  "impact"      — Float 0.0–1.0: potential scientific/societal significance.
  "risk"        — Float 0.0–1.0: scientific or practical risk (higher = riskier).

ONLY output valid JSON with this schema:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "rationale": "...",
      "evidence_ids": ["E1"],
      "novelty": 0.8,
      "testability": 0.7,
      "impact": 0.9,
      "risk": 0.3
    }}
  ]
}}
""".strip()

    def _parse(self, response: str, n: int) -> list[Hypothesis]:
        from deeptutor.agents.research.utils.json_utils import extract_json_from_text  # type: ignore

        try:
            data = extract_json_from_text(response)
        except Exception:
            data = None

        if not isinstance(data, dict):
            self.logger.warning("HypothesisAgent: failed to parse JSON response")
            return []

        raw_list = data.get("hypotheses") or []
        if not isinstance(raw_list, list):
            return []

        result: list[Hypothesis] = []
        for i, item in enumerate(raw_list[:n]):
            if not isinstance(item, dict):
                continue
            h = Hypothesis(
                id=f"H{i + 1}",
                statement=str(item.get("statement") or ""),
                rationale=str(item.get("rationale") or ""),
                evidence_ids=[str(x) for x in (item.get("evidence_ids") or [])],
                novelty=_clamp(item.get("novelty", 0.5)),
                testability=_clamp(item.get("testability", 0.5)),
                impact=_clamp(item.get("impact", 0.5)),
                risk=_clamp(item.get("risk", 0.3)),
            )
            result.append(h)

        return result


def _clamp(val, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return 0.5


__all__ = ["HypothesisAgent"]

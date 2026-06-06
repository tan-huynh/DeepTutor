"""
CritiqueAgent — performs peer review and refinement on hypotheses.

In a real scientific process, hypotheses are challenged. This agent
critiques each hypothesis and then evolves (refines) it into a stronger version.
"""

from __future__ import annotations

import json
from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, Hypothesis


_SYSTEM_ZH = (
    "你是一位资深科学家，专门对研究假设进行同行评审和完善。"
    "你会指出假设的逻辑漏洞、混杂变量或不切实际之处，然后提出改进版本。"
)

_SYSTEM_EN = (
    "You are a senior scientist acting as a rigorous peer reviewer. "
    "Your job is to critique hypotheses for logical flaws, confounding variables, "
    "or impracticality, and then propose a refined, stronger version."
)


class CritiqueAgent(BaseAgent):
    """Critiques hypotheses and provides refined versions."""

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
            agent_name="critique_agent",
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
        progress_cb=None,
    ) -> list[Hypothesis]:
        """Critiques and refines the list of hypotheses in place."""
        if progress_cb:
            progress_cb("reflection", f"Critiquing {len(hypotheses)} hypotheses…")

        system_prompt = _SYSTEM_ZH if self.language == "zh" else _SYSTEM_EN

        # We process them in a single batch to save latency, assuming n is small (3-5).
        user_prompt = self._build_prompt(cs_config.goal, hypotheses)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="critique_hypotheses",
        ):
            chunks.append(chunk)
        response = "".join(chunks)

        self._apply_updates(response, hypotheses)

        if progress_cb:
            progress_cb("reflection", "Critique and refinement complete.")

        return hypotheses

    def _build_prompt(self, goal: str, hypotheses: list[Hypothesis]) -> str:
        lines: list[str] = []
        for h in hypotheses:
            lines.append(f"ID: {h.id}\nStatement: {h.statement}\nRationale: {h.rationale}")
        
        hyp_text = "\n\n".join(lines)

        return f"""
Research Goal:
{goal}

Here are competing hypotheses generated for this goal:
{hyp_text}

Task:
For EACH hypothesis, provide:
1. "critique" — A rigorous peer review (2-3 sentences) pointing out flaws, confounding factors, or testability issues.
2. "refinement" — A concrete suggestion (1-2 sentences) to make the hypothesis stronger, more precise, or safer.

ONLY output valid JSON with this schema:
{{
  "reviews": [
    {{
      "id": "H1",
      "critique": "...",
      "refinement": "..."
    }}
  ]
}}
""".strip()

    def _apply_updates(self, response: str, hypotheses: list[Hypothesis]) -> None:
        from deeptutor.agents.research.utils.json_utils import extract_json_from_text  # type: ignore

        try:
            data = extract_json_from_text(response)
        except Exception:
            data = None

        if not isinstance(data, dict):
            self.logger.warning("CritiqueAgent: failed to parse JSON response")
            return

        reviews = data.get("reviews") or []
        if not isinstance(reviews, list):
            return

        # Map ID to hypothesis
        h_map = {h.id: h for h in hypotheses}

        for rev in reviews:
            if not isinstance(rev, dict):
                continue
            hid = str(rev.get("id") or "")
            if hid in h_map:
                h_map[hid].critique = str(rev.get("critique") or "")
                h_map[hid].refinement = str(rev.get("refinement") or "")


__all__ = ["CritiqueAgent"]

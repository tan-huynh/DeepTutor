"""
RankingAgent — performs Elo-style tournament ranking between hypotheses.

In a real Co-Scientist tournament, hypotheses compete head-to-head.
This agent acts as the judge for pairwise matchups and assigns a final rank score.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

from deeptutor.agents.base_agent import BaseAgent
from .data_structures import CoScientistConfig, Hypothesis


_SYSTEM_ZH = (
    "你是一位公正的科学评委。你需要对两个竞争的研究假设进行头对头(head-to-head)对比，"
    "选出总体科学质量（创新性、可验证性、潜在影响）更高的一方。"
)

_SYSTEM_EN = (
    "You are an impartial scientific judge. You will compare two competing research "
    "hypotheses head-to-head and select the winner based on overall scientific "
    "quality (novelty, testability, potential impact)."
)


class RankingAgent(BaseAgent):
    """Runs a round-robin tournament to rank hypotheses."""

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
            agent_name="ranking_agent",
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
        """
        Runs `tournament_rounds` of full round-robin tournaments.
        Updates tournament_wins, losses, and rank_score in place.
        """
        if len(hypotheses) < 2:
            if hypotheses:
                hypotheses[0].rank_score = 1.0
            return hypotheses

        rounds = max(1, min(3, cs_config.tournament_rounds))
        total_matches = len(list(combinations(hypotheses, 2))) * rounds

        if progress_cb:
            progress_cb("ranking", f"Running tournament ranking: {rounds} rounds, {total_matches} total matches…")

        system_prompt = _SYSTEM_ZH if self.language == "zh" else _SYSTEM_EN

        # For simplicity and latency, we bundle all matches into a single prompt
        # instead of making N*(N-1)/2 sequential LLM calls.
        user_prompt = self._build_prompt(cs_config.goal, hypotheses, rounds)

        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="rank_hypotheses",
        ):
            chunks.append(chunk)
        response = "".join(chunks)

        self._apply_results(response, hypotheses, total_matches)

        if progress_cb:
            progress_cb("ranking", "Tournament complete. Hypotheses ranked.")

        return sorted(hypotheses, key=lambda x: -x.rank_score)

    def _build_prompt(self, goal: str, hypotheses: list[Hypothesis], rounds: int) -> str:
        # We present all pairs to the LLM to judge at once
        h_text = "\n\n".join(
            f"ID: {h.id}\nStatement: {h.statement}\nRationale: {h.rationale}\nRefinement: {h.refinement}"
            for h in hypotheses
        )
        
        pairs = []
        for h1, h2 in combinations(hypotheses, 2):
            pairs.append(f"{h1.id} vs {h2.id}")

        matches_text = "\n".join(pairs)

        return f"""
Research Goal:
{goal}

Hypotheses:
{h_text}

Task:
You will run a tournament. For each matchup listed below, declare a winner.
Base your judgment on the overall strength of the hypothesis (statement, rationale, and proposed refinements).
Take into account novelty, feasibility, and impact.

Matches to judge:
{matches_text}

ONLY output valid JSON with this schema:
{{
  "matches": [
    {{
      "match": "H1 vs H2",
      "winner_id": "H1",
      "reason": "..."
    }}
  ]
}}
""".strip()

    def _apply_results(self, response: str, hypotheses: list[Hypothesis], max_wins: int) -> None:
        from deeptutor.agents.research.utils.json_utils import extract_json_from_text  # type: ignore

        try:
            data = extract_json_from_text(response)
        except Exception:
            data = None

        if not isinstance(data, dict):
            self.logger.warning("RankingAgent: failed to parse JSON response")
            return

        matches = data.get("matches") or []
        
        h_map = {h.id: h for h in hypotheses}

        for match in matches:
            if not isinstance(match, dict):
                continue
            winner = str(match.get("winner_id") or "")
            match_str = str(match.get("match") or "")
            
            ids = match_str.split(" vs ")
            if len(ids) != 2:
                continue
            
            h1, h2 = ids[0].strip(), ids[1].strip()
            
            if winner == h1 and h1 in h_map and h2 in h_map:
                h_map[h1].tournament_wins += 1
                h_map[h2].tournament_losses += 1
            elif winner == h2 and h1 in h_map and h2 in h_map:
                h_map[h2].tournament_wins += 1
                h_map[h1].tournament_losses += 1

        # Calculate rank score as win ratio
        for h in hypotheses:
            total = h.tournament_wins + h.tournament_losses
            if total > 0:
                h.rank_score = h.tournament_wins / total
            else:
                h.rank_score = 0.5


__all__ = ["RankingAgent"]

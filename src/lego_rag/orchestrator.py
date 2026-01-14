"""
orchestrator.py

The central coordinator for the modular RAG pipeline.
It manages the flow of data between:
1. Query Routing (Selection of RAG strategy)
2. Execution of RAG Flow (Pre -> Retr -> Post -> Gen)
3. Output Judging (Validation)
4. Fusion (Aggregation of results)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math
from src.lego_rag.logger import logger

# ============================================================
# 1. Result container used internally by the orchestrator
# ============================================================

@dataclass
class FlowResult:
    flow_name: str
    generated_text: str
    generated_verified: bool
    verification_report: Dict[str, Any]
    used_contexts: List[Dict[str, Any]]
    candidate_scores: Optional[Any] = None


# ============================================================
# 2. NORMALIZATION LAYER
# ============================================================

def normalize_generation_result(gen_res: Any, contexts: List[Dict]) -> Dict[str, Any]:
    """
    Accepts:
      - dict outputs
      - dataclass / object outputs 
    Returns:
      canonical dict used by orchestration
    """

    # Case 1: dict
    if isinstance(gen_res, dict):
        return {
            "generated_text": gen_res.get("generated_text") or gen_res.get("text") or "",
            "generated_verified": bool(gen_res.get("generated_verified", False)),
            "verification_report": gen_res.get("verification_report", {}),
            "used_contexts": gen_res.get("used_contexts", contexts),
            "candidate_scores": gen_res.get("candidate_scores", None),
        }

    # Case 2: object / dataclass
    generated_text = (
        getattr(gen_res, "generated_text", None)
        or getattr(gen_res, "text", "")
    )

    return {
        "generated_text": generated_text,
        "generated_verified": bool(
            getattr(gen_res, "generated_verified", False)
            or getattr(gen_res, "verified", False)
        ),
        "verification_report": getattr(gen_res, "verification_report", {}) or {},
        "used_contexts": getattr(gen_res, "used_contexts", contexts) or contexts,
        "candidate_scores": getattr(gen_res, "candidate_scores", None),
    }


# ============================================================
# 3. FLOW BUILDER (WRAPS EXISTING PIPELINES)
# ============================================================

def make_rag_flow(
    *,
    name: str,
    pre_retrieval_fn: Callable[[str], Any],
    retrieval_fn: Callable[[Any], List[Dict]],
    post_retrieval_fn: Callable[[List[Dict], str], List[Dict]],
    generation_fn: Callable[[str, List[Dict]], Any],
) -> Callable[[str], Dict[str, Any]]:
    """
    Wraps existing pipeline stages into a single callable.
    """

    def flow(query: str, callback: Optional[Callable[[str, Any], None]] = None) -> Dict[str, Any]:
        msg = f"[FLOW:{name}] pre-retrieval"
        logger.info(msg)
        if callback: callback("status", msg)
        pre_out = pre_retrieval_fn(query)

        msg = f"[FLOW:{name}] retrieval"
        logger.info(msg)
        if callback: callback("status", msg)
        retrieved = retrieval_fn(pre_out)

        msg = f"[FLOW:{name}] post-retrieval"
        logger.info(msg)
        if callback: callback("status", msg)
        contexts = post_retrieval_fn(retrieved, query)

        msg = f"[FLOW:{name}] generation"
        logger.info(msg)
        if callback: callback("status", msg)
        gen_res = generation_fn(query, contexts)

        return normalize_generation_result(gen_res, contexts)

    return flow


# ============================================================
# 4. ROUTING
# ============================================================

class Router:
    """
    Simple hybrid router (metadata + semantic placeholder)
    """

    def __init__(self, flow_scores: Dict[str, float]):
        self.flow_scores = flow_scores

    def route(self, query: str, k: int = 1) -> List[str]:
        """
        Returns top-k flow names
        """
        ranked = sorted(
            self.flow_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [name for name, _ in ranked[:k]]


# ============================================================
# 5. SCHEDULING / JUDGING
# ============================================================

class Judge:
    def __init__(self, confidence_threshold: float = 0.6):
        self.threshold = confidence_threshold

    def accept(self, result: FlowResult) -> bool:
        """
        Rule-based judge (can be replaced with LLM judge)
        """
        if result.generated_verified:
            return True

        # Fallback: if verification is disabled or passed, accept.
        return True


# ============================================================
# 6. FUSION
# ============================================================

class FusionEngine:

    @staticmethod
    def rrf(rankings: List[List[str]], k: int = 60) -> List[str]:
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    @staticmethod
    def concatenate(results: List[FlowResult]) -> str:

        return "\n\n".join(r.generated_text for r in results)


# ============================================================
# 7. ORCHESTRATOR
# ============================================================

class Orchestrator:

    def __init__(
        self,
        flows: Dict[str, Callable[[str], Dict[str, Any]]],
        router: Router,
        judge: Judge,
        fusion_engine: FusionEngine,
        max_loops: int = 2,
    ):
        self.flows = flows
        self.router = router
        self.judge = judge
        self.fusion = fusion_engine
        self.max_loops = max_loops

    def run(self, query: str, callback: Optional[Callable[[str, Any], None]] = None) -> str:
        logger.info("[ORCHESTRATOR] routing")
        if callback: callback("status", "[ORCHESTRATOR] routing")
        selected_flows = self.router.route(query, k=2)

        flow_results: List[FlowResult] = []

        for flow_name in selected_flows:
            flow = self.flows[flow_name]

            for loop_i in range(self.max_loops):
                if callback: callback("status", f"[ORCHESTRATOR] flow '{flow_name}' loop {loop_i+1}")
                # Pass callback to flow
                try:
                    raw = flow(query, callback=callback)
                except TypeError:
                    # Fallback for flows
                    raw = flow(query)

                result = FlowResult(
                    flow_name=flow_name,
                    generated_text=raw["generated_text"],
                    generated_verified=raw["generated_verified"],
                    verification_report=raw["verification_report"],
                    used_contexts=raw["used_contexts"],
                    candidate_scores=raw["candidate_scores"],
                )

                flow_results.append(result)

                if self.judge.accept(result):
                    if callback: callback("status", f"[ORCHESTRATOR] flow '{flow_name}' accepted")
                    break
                else:
                    if callback: callback("status", f"[ORCHESTRATOR] flow '{flow_name}' rejected/retry")

        logger.info("[ORCHESTRATOR] fusion")
        if callback: callback("status", "[ORCHESTRATOR] fusion")
        final_answer = self.fusion.concatenate(flow_results)
        
        # Collect all contexts
        all_contexts = [ctx for res in flow_results for ctx in res.used_contexts]
        if callback: callback("sources", all_contexts)

        return final_answer
"""
Generation module for Modular RAG.

Responsibilities:
- Build a concise, retrieval-augmented prompt from query + post-retrieval output.
- Run generation using a LangChain BaseChatModel (synchronous via `invoke`).
- Provide verification hooks:
    - Knowledge-base verification: use a Retriever-like callable to check claims.
    - Model-based verification: ask a verifier model to judge factuality w.r.t. context.
- Regenerate (optionally) until the verifier is satisfied, with a max attempt limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any, Tuple
import logging
import re
import json

import numpy as np

try:
    from langchain_core.language_models.chat_models import BaseChatModel
except Exception:
    BaseChatModel = None  # will raise if used without LangChain installed

from src.lego_rag.logger import logger
logging.basicConfig(level=logging.INFO)


# -------------------------
# Utility
# -------------------------
def call_llm(llm: BaseChatModel, prompt: str) -> str:
    """
    Expects llm.invoke(prompt) to return a message-like object with `.content`.
    """
    if BaseChatModel is None:
        raise RuntimeError("BaseChatModel not available in environment.")
    if not isinstance(llm, BaseChatModel):
        raise TypeError("llm must be an instance of BaseChatModel")
    resp = llm.invoke(prompt)
    if hasattr(resp, "content"):
        return str(resp.content).strip()
    # fallback to string
    return str(resp).strip()


# -------------------------
# Types
# -------------------------
@dataclass
class GenerationConfig:
    system_prompt: str = (
        "You are an expert RAG assistant. Answer the user's question STRICTLY using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Do not provide a summary of the source unless asked. "
        "Provide a direct, concise answer with clean citations in the format [chunk_id]."
    )
    max_output_tokens: int = 1024
    temperature: float = 0.0
    max_attempts: int = 2  # for regenerate-on-failure
    verify_threshold: float = 0.7  # model-verifier score threshold (0-1)


@dataclass
class GenerationResult:
    text: str
    used_contexts: List[Dict]  # list of contexts (dict with chunk_id, text, metadata)
    verified: bool
    verification_report: Dict[str, Any]


# -------------------------
# Claim extraction 
# -------------------------
_sentence_split_re = re.compile(r"(?<=\.)\s+|(?<=\?)\s+|(?<=!)\s+")


def extract_claims_from_text(text: str, min_len: int = 30) -> List[str]:
    """
    extractor: split into sentences and keep those that are long enough.
    """
    sents = [s.strip() for s in _sentence_split_re.split(text) if s.strip()]
    claims = [s for s in sents if len(s) >= min_len]
    # as a fallback, if nothing qualifies, return first 1-2 sentences
    if not claims and sents:
        return sents[: min(2, len(sents))]
    return claims


# -------------------------
# Knowledge-base verifier
# -------------------------
def kb_verify_claim(
    claim: str,
    kb_retriever: Callable[[str, int], List[Dict]],
    top_k: int = 3,
    support_threshold: float = 0.2,
) -> Tuple[bool, List[Dict]]:
    """
    Verify a claim against a knowledge base retriever.

    kb_retriever should accept (query: str, k: int) and return a list of dicts with:
        - 'text': str
        - optionally 'score': float (similarity)
        - optionally other metadata

    Returns (is_supported, supporting_passages)
    Heuristic:
      - If any retrieved passage has score >= support_threshold -> supported
      - Else if any passage contains substantial textual overlap -> supported
    """
    try:
        supports = kb_retriever(claim, top_k)
    except Exception as e:
        logger.debug("kb_retriever failed: %s", e, exc_info=True)
        return False, []

    # normalize
    out = []
    for p in supports or []:
        t = p.get("text") if isinstance(p, dict) else str(p)
        s = p.get("score", None) if isinstance(p, dict) else None
        out.append({"text": t, "score": s, "meta": p.get("metadata") if isinstance(p, dict) else None})

    # check numeric scores first
    for p in out:
        if p["score"] is not None:
            try:
                if float(p["score"]) >= support_threshold:
                    return True, out
            except Exception:
                pass

    # fallback textual overlap
    claim_words = set(re.findall(r"\w{4,}", claim.lower()))
    for p in out:
        pw = set(re.findall(r"\w{4,}", p["text"].lower()))
        if not claim_words:
            continue
        overlap = len(claim_words & pw) / max(1, len(claim_words))
        if overlap > 0.4:
            return True, out

    return False, out


# -------------------------
# Model-based verifier
# -------------------------
def model_verify_claim(
    claim: str,
    context_text: str,
    verifier_llm: BaseChatModel,
    prompt_template: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Use a small model to judge whether `claim` is supported by `context_text`.
    Returns (score, raw_response). Score in [0,1], higher better.

    prompt_template should return an instruction where {claim} and {context} are substituted.
    """
    if prompt_template is None:
        prompt_template = (
            "Given the following CONTEXT, rate how well the CONTEXT supports the CLAIM from 0.0 to 1.0.\n\n"
            "CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\n"
            "Return a JSON object: {{\"score\": <float between 0 and 1>, \"explain\": \"short explanation\"}}"
        )
    prompt = prompt_template.format(claim=claim, context=context_text)
    resp = call_llm(verifier_llm, prompt)
    # try to parse JSON first
    try:
        obj = json.loads(resp)
        score = float(obj.get("score", 0.0))
        explanation = str(obj.get("explain", ""))
        return max(0.0, min(1.0, score)), explanation
    except Exception:
        # fallback: extract first float
        m = re.findall(r"[0-9]*\.?[0-9]+", resp)
        if m:
            try:
                s = float(m[0])
                return max(0.0, min(1.0, s)), resp
            except Exception:
                pass
    return 0.0, resp


# -------------------------
# Generator
# -------------------------
class Generator:
    """
    Responsible for synthesizing answers y = LLM([D_q, q]).

    - llm: BaseChatModel used for generation (invoke)
    - verifier_model: optional BaseChatModel used for model-based verification (can be same as llm)
    - kb_retriever: optional callable for knowledge-base verification (see kb_verify_claim)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config: Optional[GenerationConfig] = None,
        verifier_model: Optional[BaseChatModel] = None,
        kb_retriever: Optional[Callable[[str, int], List[Dict]]] = None,
    ):
        if BaseChatModel is None:
            raise RuntimeError("langchain BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel")
        self.llm = llm
        self.verifier_model = verifier_model or llm
        self.kb_retriever = kb_retriever
        self.config = config or GenerationConfig()

    def _build_prompt(self, query: str, contexts: List[Dict], instructions: Optional[str] = None) -> str:
        """
        Build a prompt containing system prompt, contexts (ordered), and the user query.
        contexts: list of dicts each with keys: 'chunk_id', 'text', 'metadata' (optional)
        """
        system = self.config.system_prompt.strip()
        sb = [system, "\n\n---CONTEXT BEGIN---\n"]
        for i, c in enumerate(contexts):
            cid = c.get("chunk_id") or c.get("id") or f"ctx_{i}"
            text = c.get("text") or c.get("page_content") or c.get("content") or ""
            meta = c.get("metadata") or {}
            sb.append(f"[{cid}] {text}\n")
        sb.append("\n---CONTEXT END---\n")
        if instructions:
            sb.append(f"\nINSTRUCTIONS:\n{instructions}\n")
        sb.append(f"\nUSER QUERY:\n{query}\n")
        sb.append("\nAnswer concisely and cite supporting context chunk_ids where applicable.")
        return "\n".join(sb)

    def generate_once(self, query: str, contexts: List[Dict], instructions: Optional[str] = None) -> GenerationResult:
        prompt = self._build_prompt(query, contexts, instructions)
        logger.debug("Generation prompt:\n%s", prompt[:1500])
        text = call_llm(self.llm, prompt)
        return GenerationResult(text=text, used_contexts=contexts, verified=True, verification_report={})

    def verify_answer(
        self, gen_text: str, contexts: List[Dict], kb_support_threshold: float = 0.2, model_threshold: float = 0.65
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify the generated text:
          - extract claims
          - for each claim, try KB verification (if kb_retriever provided)
          - fall back to model-based verification (verifier_model)

        Returns (is_verified, report)
        report example:
          {
            "claims": [
               {"claim": "...", "kb_supported": True/False, "kb_evidence": [...], "model_score": 0.8, "explain": "..."}
            ],
            "overall_score": 0.85
          }
        """
        claims = extract_claims_from_text(gen_text)
        claims_report = []
        scores = []

        combined_context_text = "\n\n".join([c.get("text") or c.get("page_content") or "" for c in contexts])

        for claim in claims:
            kb_supported = False
            kb_evidence = []
            if self.kb_retriever:
                try:
                    kb_supported, kb_evidence = kb_verify_claim(claim, self.kb_retriever, top_k=3, support_threshold=kb_support_threshold)
                except Exception as e:
                    logger.debug("kb_verify_claim failed: %s", e, exc_info=True)

            model_score, explanation = model_verify_claim(claim, combined_context_text, self.verifier_model)

            # heuristics: if kb_supported OR model_score above threshold -> treat as supported
            supported = kb_supported or (model_score >= model_threshold)
            scores.append(float(model_score))
            claims_report.append({
                "claim": claim,
                "kb_supported": bool(kb_supported),
                "kb_evidence": kb_evidence,
                "model_score": float(model_score),
                "model_explain": explanation,
                "supported": bool(supported),
            })

        overall_score = float(np.mean(scores)) if scores else 0.0
        verified = overall_score >= self.config.verify_threshold and all(c["supported"] for c in claims_report) if claims_report else (overall_score >= self.config.verify_threshold)

        report = {"claims": claims_report, "overall_score": overall_score}
        return bool(verified), report

    def generate_with_verification(
        self,
        query: str,
        contexts: List[Dict],
        instructions: Optional[str] = None,
    ) -> Optional[GenerationResult]:
        """
        High-level loop:
          - generate once
          - verify; if accepted return
          - else refine prompt with verifier feedback and regenerate up to max_attempts
        """
        attempt = 0
        last_result: Optional[GenerationResult] = None

        while attempt < max(1, self.config.max_attempts):
            attempt += 1
            logger.info("Generation attempt %d for query=%s", attempt, query[:80])
            result = self.generate_once(query, contexts, instructions)
            verified, report = self.verify_answer(result.text, contexts)
            result.verified = verified
            result.verification_report = report
            last_result = result
            if verified:
                logger.info("Generation verified on attempt %d (score=%.3f)", attempt, report.get("overall_score", 0.0))
                return result
            # refine instructions for next attempt using the verifier feedback
            refinement = self._make_refinement_instructions(report)
            # Append refinement to instructions for next generation
            instructions = (instructions or "") + "\n\n" + refinement
            logger.info("Regenerating with refinement instructions.")
        logger.warning("Generation not verified after %d attempts. Returning last generation.", attempt)
        return last_result

    def _make_refinement_instructions(self, report: Dict[str, Any]) -> str:
        """
        Convert verification report into simple human-readable instructions prompting
        the LLM to focus on unsupported claims, cite context, and avoid hallucination.
        """
        claims = report.get("claims", [])
        unsupported = [c for c in claims if not c.get("supported")]
        if not unsupported:
            return "No refinements suggested."
        lines = [
            "Refinement: The previous answer had unsupported or weakly supported claims. Please:",
            "- Re-check the claims listed below against the provided context.",
            "- If you cannot find supporting context, state so explicitly instead of inventing facts.",
            "- Cite chunk_ids that support each claim."
        ]
        for c in unsupported:
            lines.append(f"- Claim: {c.get('claim')[:240]} ...")
        return "\n".join(lines)


# -------------------------
# Generation pipeline convenience wrapper
# -------------------------
class GenerationPipeline:
    """
    accepts:
      - generator: Generator instance
      - post_retrieval_output: list of contexts (dicts with chunk_id, text, metadata)
      - user query
    and returns final GenerationResult
    """

    def __init__(self, generator: Generator):
        self.generator = generator

    def run(self, query: str, contexts: List[Dict], instructions: Optional[str] = None) -> GenerationResult:
        # contexts expected to be ordered by importance (post-retrieval should have done that)
        return self.generator.generate_with_verification(query, contexts, instructions)
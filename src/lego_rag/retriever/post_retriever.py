"""
Post-retrieval module for Modular RAG

Components:
 - Reranking (MMR + LLM-based)
 - Selection (LLM-based + embedding fallback)
 - Compression (LLM + heuristic)
 - Contradiction detection (LLM-assisted)
 - PostRetrievalPipeline orchestrator

 - retrieved: List[{"chunk_id": str, "score": float, "example_doc": Document}]
 - embedding_model: implements embed_documents / embed_query
 - llm: LangChain BaseChatModel
"""

from typing import List, Dict, Optional, Any
import json
import re
from src.lego_rag.logger import logger
import numpy as np

from langchain_core.documents import Document

from src.lego_rag.logger import logger

# Attempt to import BaseChatModel; require it for call_llm usage
try:
    from langchain_core.language_models.chat_models import BaseChatModel
except Exception:
    BaseChatModel = None


# ---------------------------------------------------------------------
# Canonical LLM call wrapper
# ---------------------------------------------------------------------
def call_llm(llm: BaseChatModel, prompt: str) -> str:
    """
    Canonical way to call a LangChain chat model (synchronous).
    Expects llm to be an instance of BaseChatModel and to support `invoke`.
    """
    if BaseChatModel is None:
        raise RuntimeError("BaseChatModel not available in environment. Install the LangChain core package.")
    if not isinstance(llm, BaseChatModel):
        raise TypeError("llm must be an instance of BaseChatModel")
    resp = llm.invoke(prompt)
    # resp should be a BaseMessage-like object with `.content`
    if hasattr(resp, "content"):
        return str(resp.content).strip()
    # fallback: string representation
    return str(resp).strip()


# ---------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------
def _safe_embed(embedding_model: Any, texts: List[str], batch: int = 32) -> List[List[float]]:
    out = []
    if hasattr(embedding_model, "embed_documents"):
        for i in range(0, len(texts), batch):
            out.extend(embedding_model.embed_documents(texts[i:i + batch]))
    elif hasattr(embedding_model, "embed_query"):
        for t in texts:
            out.append(embedding_model.embed_query(t))
    else:
        raise ValueError("embedding_model must implement embed_documents or embed_query")
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------
class Reranker:
    def __init__(self, embedding_model: Any, llm: BaseChatModel, mmr_lambda: float = 0.7):
        if BaseChatModel is None:
            raise RuntimeError("BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Reranker requires llm to be a BaseChatModel instance.")
        self.embedding = embedding_model
        self.llm = llm
        self.lmbda = mmr_lambda

    def mmr_rerank(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        texts = [c["example_doc"].page_content for c in candidates]
        q_vec = np.array(_safe_embed(self.embedding, [query])[0])
        doc_vecs = np.array(_safe_embed(self.embedding, texts))

        sims_q = [cosine(q_vec, d) for d in doc_vecs]
        sims_dd = doc_vecs @ doc_vecs.T

        selected = []
        used = set()

        for _ in range(min(k, len(candidates))):
            scores = []
            for i in range(len(candidates)):
                if i in used:
                    scores.append(-1e9)
                    continue
                diversity = max(sims_dd[i, j] for j in used) if used else 0.0
                score = self.lmbda * sims_q[i] - (1 - self.lmbda) * diversity
                scores.append(score)

            best = int(np.argmax(scores))
            used.add(best)
            selected.append(candidates[best])

        return selected

    def model_rerank(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        scored = []
        for c in candidates:
            prompt = (
                "Rate relevance from 0.0 to 1.0.\n"
                f"Query: {query}\nPassage: {c['example_doc'].page_content}\nScore:"
            )
            s_txt = call_llm(self.llm, prompt)
            s = self._parse(s_txt)
            scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    @staticmethod
    def _parse(txt: str) -> float:
        try:
            v = float(re.findall(r"[0-9]*\.?[0-9]+", txt)[0])
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------
# Selector (Embedding-based fallback)
# ---------------------------------------------------------------------
class Selector:
    def __init__(self, embedding_model: Any, llm: BaseChatModel):
        if BaseChatModel is None:
            raise RuntimeError("BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Selector requires llm to be a BaseChatModel instance.")
        self.embedding = embedding_model
        self.llm = llm

    def embedding_select(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        texts = [c["example_doc"].page_content for c in candidates]
        q = np.array(_safe_embed(self.embedding, [query])[0])
        docs = np.array(_safe_embed(self.embedding, texts))

        sims = [cosine(q, d) for d in docs]
        idxs = np.argsort(sims)[::-1][:k]
        return [candidates[i] for i in idxs]

    def llm_select(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        enumerated = [
            f"[{i}] {c['example_doc'].page_content[:300].replace(chr(10), ' ')}"
            for i, c in enumerate(candidates)
        ]

        prompt = (
            "Return a JSON array of indices (0-based) of the most relevant passages.\n\n"
            f"Query: {query}\n\nPassages:\n" + "\n".join(enumerated)
        )

        try:
            arr_txt = call_llm(self.llm, prompt)
            arr = json.loads(arr_txt)
            return [candidates[i] for i in arr[:k] if isinstance(i, int) and 0 <= i < len(candidates)]
        except Exception:
            logger.warning("LLM selection failed, falling back to embeddings")
            return self.embedding_select(query, candidates, k)


# ---------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------
class Compressor:
    def __init__(self, llm: BaseChatModel):
        if BaseChatModel is None:
            raise RuntimeError("BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Compressor requires llm to be a BaseChatModel instance.")
        self.llm = llm

    def llm_compress(self, text: str, query: str, pct: float = 0.35) -> str:
        prompt = (
            f"Compress to ~{int(pct*100)}% preserving facts.\n"
            f"Query: {query}\nText:\n{text}\nCompressed:"
        )
        return call_llm(self.llm, prompt).strip()

    def heuristic_compress(self, text: str, query: str, max_sentences: int = 3) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text)
        q = set(re.findall(r"\w+", query.lower()))

        scored = []
        for s in sents:
            overlap = len(q & set(re.findall(r"\w+", s.lower())))
            scored.append((overlap, s))

        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return " ".join(s for _, s in scored[:max_sentences])


# ---------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------
class ContradictionDetector:
    def __init__(self, llm: BaseChatModel):
        if BaseChatModel is None:
            raise RuntimeError("BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("ContradictionDetector requires llm to be a BaseChatModel instance.")
        self.llm = llm

    def detect(self, query: str, candidates: List[Dict]) -> List[int]:
        enumerated = [
            f"[{i}] {c['example_doc'].page_content[:300]}"
            for i, c in enumerate(candidates)
        ]

        prompt = (
            "Return JSON indices of passages contradicting the majority.\n\n"
            f"Query: {query}\n\nPassages:\n" + "\n".join(enumerated)
        )

        try:
            arr_txt = call_llm(self.llm, prompt)
            arr = json.loads(arr_txt)
            return [i for i in arr if isinstance(i, int) and 0 <= i < len(candidates)]
        except Exception:
            return []


# ---------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------
class PostRetrievalPipeline:
    def __init__(self, embedding_model: Any, llm: BaseChatModel):
        if BaseChatModel is None:
            raise RuntimeError("BaseChatModel not available in environment.")
        if not isinstance(llm, BaseChatModel):
            raise TypeError("PostRetrievalPipeline requires llm to be a BaseChatModel instance.")

        self.llm = llm
        self.reranker = Reranker(embedding_model, self.llm)
        self.selector = Selector(embedding_model, self.llm)
        self.compressor = Compressor(self.llm)
        self.detector = ContradictionDetector(self.llm)

    def process(
        self,
        retrieved: List[Dict],
        query: str,
        final_k: int = 6,
        rerank: str = "mmr",
        compress: bool = True,
    ) -> List[Dict]:

        if not retrieved:
            return []

        candidates = [r for r in retrieved if r.get("example_doc")]

        # Rerank
        if rerank == "model":
            reranked = self.reranker.model_rerank(query, candidates, k=len(candidates))
        else:
            reranked = self.reranker.mmr_rerank(query, candidates, k=len(candidates))

        # Selection (LLM-driven with embedding fallback)
        selected = self.selector.llm_select(query, reranked, k=final_k * 2)

        # Contradiction detection: remove contradicted passages
        drop_idxs = set(self.detector.detect(query, selected))
        selected = [c for i, c in enumerate(selected) if i not in drop_idxs][:final_k]

        out = []
        for c in selected:
            text = c["example_doc"].page_content
            if compress:
                try:
                    compressed = self.compressor.llm_compress(text, query)
                    if compressed and len(compressed) < len(text):
                        text = compressed
                except Exception:
                    # if LLM compression fails, fall back to heuristic
                    text = self.compressor.heuristic_compress(text, query)

            out.append({
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score", 0.0),
                "text": text,
                "metadata": c["example_doc"].metadata,
            })

        return out


# ---------------------------------------------------------------------
# Quick debug block (uses BaseChatModel implementations)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pass
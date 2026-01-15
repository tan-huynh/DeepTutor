"""
retriever.py

Fixed and improved Retrieval module for Modular RAG.

Features / fixes:
- Robust DenseRetriever that supports several Chroma/LangChain response shapes
  (similarity_search_by_vector, query(...) returning ids/distances/documents/metadatas).
- SparseRetriever uses BM25 (if available) or TF-IDF fallback, with consistent doc canonicalization.
- HybridRetriever fuses normalized dense + sparse scores with configurable alpha.
- RetrievalPipeline orchestrates pre-retrieval candidates -> retrieval -> weighted aggregation.
- Utilities to load child chunks from a docstore JSON (as produced by the indexer).
- Defensive handling of missing chunk ids / missing vectorstore score outputs.

Notes:
- This module intentionally does not assume internal access to vector embeddings; it relies on the
  vectorstore (Chroma) query APIs. If your vectorstore stores distances rather than similarities,
  we convert distances -> similarity using sim = 1/(1+dist) as a reasonable default.
- For production, consider batching embeddings and using vectorstore-specific client APIs to
  obtain consistent scores.
"""

from typing import List, Dict, Tuple, Optional, Any, Iterable
import json
import math
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi  # optional dependency
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

from langchain_core.documents import Document
from langchain_chroma import Chroma  # require same adapter used in indexing


from .base import BaseRetriever

from src.lego_rag.logger import logger
# ------------------------------
# Helpers
# ------------------------------
def _to_document(item: Any) -> Document:
    """
    Accept either:
      - langchain Document
      - dict with keys {'id'|'doc_id', 'text'|'page_content', 'metadata'}
    Returns a langchain Document.
    """
    if isinstance(item, Document):
        return item
    if isinstance(item, dict):
        text = item.get("text") or item.get("page_content") or item.get("content") or ""
        metadata = dict(item.get("metadata") or {})
        # prefer explicit chunk_id in metadata, else try fallback keys
        cid = metadata.get("chunk_id") or item.get("id") or item.get("doc_id")
        if cid:
            metadata["chunk_id"] = cid
        return Document(page_content=text, metadata=metadata)
    raise ValueError("Unsupported document type for conversion to Document.")


def _safe_embed(embedding_model: Any, text: str) -> List[float]:
    """
    Call embedding model, prefer embed_query then embed_documents fallback.
    Raise on failure.
    """
    if embedding_model is None:
        raise ValueError("embedding_model is required for embedding queries.")
    try:
        if hasattr(embedding_model, "embed_query"):
            emb = embedding_model.embed_query(text)
            # Some embed_query return list-like; ensure 1-D list
            if isinstance(emb, (list, tuple, np.ndarray)):
                return list(emb)
    except Exception:
        logger.debug("embed_query failed (or not present); trying embed_documents", exc_info=True)

    # fallback
    try:
        emb_list = embedding_model.embed_documents([text])
        if not emb_list:
            raise RuntimeError("embed_documents returned empty list")
        return list(emb_list[0])
    except Exception as e:
        logger.error("Embedding call failed: %s", e)
        raise


def _minmax_normalize(arr: Iterable[float]) -> np.ndarray:
    a = np.array(list(arr), dtype=float)
    if a.size == 0:
        return a
    mn, mx = a.min(), a.max()
    if math.isclose(mn, mx):
        return np.ones_like(a)
    return (a - mn) / (mx - mn)


def _distance_to_similarity(dist: float) -> float:
    """
    Convert a distance metric into a similarity in (0,1], conservative conversion.
    Default: sim = 1 / (1 + dist)
    """
    try:
        d = float(dist)
        return 1.0 / (1.0 + max(0.0, d))
    except Exception:
        return 0.0


# ------------------------------
# Config holder
# ------------------------------
class RetrievalConfig:
    def __init__(self, vectorstore: Optional[Any] = None, embedding_model: Optional[Any] = None,
                 collection_name: str = "rag_index"):
        """
        vectorstore: either
           - a Chroma instance, or
           - a persist_directory path (str) where Chroma was persisted during indexing
        embedding_model: the same embeddings object used in indexing (e.g., HuggingFaceEmbeddings instance)
        collection_name: the collection name used when indexing (default 'rag_index')
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    def get_vectorstore(self):
        if isinstance(self.vectorstore, Chroma):
            return self.vectorstore
        if isinstance(self.vectorstore, str):
            if self.embedding_model is None:
                raise ValueError("embedding_model required to instantiate Chroma from path.")
            return Chroma(collection_name=self.collection_name,
                          embedding_function=self.embedding_model,
                          persist_directory=self.vectorstore)
        raise ValueError("vectorstore must be a Chroma instance or a persist directory path string.")


# ------------------------------
# Sparse Retriever
# ------------------------------
class SparseRetriever(BaseRetriever):
    """
    Build over a list of Documents (langchain Document or convertible dict).
    Uses BM25 when available, else TF-IDF.
    """

    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer or (lambda s: s.split())
        self._docs: List[Document] = []
        self._doc_ids: List[str] = []
        self._bm25 = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._doc_term_matrix = None

    def build(self, docs: List[Any]):
        # canonicalize to Document
        self._docs = [_to_document(d) for d in docs]
        self._doc_ids = [ (d.metadata.get("chunk_id") if d.metadata else f"doc_{i}") or f"doc_{i}" for i, d in enumerate(self._docs) ]
        texts = [d.page_content or "" for d in self._docs]

        if _HAS_BM25:
            tokenized = [self.tokenizer(t.lower()) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("SparseRetriever: built BM25 index with %d docs", len(texts))
        else:
            self._vectorizer = TfidfVectorizer(max_features=50000)
            self._doc_term_matrix = self._vectorizer.fit_transform(texts)
            logger.info("SparseRetriever: built TF-IDF matrix with %d docs", len(texts))

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float, Document]]:
        if _HAS_BM25 and self._bm25 is not None:
            tokens = self.tokenizer(query.lower())
            scores = self._bm25.get_scores(tokens)
            idx = np.argsort(scores)[::-1][:k]
            return [(self._doc_ids[i], float(scores[i]), self._docs[i]) for i in idx]

        if self._vectorizer is None or self._doc_term_matrix is None:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._doc_term_matrix).flatten()
        idx = np.argsort(sims)[::-1][:k]
        return [(self._doc_ids[i], float(sims[i]), self._docs[i]) for i in idx]


# ------------------------------
# Dense Retriever
# ------------------------------
class DenseRetriever(BaseRetriever):
    """
    Use embeddings + vectorstore (Chroma) to retrieve nearest neighbor chunks.
    Exposes retrieve_by_text and retrieve_by_vector.
    """

    def __init__(self, vectorstore: Any, embedding_model: Any):
        """
        vectorstore: Chroma instance (or object exposing query/similarity_search_by_vector)
        embedding_model: embeddings object used in indexing (must implement embed_query or embed_documents)
        """
        self.vs = vectorstore
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Optional[str], float, Document]]:
        return self.retrieve_by_text(query, k=k)

    def retrieve_by_text(self, q: str, k: int = 10) -> List[Tuple[Optional[str], float, Document]]:
        emb = _safe_embed(self.embedding_model, q)
        return self.retrieve_by_vector(emb, k=k)

    def retrieve_by_vector(self, emb: List[float], k: int = 10) -> List[Tuple[Optional[str], float, Document]]:
        """
        Query vectorstore using embedding. Handles multiple common shapes of Chroma/langchain clients.
        Returns list of tuples: (chunk_id | parent_id or None, score (similarity), Document)
        """
        vs = self.vs

        # 1) Try high-level similarity_search_by_vector(embedding, k) -> list[Document]
        try:
            if hasattr(vs, "similarity_search_by_vector"):
                docs = vs.similarity_search_by_vector(emb, k=k)
                out = []
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    cid = meta.get("chunk_id") or meta.get("parent_id")
                    # Some adapters include score in metadata; fallback to 1.0
                    score = float(meta.get("score")) if meta and "score" in meta else 1.0
                    out.append((cid, score, d))
                return out
        except Exception:
            logger.debug("similarity_search_by_vector failed or incompatible.", exc_info=True)

        # 2) Try a query(...) API that returns dict with ids/distances/documents/metadatas
        try:
            if hasattr(vs, "query"):
                qres = vs.query(query_embeddings=[emb], n_results=k)
                # typical shape: {'ids': [[...]], 'distances': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
                ids = qres.get("ids", [[]])[0] if isinstance(qres, dict) else []
                distances = qres.get("distances", [[]])[0] if isinstance(qres, dict) else []
                docs = qres.get("documents", [[]])[0] if isinstance(qres, dict) else []
                metas = qres.get("metadatas", [[]])[0] if isinstance(qres, dict) else []

                out = []
                for i, doc_obj in enumerate(docs):
                    meta = metas[i] if metas and i < len(metas) else {}
                    cid = meta.get("chunk_id") or (ids[i] if ids and i < len(ids) else None)
                    dist = distances[i] if distances and i < len(distances) else None
                    score = _distance_to_similarity(dist) if dist is not None else (meta.get("score") or 1.0)
                    # doc_obj may be raw text or a Document; canonicalize
                    d = _to_document({"page_content": doc_obj, "metadata": meta}) if not isinstance(doc_obj, Document) else doc_obj
                    out.append((cid, float(score), d))
                return out
        except Exception:
            logger.debug("vectorstore.query(...) failed or incompatible.", exc_info=True)

        # 3) Try lower-level collection access if available (best-effort)
        try:
            collection = getattr(vs, "collection", None) or getattr(vs, "_collection", None) or getattr(vs, "client", None)
            if collection:
                # many Chroma clients: collection.query(query_embeddings=[emb], n_results=k)
                qres = collection.query(query_embeddings=[emb], n_results=k)
                ids = qres.get("ids", [[]])[0]
                distances = qres.get("distances", [[]])[0]
                docs = qres.get("documents", [[]])[0]
                metas = qres.get("metadatas", [[]])[0]
                out = []
                for i, doc_obj in enumerate(docs):
                    meta = metas[i] if metas and i < len(metas) else {}
                    cid = meta.get("chunk_id") or (ids[i] if ids and i < len(ids) else None)
                    dist = distances[i] if distances and i < len(distances) else None
                    score = _distance_to_similarity(dist) if dist is not None else (meta.get("score") or 1.0)
                    d = _to_document({"page_content": doc_obj, "metadata": meta}) if not isinstance(doc_obj, Document) else doc_obj
                    out.append((cid, float(score), d))
                return out
        except Exception:
            logger.debug("lower-level collection.query failed.", exc_info=True)

        # 4) Last resort: return empty to let caller handle
        logger.error("DenseRetriever: vectorstore query failed for all fallbacks.")
        return []


# ------------------------------
# Hybrid Retriever
# ------------------------------
class HybridRetriever(BaseRetriever):
    """
    Combine dense and sparse retrievals:
      fused_score = alpha * dense_norm + (1-alpha) * sparse_norm
    """

    def __init__(self, dense: DenseRetriever, sparse: Optional[SparseRetriever] = None, alpha: float = 0.6):
        self.dense = dense
        self.sparse = sparse
        self.alpha = float(alpha)

    def retrieve(self, query: str, k: int = 10, dense_k: int = 50, sparse_k: int = 200) -> List[Tuple[str, float, Optional[Document]]]:
        dense_hits = self.dense.retrieve_by_text(query, k=dense_k)  # list[(cid, score, doc)]
        dense_ids = [h[0] for h in dense_hits]
        dense_scores = [h[1] for h in dense_hits]
        dense_norm = _minmax_normalize(dense_scores) if len(dense_scores) else np.array([])

        sparse_map = {}
        sparse_docs = {}
        if self.sparse:
            sparse_hits = self.sparse.retrieve(query, k=sparse_k)
            sparse_ids = [h[0] for h in sparse_hits]
            sparse_scores = [h[1] for h in sparse_hits]
            sparse_norm = _minmax_normalize(sparse_scores) if len(sparse_scores) else np.array([])
            for i, sid in enumerate(sparse_ids):
                sparse_map[sid] = float(sparse_norm[i] if sparse_norm.size else sparse_scores[i])
                sparse_docs[sid] = sparse_hits[i][2]

        score_map = {}
        doc_example = {}

        # dense contribution
        for i, cid in enumerate(dense_ids):
            sc = float(dense_norm[i]) if dense_norm.size else float(dense_scores[i])
            score_map[cid] = score_map.get(cid, 0.0) + self.alpha * sc
            doc_example[cid] = dense_hits[i][2]

        # sparse contribution
        for cid, s_norm in sparse_map.items():
            score_map[cid] = score_map.get(cid, 0.0) + (1.0 - self.alpha) * float(s_norm)
            if cid not in doc_example:
                doc_example[cid] = sparse_docs.get(cid)

        # return top-k
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
        out = [(cid, float(score), doc_example.get(cid)) for cid, score in ranked]
        return out


# ------------------------------
# RetrievalPipeline (orchestrator)
# ------------------------------
class RetrievalPipeline:
    """
    Accepts pre-retrieval output: {"candidates": [{"id","text","weight","type"}, ...], "hyde": "..."}
    Produces top-k aggregated retrieval results with weighted fusion across candidates.
    """

    def __init__(self, config: RetrievalConfig, sparse_index: Optional[SparseRetriever] = None, alpha: float = 0.6):
        self.config = config
        self.vs = config.get_vectorstore()
        self.embedding_model = config.embedding_model
        self.dense = DenseRetriever(self.vs, self.embedding_model)
        self.sparse = sparse_index
        self.hybrid = HybridRetriever(self.dense, self.sparse, alpha=alpha) if self.sparse else None

    def retrieve(self, pre_retrieval_output: Dict, k: int = 10, use_hyde: bool = True, hybrid: bool = True) -> List[Dict]:
        candidates = pre_retrieval_output.get("candidates", [])
        hyde_text = pre_retrieval_output.get("hyde")

        searches: List[Tuple[str, float]] = []
        if use_hyde and hyde_text:
            searches.append((hyde_text, 0.5))
        for c in candidates:
            searches.append((c.get("text", ""), float(c.get("weight", 0.1))))

        agg_scores: Dict[str, float] = {}
        example_doc: Dict[str, Document] = {}

        for text, weight in searches:
            if not text:
                continue
            if hybrid and self.hybrid:
                hits = self.hybrid.retrieve(text, k=max(50, k))
                for cid, sc, doc in hits:
                    if cid is None:
                        continue
                    agg_scores[cid] = agg_scores.get(cid, 0.0) + weight * sc
                    if cid not in example_doc and doc is not None:
                        example_doc[cid] = doc
            else:
                hits = self.dense.retrieve_by_text(text, k=max(50, k))
                ids = [h[0] for h in hits]
                scores = [h[1] for h in hits]
                norm = _minmax_normalize(scores) if len(scores) else np.array([])
                for i, cid in enumerate(ids):
                    if cid is None:
                        continue
                    sc = float(norm[i]) if norm.size else float(scores[i])
                    agg_scores[cid] = agg_scores.get(cid, 0.0) + weight * sc
                    if cid not in example_doc:
                        example_doc[cid] = hits[i][2]

        # Rank final aggregated scores and return top-k with example doc
        ranked = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for cid, score in ranked:
            out.append({
                "chunk_id": cid,
                "score": float(score),
                "example_doc": example_doc.get(cid)
            })
        return out


# ------------------------------
# Utilities: load documents from docstore.json (indexer output)
# ------------------------------
def load_chunks_from_docstore(docstore_path: str) -> List[Document]:
    """
    docstore.json expected format (as in the indexing module):
      { "<chunk_id>": { "type": "child", "text": "...", "metadata": {...} }, ... }
    Returns a list of langchain Document objects for child chunks.
    """
    with open(docstore_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    docs: List[Document] = []
    for doc_id, entry in ds.items():
        if entry.get("type") != "child":
            continue
        meta = dict(entry.get("metadata", {}))
        meta["chunk_id"] = doc_id
        docs.append(Document(page_content=entry.get("text", "") or entry.get("summary", ""), metadata=meta))
    return docs

if __name__ == "__main__":
    """
    Minimal self-debug for Retrieval module.

    What this checks:
    - Embedding model works
    - Chroma vectorstore responds
    - Dense + Sparse + Hybrid retrieval run end-to-end
    - RetrievalPipeline output structure is sane
    """

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from settings import settings
    import os
    import shutil

    # ---------------------------
    # Step 1: Setup embedding model
    # ---------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name= settings.EMBEDDING_MODEL_PATH,
    )

    # ---------------------------
    # Step 2: Create tiny debug corpus
    # ---------------------------
    debug_docs = [
        Document(
            page_content="Transformers are neural networks based on self-attention.",
            metadata={"chunk_id": "c1"}
        ),
        Document(
            page_content="BM25 is a sparse retrieval algorithm used in information retrieval.",
            metadata={"chunk_id": "c2"}
        ),
        Document(
            page_content="Dense retrieval maps text to vectors using neural embeddings.",
            metadata={"chunk_id": "c3"}
        ),
    ]

    # ---------------------------
    # Step 3: Build Chroma index
    # ---------------------------
    # cleanup old debug dir if exists
    if os.path.exists("./debug_chroma_db"):
        shutil.rmtree("./debug_chroma_db")

    vectorstore = Chroma.from_documents(
        documents=debug_docs,
        embedding=embedding_model,
        collection_name="debug_rag",
        persist_directory="./debug_chroma_db"
    )

    # ---------------------------
    # Step 4: Build sparse retriever
    # ---------------------------
    sparse = SparseRetriever()
    sparse.build(debug_docs)

    # ---------------------------
    # Step 5: Create RetrievalPipeline
    # ---------------------------
    config = RetrievalConfig(
        vectorstore=vectorstore,
        embedding_model=embedding_model,
        collection_name="debug_rag"
    )

    pipeline = RetrievalPipeline(
        config=config,
        sparse_index=sparse,
        alpha=0.6
    )

    # ---------------------------
    # Step 6: Fake pre-retrieval output
    # ---------------------------
    pre_retrieval_output = {
        "hyde": "Explain dense and sparse retrieval in RAG systems",
        "candidates": [
            {"text": "What is dense retrieval?", "weight": 0.3},
            {"text": "BM25 vs embeddings", "weight": 0.2},
        ]
    }

    # ---------------------------
    # Step 7: Run retrieval
    # ---------------------------
    results = pipeline.retrieve(
        pre_retrieval_output=pre_retrieval_output,
        k=3
    )

    # ---------------------------
    # Step 8: Inspect output
    # ---------------------------
    print("\n=== RETRIEVAL DEBUG OUTPUT ===")
    for r in results:
        print(f"\nChunk ID : {r['chunk_id']}")
        print(f"Score    : {r['score']:.4f}")
        print(f"Text     : {r['example_doc'].page_content[:80]}...")
"""
Indexer serves as the mechanism for transforming raw data into a structured, searchable format

contains:
- IndexingConfig centralizes configuration 
- ChunkingOptimizer implements Small-to-Big setup, sliding window,
  and attaches provenance metadata with timestamps.
- HierarchicalIndexer indexes parent & child nodes into the vectorstore and stores an docstore.
- KnowledgeGraphIndexer builds a simple semantic graph and exposes neighbors().

"""

import os
import uuid
import json
import pickle
import datetime
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import numpy as np
import networkx as nx

#dependices
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

# Ingestion functions
from src.lego_rag.ingestion.loader_dispatch import load_documents, load_documents_from_dir
from src.lego_rag.ingestion.errors import IngestionError

#configs
from settings import settings
from src.lego_rag.logger import logger


def get_device() -> str:
    """
    basic setup for detecting if device has gpu available,
    if not switch to cpu mode.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
    
# ============================================================
# 1. CONFIGURATION
# ============================================================
class IndexingConfig:
    """
    Centralized configuration for indexing. Pass an optional LLM instance
    that will be used for summarization.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.embedding_model_path = str(Path(settings.EMBEDDING_MODEL_PATH).resolve())
        self.vector_db_path = str(Path(settings.VECTOR_DB_PATH).resolve())
        self.docstore_path = str(Path(settings.DOCSTORE_PATH).resolve())
        self.kg_path = str(Path(settings.KNOWLEDGE_GRAPH_PATH).resolve())
        self.collection_name = settings.COLLECTION_NAME
        self.llm = llm   # llm model to use for summary purpose.

        self._validate_paths()
        # Embedding model initialization
        self.embedding_model: Embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={"device": get_device()},
        )
        
    def _validate_paths(self):
        if Path(self.docstore_path).is_dir():
            raise ValueError("DOCSTORE_PATH must be a file")

        if Path(self.kg_path).is_dir():
            raise ValueError("KNOWLEDGE_GRAPH_PATH must be a file")

    

# ============================================================
# 2. CHUNK OPTIMIZATION (Small-to-Big + metadata)
# ============================================================
class ChunkingOptimizer:
    """
    Chunking optimizer that:
      - produces parent (larger) chunks and child (smaller) chunks (Small-to-Big)
      - attaches metadata
      - uses sliding window splitters for both levels
    """

    def __init__(
        self,
        small_chunk_size: int = 250,
        small_chunk_overlap: int = 50,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 400,
        parent_summary_sentences: int = 3,
    ):
        # child-level (fine retrieval)
        self.small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=small_chunk_size,
            chunk_overlap=small_chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " "],
        )

        # parent-level (context for generation)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!"],
        )

        self.parent_summary_sentences = parent_summary_sentences

    def _extractive_summary(self, text: str) -> str:
        """
        Generate a simple extractive summary from the input text.

        creates a lightweight summary by:
        - Normalizing whitespace and removing newlines
        - Splitting the text into sentences using periods as delimiters
        - Selecting the first `parent_summary_sentences` sentences

        The approach is purely extractive (no semantic understanding)
        and is intended as a fast, fallback summarization strategy
        when an LLM-based summarizer is unavailable.

        Args:
            text (str): The full input text to summarize.

        Returns:
            str: An extractive summary composed of the first
            `parent_summary_sentences` sentences from the text.
        """

        pieces = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return ". ".join(pieces[: self.parent_summary_sentences]).strip()

    def _now_iso(self) -> str:
        """
        Get the current UTC timestamp.

        Returns:
            str: The current time in UTC, formatted string
            (e.g., "2026-01-12T10:45:30.123456+00:00").
        """

        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def chunk(self, text: str, base_metadata: Optional[Dict], llm: Optional[BaseChatModel] = None, dual_chunk: bool = True) -> List[Document]:
        """
        Dual-chunk the input text and return Documents in order:
          - parents (node_type='parent')
          - children (node_type='child', each with parent_id)
        Each Document.metadata includes provenance keys.

        args:
          - text: raw text string
          - base_metadata: provenance dict (source, filename, page, etc.)
          - llm: optional LLM used only for short summaries of parent chunks
          - dual_chunk: if False, behave like a single-level chunker (child chunks only)

        Returns:
          - List[Document]
        """

        if not isinstance(base_metadata, dict):
            base_metadata = {}

        now = self._now_iso()

        if not dual_chunk:
            raw_child_docs = self.small_splitter.split_documents([Document(page_content=text)])
            enriched_children: List[Document] = []
            for idx, d in enumerate(raw_child_docs):
                metadata = {
                    **base_metadata,
                    "node_type": "child",
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": idx,
                    "created_at": now,
                }
                enriched_children.append(Document(page_content=d.page_content, metadata=metadata))
            return enriched_children

        # Parent chunks
        parent_lc = self.parent_splitter.split_documents([Document(page_content=text)])
        parent_map: List[Dict[str, Any]] = []
        parents: List[Document] = []

        for p_idx, p in enumerate(parent_lc):
            parent_id = str(uuid.uuid4())
            p_text = p.page_content

            # optional LLM summary (attempt; fall back to extractive)
            summary = None
            if llm:
                try:
                    prompt = f"Summarize the following text in concise sentences:\n\n{p_text}"
                    # tolerant invocation for different LLM adapters
                    if callable(llm):
                        summary_raw = llm(prompt)
                        if isinstance(summary_raw, str) and summary_raw.strip():
                            summary = summary_raw.strip()
                    elif hasattr(llm, "generate"):
                        gen = llm.generate(prompt)
                        summary_raw = getattr(gen, "text", None) or str(gen)
                        if summary_raw and isinstance(summary_raw, str):
                            summary = summary_raw.strip()
                    elif hasattr(llm, "invoke"):
                        gen = llm.invoke(prompt)
                        summary_raw = getattr(gen, "content", None) or str(gen)
                        if summary_raw and isinstance(summary_raw, str):
                            summary = summary_raw.strip()
                except Exception:
                    logger.debug("LLM summarization failed; falling back to extractive.", exc_info=True)

            if not summary:
                summary = self._extractive_summary(p_text)

            p_meta = {
                **base_metadata,
                "node_type": "parent",
                "parent_id": parent_id,
                "parent_index": p_idx,
                "summary": summary,
                "created_at": now,
            }
            parents.append(Document(page_content=p_text, metadata=p_meta))
            parent_map.append({"id": parent_id, "text": p_text, "summary": summary})

        # Child chunks
        child_lc = self.small_splitter.split_documents([Document(page_content=text)])
        children: List[Document] = []
        for c_idx, c in enumerate(child_lc):
            c_text = c.page_content.strip()
            matched_parent_id = None
            # containment mapping; fallback to first parent if none
            for pm in parent_map:
                if c_text and c_text in pm["text"]:
                    matched_parent_id = pm["id"]
                    break
            if matched_parent_id is None and parent_map:
                matched_parent_id = parent_map[0]["id"]

            child_id = str(uuid.uuid4())
            c_meta = {
                **base_metadata,
                "node_type": "child",
                "parent_id": matched_parent_id,
                "chunk_id": child_id,
                "chunk_index": c_idx,
                "created_at": now,
            }
            children.append(Document(page_content=c.page_content, metadata=c_meta))

        return parents + children

    async def chunk_from_file(self, file_path: Union[str, List[str]], base_metadata: Dict, llm: Optional[BaseChatModel] = None, dual_chunk: bool = True) -> List[Document]:
        """
        Async convenience wrapper: load file(s) via your ingestion loader functions and chunk them.
        Returns a flat list of Documents (parents + children per document) and preserves ingestion metadata.

        Note: This is a convenience helper. Prefer calling ingestion loader_dispatch in the pipeline
        and calling chunk(...) directly for better testability and clearer responsibilities.
        """
        file_inputs = [file_path] if isinstance(file_path, str) else list(file_path)
        all_chunks: List[Document] = []

        for p in file_inputs:
            if Path(p).is_dir():
                raw_docs = await load_documents_from_dir(p)
            else:
                raw_docs = await load_documents(p)

            if not raw_docs:
                continue

            # raw_docs expected to be list[dict] with keys 'text','source','metadata'
            for rd in raw_docs:
                rd_text = rd.get("text", "")
                rd_meta = {"source": rd.get("source"), **(rd.get("metadata") or {}), **(base_metadata or {})}
                chunks = self.chunk(rd_text, rd_meta, llm=llm, dual_chunk=dual_chunk)
                all_chunks.extend(chunks)

        return all_chunks


# ============================================================
# 3. HIERARCHICAL (SMALL-TO-BIG) INDEXER
# ============================================================
class HierarchicalIndexer:
    """
    - Parent summary nodes (embedded)
    - Child chunks (embedded)
    - Persistent docstore (atomic writes)
    """

    def __init__(self, config: IndexingConfig):
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=config.embedding_model,
            persist_directory=str(config.vector_db_path),
        )

        self.docstore: Dict[str, Dict] = {}
        self.config = config
        self._load_docstore()

    def _load_docstore(self):
        docstore_path = Path(self.config.docstore_path)
        if docstore_path.exists() and docstore_path.is_file():
            try:
                with docstore_path.open("r", encoding="utf-8") as f:
                    self.docstore = json.load(f)
                logger.info(f"Loaded {len(self.docstore)} items from docstore.")
            except Exception as e:
                logger.error(f"Failed to load docstore: {e}")

    def _summarize_parent(self, text: str, llm: Optional[BaseChatModel] = None) -> str:
        """
        summarize via provided LLM; fall back to extractive.
        """
        chosen_llm = llm or getattr(self.config, "llm", None)
        if chosen_llm:
            try:
                prompt = f"Summarize the following text in concise sentences:\n\n{text}"
                if callable(chosen_llm):
                    out = chosen_llm(prompt)
                    if isinstance(out, str) and out.strip():
                        return out.strip()
                elif hasattr(chosen_llm, "generate"):
                    gen = chosen_llm.generate(prompt)
                    text_out = getattr(gen, "text", None) or str(gen)
                    if text_out:
                        return text_out.strip()
                elif hasattr(chosen_llm, "invoke"):
                    gen = chosen_llm.invoke(prompt)
                    text_out = getattr(gen, "content", None) or str(gen)
                    if text_out:
                        return text_out.strip()
            except Exception:
                logger.debug("LLM summarization failed, using extractive fallback", exc_info=True)

        # Extractive fallback
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return ". ".join(sentences[:3]).strip()

    def index_document(self, text: str, source_metadata: Dict):
        """
        Index a document by creating parent and child nodes, storing both in vectorstore
        and recording provenance in docstore.
        """
        parent_id = str(uuid.uuid4())

        parent_summary = self._summarize_parent(text, llm=getattr(self.config, "llm", None))

        # Parent docstore entry
        self.docstore[parent_id] = {
            "type": "parent",
            "summary": parent_summary,
            "metadata": source_metadata,
        }

        # Add parent to vectorstore (embedding)
        parent_doc = Document(
            page_content=parent_summary,
            metadata={
                **source_metadata,
                "node_type": "parent",
                "parent_id": parent_id,
            },
        )
        self.vectorstore.add_documents([parent_doc])

        # Create child chunks using a child splitter (preserve original logic)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40,
            separators=["\n\n", "\n", ".", "?", "!"],
        )
        children = splitter.split_documents([Document(page_content=text)])

        child_docs = []
        for idx, c in enumerate(children):
            chunk_id = str(uuid.uuid4())
            meta = {
                **source_metadata,
                "node_type": "child",
                "parent_id": parent_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
            }
            child_docs.append(Document(page_content=c.page_content, metadata=meta))

            # docstore record for child with provenance (citation trail)
            self.docstore[chunk_id] = {
                "type": "child",
                "parent_id": parent_id,
                "text": c.page_content,
                "metadata": meta,
            }

        # Embed child docs (batch)
        if child_docs:
            self.vectorstore.add_documents(child_docs)
        
        # Persist docstore updates immediately for UI sync
        self.persist_docstore()

    def _get_node(self, node_id: str) -> Dict:
        if node_id not in self.docstore:
            raise KeyError(f"Document with id '{node_id}' not found in docstore.")
        return self.docstore[node_id]

    def clear(self):
        """Wipe Chroma, Docstore, and Knowledge Graph from disk"""
        import shutil
        import os

        # 1. Clear Vector DB (Chroma)
        try:
            self.vectorstore.delete_collection()
        except:
            pass # Collection might not exist
        
        # Force delete persistence directory
        if Path(self.config.vector_db_path).exists():
             shutil.rmtree(self.config.vector_db_path)
        
        # 2. Clear Docstore
        self.docstore = {}
        if Path(self.config.docstore_path).exists():
            os.remove(self.config.docstore_path)
            
        # 3. Clear Knowledge Graph
        if Path(self.config.kg_path).exists():
            os.remove(self.config.kg_path)

        # Re-initialize clean state
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.config.embedding_model,
            persist_directory=str(self.config.vector_db_path),
        )
        # Re-persist empty docstore to sync UI
        self.persist_docstore()
        
        logger.info("✅ Hard cleared Vectorstore, Docstore, and KG.")
    
    def remove_source(self, source_name: str):
        """Remove all nodes related to a specific source"""
        ids_to_del = [k for k, v in self.docstore.items() if v.get("metadata", {}).get("source") == source_name]
        if ids_to_del:
            self.vectorstore.delete(ids=ids_to_del)
            for cid in ids_to_del:
                del self.docstore[cid]
            logger.info(f"Removed {len(ids_to_del)} chunks for source: {source_name}")
    
    def persist_docstore(self):
        docstore_path = Path(self.config.docstore_path)

        if docstore_path.exists() and docstore_path.is_dir():
            raise ValueError(f"DOCSTORE_PATH must be a file, got directory: {docstore_path}")

        tmp_path = docstore_path.with_suffix(".tmp")
        docstore_path.parent.mkdir(parents=True, exist_ok=True)

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.docstore, f, indent=2)

        tmp_path.replace(docstore_path)

    def list_parents(self) -> List[Dict]:
        """
        List all parent nodes stored in the docstore.
        This method iterates over the internal docstore and returns
        all entries marked as parent nodes, preserving their insertion
        order.
        - Only nodes with `"type" == "parent"` are included.
        Returns:
            List[Dict]: A list of dictionaries representing parent nodes.
 
        """

        return [{"parent_id": doc_id, **node} for doc_id, node in self.docstore.items() if node.get("type") == "parent"]

    def get_parent_by_index(self, index: int) -> Dict:
        """
        Retrieve a parent node by its positional index.
        Parents are ordered based on their insertion order
        in the internal docstore.

        Args:
            index (int): index of the parent node.

        Returns:
            Dict: A dictionary representing the parent node
        """

        parents = list(self.list_parents())
        if index < 0 or index >= len(parents):
            raise IndexError(f"Parent index {index} out of range (0–{len(parents) - 1})")
        return parents[index]

    def get_child_by_index(self, index: int) -> Dict:
        """
        Retrieve a child node by its positional index.
        Child nodes are ordered based on their insertion order
        in the internal docstore.

        Args:
            index (int): index of the child node.

        Returns:
            Dict: A dictionary representing the child node
        """

        children = [(doc_id, node) for doc_id, node in self.docstore.items() if node.get("type") == "child"]
        if index < 0 or index >= len(children):
            raise IndexError(f"Child index {index} out of range (0–{len(children) - 1})")
        chunk_id, child_node = children[index]
        return {"chunk_id": chunk_id, **child_node}

    def get_child_by_parent_index(self, parent_index: int, child_index: int) -> Dict:
        """
        Retrieve a child node by parent index and child index.
        This method first resolves the parent using its index,
        then retrieves the child belonging to that parent based
        on the child's chunk index.

        Args:
            parent_index (int): index of the parent node.
            child_index (int): index of the child node
                within the specified parent.

        Returns:
            Dict: A dictionary representing the child node, including:
                - chunk_id (str): Unique identifier of the child chunk
                - parent_id (str): Identifier of the parent node
                - text (str): Chunk text content
                - metadata (Dict): Associated metadata
        """

        parent = self.get_parent_by_index(parent_index)
        parent_id = parent["parent_id"]
        children = sorted(
            [(doc_id, node) for doc_id, node in self.docstore.items() if node.get("type") == "child" and node.get("parent_id") == parent_id],
            key=lambda x: x[1]["metadata"]["chunk_index"],
        )
        if child_index < 0 or child_index >= len(children):
            raise IndexError(f"Child index {child_index} out of range (0–{len(children) - 1})")
        chunk_id, child_node = children[child_index]
        return {"chunk_id": chunk_id, **child_node}

    # ---------------------------
    # Search helpers (simple wrappers)
    # ---------------------------
    def search_by_vector(self, vector: List[float], top_k: int = 10) -> List[Dict]:
        """
        Query the vectorstore using a vector. Normalizes some common result.
        """

        emb = vector
        try:
            results = self.vectorstore.similarity_search_by_vector(emb, k=top_k)
            out = []
            for r in results:
                meta = getattr(r, "metadata", {}) or {}
                out.append({"id": meta.get("chunk_id") or meta.get("parent_id"), "score": None, "document": r, "metadata": meta})
            return out
        except Exception:
            pass

        raise RuntimeError("Vectorstore similarity search failed. Adapt HierarchicalIndexer.search_by_vector for your vectorstore.")

    def search_by_text(self, text: str, top_k: int = 10) -> List[Dict]:
        """
        Embed `text` using the embedding model and perform similarity search
        """

        emb = self.config.embedding_model.embed_documents([text])[0]
        return self.search_by_vector(emb, top_k=top_k)


# ============================================================
# 4. KNOWLEDGE GRAPH INDEX
# ============================================================
class KnowledgeGraphIndexer:
    """
    Graph G = {V, E, X}
    - Nodes = chunk_id (child or parent)
    - Edges = semantic similarity (with provenance stored in node metadata)
    """

    def __init__(self, embedding_model: Embeddings, kg_path: Optional[str] = None):
        self.graph = nx.Graph()
        self.embedding_model = embedding_model
        if kg_path:
            self.load(kg_path)

    def load(self, path: Union[str, Path]):
        path = Path(path)
        if path.exists() and path.is_file():
            try:
                with path.open("rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded KG with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load KG: {e}")

    def build(self, docs: List[Document], top_k: int = 5, similarity_threshold: float = 0.6):
        """
        Build or Update the KG.
        """
        if not docs:
            return

        new_texts = [d.page_content for d in docs]
        new_ids = [(d.metadata.get("chunk_id") or d.metadata.get("parent_id") or str(uuid.uuid4())) for d in docs]
        new_embeddings = self.embedding_model.embed_documents(new_texts)
        
        # Add new nodes
        for i, cid in enumerate(new_ids):
            self.graph.add_node(cid, content=new_texts[i], metadata=docs[i].metadata, embedding=new_embeddings[i])

        # Get all existing nodes with embeddings
        existing_node_data = []
        for n, data in self.graph.nodes(data=True):
            if "embedding" in data:
                existing_node_data.append((n, data["embedding"]))
        
        if not existing_node_data:
            return

        all_ids = [n for n, _ in existing_node_data]
        all_embs = np.array([e for _, e in existing_node_data])
        
        # Cosine similarity between new and all
        new_embs_arr = np.array(new_embeddings)
        norms_new = np.linalg.norm(new_embs_arr, axis=1, keepdims=True)
        norms_all = np.linalg.norm(all_embs, axis=1, keepdims=True)
        
        norms_new[norms_new == 0] = 1.0
        norms_all[norms_all == 0] = 1.0
        
        sim = np.dot(new_embs_arr / norms_new, (all_embs / norms_all).T)

        for i, nid in enumerate(new_ids):
            # Sort neighbors
            neighbors_idx = np.argsort(sim[i])[::-1]
            count = 0
            for j in neighbors_idx:
                neighbor_id = all_ids[j]
                if nid == neighbor_id: continue
                if sim[i, j] > similarity_threshold:
                    self.graph.add_edge(nid, neighbor_id, weight=float(sim[i, j]))
                    count += 1
                if count >= top_k:
                    break

    def neighbors(self, node_id: str, top_k: int = 10) -> List[Dict]:
        """
        Return neighbor nodes with weights sorted descending.
        """

        if node_id not in self.graph:
            return []
        nbrs = [(nbr, self.graph.edges[node_id, nbr]["weight"]) for nbr in self.graph.neighbors(node_id)]
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for nid, w in nbrs_sorted:
            out.append({"node_id": nid, "weight": w, "metadata": self.graph.nodes[nid].get("metadata")})
        return out

    def persist(self, path: Union[str, Path]):
        path = Path(path)

        if path.exists() and path.is_dir():
            raise ValueError(f"KNOWLEDGE_GRAPH_PATH must be a file, got directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self.graph, f)

# ============================================================
# Async helpers & pipeline orchestration
# ============================================================
async def aindex_single_document(hierarchy: HierarchicalIndexer, chunker: ChunkingOptimizer, d: Dict) -> List[Document]:
    """
    Index a single raw document dict, where d is expected to have:
      - 'text'
      - 'source'
      - 'metadata' (dict)

    The function indexes the document (parent+children) and returns produced chunks
    (so the pipeline can collect them for KG building).
    """

    # index_document may be CPU-bound due to embedding so offload to thread
    await asyncio.to_thread(hierarchy.index_document, d["text"], {"source": d.get("source"), **(d.get("metadata") or {})})

    # produce chunks (parents + children) via chunker.chunk (sync) but offload to thread
    chunks = await asyncio.to_thread(chunker.chunk, d["text"], {"source": d.get("source"), **(d.get("metadata") or {})}, hierarchy.config.llm, True)
    return chunks


async def arun_indexing_pipeline(input_path: str, concurrency: int = 8):
    """
      - loads documents
      - builds index via HierarchicalIndexer (parents + children)
      - persists docstore
      - builds KG from all chunks and persists it

    args:
      - input_path: file or directory path
      - concurrency: max parallel ingestion tasks
    """

    try:
        path_obj = Path(input_path).resolve()
        if not path_obj.exists():
            raise IngestionError(f"Path does not exist: {path_obj}")
        
        if path_obj.is_dir():
            raw_docs = await load_documents_from_dir(str(path_obj))
        else:
            raw_docs = await load_documents(str(path_obj))

    except IngestionError as e:
        logger.error("Ingestion failed: %s", e)
        return

    if not raw_docs:
        logger.warning("No documents loaded from input path: %s", input_path)
        return

    config = IndexingConfig()
    chunker = ChunkingOptimizer()
    hierarchy = HierarchicalIndexer(config)
    kg = KnowledgeGraphIndexer(config.embedding_model, config.kg_path)

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(doc):
        async with sem:
            try:
                # Remove existing chunks for this source to prevent metadata fragmentation/duplicates
                source_name = doc.get("source")
                if source_name:
                    hierarchy.remove_source(source_name)
                    
                return await aindex_single_document(hierarchy, chunker, doc)
            except Exception as e:
                logger.exception("Indexing failed for a document: %s", e)
                return []

    tasks = [asyncio.create_task(sem_task(d)) for d in raw_docs]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # flatten chunks list
    all_chunks = [c for sub in results for c in sub]

    # persist docstore
    await asyncio.to_thread(hierarchy.persist_docstore)

    # KG build & persist offloaded to thread
    await asyncio.to_thread(kg.build, all_chunks)
    await asyncio.to_thread(kg.persist, config.kg_path)

    logger.info("✅ Indexing pipeline complete.")
    logger.info("Chunks indexed: %d", len(all_chunks))
    logger.info("KG nodes: %d", kg.graph.number_of_nodes())
    logger.info("KG edges: %d", kg.graph.number_of_edges())

    return {"chunks_indexed": len(all_chunks), "kg_nodes": kg.graph.number_of_nodes(), "kg_edges": kg.graph.number_of_edges()}

# Quick debugging
if __name__ == "__main__":
    import asyncio
    import os
    import shutil
    from settings import settings

    async def debug_indexing_pipeline():
        print("--- Debugging Indexing Pipeline ---")
        
        # Setup temp test dir
        test_dir = "./debug_data_indexing"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir)

        # Create dummy file
        dummy_file = os.path.join(test_dir, "debug_doc.txt")
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("This is a dummy document for checking the indexer pipeline.")

        try:            
            print(f"Indexing data from: {test_dir}")
            res = await arun_indexing_pipeline(test_dir)
            print("Pipeline Result:", res)
            
        finally:
            # Cleanup input data
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            print("--- End Debugging ---")

    asyncio.run(debug_indexing_pipeline())
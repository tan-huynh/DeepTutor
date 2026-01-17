"""
LlamaIndex Pipeline
===================

Pipeline wrapper around LlamaIndex. Accepts pre-parsed documents.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Document as LlamaDocument
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.settings import Settings

from src.logging import get_logger
from src.services.embedding import get_embedding_client, get_embedding_config
from src.services.rag.types import Document

from .base import BasePipeline


class CustomEmbedding(BaseEmbedding):
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = get_embedding_client()

    @classmethod
    def class_name(cls) -> str:
        return "custom_embedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._aget_query_embedding(query))
                return future.result()
        return asyncio.run(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._aget_text_embedding(text))
                return future.result()
        return asyncio.run(self._aget_text_embedding(text))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        embeddings = await self._client.embed([query])
        return embeddings[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        embeddings = await self._client.embed([text])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._client.embed(texts)


class LlamaIndexPipeline(BasePipeline):
    name = "llamaindex"

    def __init__(self, kb_base_dir: str):
        super().__init__(kb_base_dir=kb_base_dir)
        self.logger = get_logger("LlamaIndexPipeline")
        self._configure_llamaindex()

    def _configure_llamaindex(self) -> None:
        embedding_cfg = get_embedding_config()
        Settings.embed_model = CustomEmbedding()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        self.logger.info(
            f"LlamaIndex configured: embedding={embedding_cfg.model} "
            f"({embedding_cfg.dim}D, {embedding_cfg.binding}), chunk_size=512"
        )

    async def initialize(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        self.logger.info(
            f"Initializing KB '{kb_name}' with {len(documents)} documents using LlamaIndex"
        )
        kb_dir = Path(self.kb_base_dir) / kb_name
        storage_dir = kb_dir / "llamaindex_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        llama_docs: List[LlamaDocument] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            llama_docs.append(
                LlamaDocument(
                    text=doc.content,
                    metadata={"file_name": Path(doc.file_path).name, "file_path": doc.file_path},
                )
            )

        if not llama_docs:
            self.logger.error("No valid documents found")
            return False

        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None, lambda: VectorStoreIndex.from_documents(llama_docs, show_progress=True)
        )
        index.storage_context.persist(persist_dir=str(storage_dir))
        return True

    async def add_documents(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        storage_dir = Path(self.kb_base_dir) / kb_name / "llamaindex_storage"
        if not storage_dir.exists():
            raise ValueError(f"LlamaIndex storage not found: {storage_dir}")

        llama_docs: List[LlamaDocument] = []
        for doc in documents:
            if not doc.content.strip():
                continue
            llama_docs.append(
                LlamaDocument(
                    text=doc.content,
                    metadata={"file_name": Path(doc.file_path).name, "file_path": doc.file_path},
                )
            )

        if not llama_docs:
            return True

        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None,
            lambda: load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(storage_dir))
            ),
        )
        await loop.run_in_executor(None, lambda: index.insert_documents(llama_docs))
        index.storage_context.persist(persist_dir=str(storage_dir))
        return True

    async def search(self, query: str, kb_name: str, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        self.logger.info(f"LlamaIndex search in {kb_name}: {query[:50]}...")
        storage_dir = Path(self.kb_base_dir) / kb_name / "llamaindex_storage"
        if not storage_dir.exists():
            raise ValueError(f"LlamaIndex storage not found: {storage_dir}")

        top_k = kwargs.get("top_k", 5)

        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(
            None,
            lambda: load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(storage_dir))
            ),
        )

        # Use retriever instead of query_engine to avoid requiring OpenAI LLM
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = await loop.run_in_executor(None, lambda: retriever.retrieve(query))

        # Combine retrieved content
        chunks = []
        for node in nodes:
            text = node.node.get_content() if hasattr(node.node, "get_content") else str(node.node)
            chunks.append(text)

        content = "\n\n---\n\n".join(chunks) if chunks else ""

        return {
            "query": query,
            "answer": content,
            "content": content,
            "mode": mode,
            "provider": self.name,
        }

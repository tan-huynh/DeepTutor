"""
LightRAG Pipeline
=================

Pipeline wrapper around LightRAG. Accepts pre-parsed documents.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List

from src.logging import get_logger
from src.logging.adapters import LightRAGLogContext
from src.services.embedding import get_embedding_client
from src.services.llm import get_llm_client
from src.services.rag.types import Document

from .base import BasePipeline


class LightRAGPipeline(BasePipeline):
    name = "lightrag"
    _instances: Dict[str, Any] = {}

    def __init__(self, kb_base_dir: str):
        super().__init__(kb_base_dir=kb_base_dir)
        self.logger = get_logger("LightRAGPipeline")

    def _get_instance(self, kb_name: str):
        working_dir = str(Path(self.kb_base_dir) / kb_name / "rag_storage")
        if working_dir in self._instances:
            return self._instances[working_dir]

        from lightrag import LightRAG

        llm_client = get_llm_client()
        embed_client = get_embedding_client()

        # Use unified LLM service interface
        llm_model_func = llm_client.get_model_func()

        # Align LightRAG timeout with our environment config
        llm_timeout = int(os.getenv("LLM_TIMEOUT", 180))
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embed_client.get_embedding_func(),
            llm_model_kwargs={"timeout": llm_timeout},
            default_llm_timeout=llm_timeout,
        )
        self._instances[working_dir] = rag
        return rag

    async def initialize(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        self.logger.info(f"Initializing LightRAG KB '{kb_name}' with {len(documents)} documents")
        with LightRAGLogContext(scene="knowledge_init"):
            from lightrag.kg.shared_storage import initialize_pipeline_status

            rag = self._get_instance(kb_name)
            await rag.initialize_storages()
            await initialize_pipeline_status()
            for doc in documents:
                if doc.content.strip():
                    await rag.ainsert(doc.content, file_paths=doc.file_path)
        return True

    async def add_documents(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        self.logger.info(f"Adding {len(documents)} documents to LightRAG KB '{kb_name}'")
        with LightRAGLogContext(scene="knowledge_add"):
            from lightrag.kg.shared_storage import initialize_pipeline_status

            rag = self._get_instance(kb_name)
            await rag.initialize_storages()
            await initialize_pipeline_status()
            for doc in documents:
                if doc.content.strip():
                    await rag.ainsert(doc.content, file_paths=doc.file_path)
        return True

    async def search(self, query: str, kb_name: str, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        self.logger.info(f"LightRAG search ({mode}) in {kb_name}: {query[:50]}...")
        with LightRAGLogContext(scene="LightRAG-Search"):
            rag = self._get_instance(kb_name)
            await rag.initialize_storages()
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag import QueryParam

            await initialize_pipeline_status()
            query_param = QueryParam(mode=mode, only_need_context=kwargs.get("only_need_context", False))
            answer = await rag.aquery(query, param=query_param)
            answer_str = answer if isinstance(answer, str) else str(answer)
            return {
                "query": query,
                "answer": answer_str,
                "content": answer_str,
                "mode": mode,
                "provider": self.name,
            }

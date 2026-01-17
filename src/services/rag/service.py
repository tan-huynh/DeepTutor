"""
RAG Service
===========

Unified RAG service providing a single entry point for all RAG operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging import get_logger
from src.knowledge.storage import KnowledgeBaseStorage

# Default knowledge base directory
DEFAULT_KB_BASE_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge_bases"
)


class RAGService:
    def __init__(
        self,
        kb_base_dir: Optional[str] = None,
        method: Optional[str] = None,
        storage: Optional[KnowledgeBaseStorage] = None,
    ):
        self.logger = get_logger("RAGService")
        self.kb_base_dir = kb_base_dir or DEFAULT_KB_BASE_DIR
        self.default_method = method or os.getenv("RAG_METHOD", "text-lightrag")
        self.storage = storage or KnowledgeBaseStorage()

    async def initialize(self, kb_name: str, file_paths: List[str], method: Optional[str] = None, **kwargs) -> bool:
        method_id = method or self.default_method
        self.storage.register_kb(kb_name, method_id)
        self.storage.update_progress(kb_name, stage="initializing", percent=10, message="Parsing documents")

        from .factory import get_method

        rag_method = get_method(method_id, kb_base_dir=self.kb_base_dir, storage=self.storage)
        success = await rag_method.initialize(kb_name=kb_name, file_paths=file_paths, **kwargs)

        if success:
            self.storage.set_method(kb_name, method_id)
            self.storage.add_documents(kb_name, file_paths)
            self.storage.update_progress(kb_name, stage="completed", percent=100, message="Initialization completed")
        return success

    async def add_documents(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        method_id = self._get_method_for_kb(kb_name)
        self.storage.update_progress(kb_name, stage="adding", percent=10, message="Adding documents")

        from .factory import get_method

        rag_method = get_method(method_id, kb_base_dir=self.kb_base_dir, storage=self.storage)
        success = await rag_method.add_documents(kb_name=kb_name, file_paths=file_paths, **kwargs)

        if success:
            self.storage.add_documents(kb_name, file_paths)
            self.storage.update_progress(kb_name, stage="completed", percent=100, message="Documents added")
        return success

    async def search(self, query: str, kb_name: str, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        method_id = self._get_method_for_kb(kb_name)
        from .factory import get_method

        rag_method = get_method(method_id, kb_base_dir=self.kb_base_dir, storage=self.storage)
        return await rag_method.search(query=query, kb_name=kb_name, mode=mode, **kwargs)

    async def delete(self, kb_name: str) -> bool:
        kb_dir = Path(self.kb_base_dir) / kb_name
        if kb_dir.exists():
            import shutil

            shutil.rmtree(kb_dir)
        kb_meta = self.storage.user_dir / f"{kb_name}.json"
        if kb_meta.exists():
            kb_meta.unlink()
        if self.storage.get_default_kb() == kb_name:
            self.storage.set_default_kb(None)
        return True

    def _get_method_for_kb(self, kb_name: str) -> str:
        data = self.storage.load_kb(kb_name)
        return data.get("method") or self.default_method

    @staticmethod
    def list_providers() -> List[Dict[str, str]]:
        from .factory import list_methods

        return list_methods()

    @staticmethod
    def get_current_provider() -> str:
        return os.getenv("RAG_METHOD", "text-lightrag")

    @staticmethod
    def has_provider(name: str) -> bool:
        from .factory import has_method

        return has_method(name)

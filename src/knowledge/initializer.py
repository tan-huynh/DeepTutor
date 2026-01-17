#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Base Initialization
"""

from pathlib import Path
from typing import List, Optional
import shutil

from src.logging import get_logger
from src.knowledge.storage import KnowledgeBaseStorage

logger = get_logger("KnowledgeInit")


class KnowledgeBaseInitializer:
    def __init__(
        self,
        kb_name: str,
        base_dir: str = "./data/knowledge_bases",
        method: Optional[str] = None,
        api_key: str | None = None,
        base_url: str | None = None,
        progress_tracker=None,
        rag_provider: str | None = None,
    ) -> None:
        self.kb_name = kb_name
        self.base_dir = Path(base_dir)
        self.kb_dir = self.base_dir / kb_name
        self.raw_dir = self.kb_dir / "raw"
        self.images_dir = self.kb_dir / "images"
        self.content_list_dir = self.kb_dir / "content_list"
        self.rag_storage_dir = self.kb_dir / "rag_storage"
        self.method = method or self._map_provider_to_method(rag_provider)
        self.storage = KnowledgeBaseStorage(project_root=Path(__file__).resolve().parent.parent.parent)
        self.progress_tracker = progress_tracker

    def create_directory_structure(self) -> None:
        for dir_path in [
            self.raw_dir,
            self.images_dir,
            self.content_list_dir,
            self.rag_storage_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _copy_raw_files(self, file_paths: List[str]) -> List[str]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for file_path in file_paths:
            src = Path(file_path)
            if not src.exists():
                raise FileNotFoundError(f"File not found: {src}")
            dest = self.raw_dir / src.name
            if src.resolve() == dest.resolve():
                copied.append(str(dest))
                continue
            shutil.copy2(src, dest)
            copied.append(str(dest))
        return copied

    async def initialize(self, file_paths: List[str]) -> bool:
        from src.services.rag.service import RAGService

        logger.info(f"Initializing knowledge base: {self.kb_name}")
        self.create_directory_structure()
        copied_files = self._copy_raw_files(file_paths)

        service = RAGService(kb_base_dir=str(self.base_dir), storage=self.storage)
        return await service.initialize(
            kb_name=self.kb_name,
            file_paths=copied_files,
            method=self.method,
        )

    async def process_documents(self) -> bool:
        file_paths = [str(p) for p in self.raw_dir.iterdir() if p.is_file()]
        return await self.initialize(file_paths)

    def extract_numbered_items(self) -> None:
        return None

    @staticmethod
    def _map_provider_to_method(provider: str | None) -> str | None:
        if not provider:
            return None
        # Accept both aliases and full method IDs
        mapping = {
            # Aliases
            "raganything": "mineru-lightrag",
            "lightrag": "text-lightrag",
            "llamaindex": "text-llamaindex",
            # Full method IDs (pass through)
            "mineru-lightrag": "mineru-lightrag",
            "text-lightrag": "text-lightrag",
            "text-llamaindex": "text-llamaindex",
            "docling-lightrag": "docling-lightrag",
        }
        return mapping.get(provider)

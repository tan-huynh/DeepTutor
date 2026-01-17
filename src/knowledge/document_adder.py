#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incrementally add documents to existing knowledge base.
"""

from pathlib import Path
from typing import List
import shutil

from src.logging import get_logger
from src.knowledge.storage import KnowledgeBaseStorage

logger = get_logger("KnowledgeAdd")


class DocumentAdder:
    def __init__(
        self,
        kb_name: str,
        base_dir: str = "./data/knowledge_bases",
        api_key: str | None = None,
        base_url: str | None = None,
        progress_tracker=None,
        rag_provider: str | None = None,
    ) -> None:
        self.kb_name = kb_name
        self.base_dir = Path(base_dir)
        self.kb_dir = self.base_dir / kb_name
        self.raw_dir = self.kb_dir / "raw"
        self.storage = KnowledgeBaseStorage(project_root=Path(__file__).resolve().parent.parent.parent)

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

    async def add(self, file_paths: List[str]) -> bool:
        from src.services.rag.service import RAGService

        logger.info(f"Adding documents to knowledge base: {self.kb_name}")
        copied_files = self._copy_raw_files(file_paths)
        service = RAGService(kb_base_dir=str(self.base_dir), storage=self.storage)
        return await service.add_documents(kb_name=self.kb_name, file_paths=copied_files)

    async def process_new_documents(self, file_paths: List[Path]) -> List[Path]:
        copied_files = self._copy_raw_files([str(path) for path in file_paths])
        await self.add(copied_files)
        return [Path(path) for path in copied_files]

    def extract_numbered_items_for_new_docs(self, processed_files: List[Path], batch_size: int = 20) -> None:
        # Numbered items are handled in method pipeline; no-op for compatibility.
        return None

    def update_metadata(self, added_count: int) -> None:
        # Metadata is stored in KnowledgeBaseStorage; no-op for compatibility.
        return None

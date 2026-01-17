#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Base Manager

Manages knowledge base metadata and filesystem locations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging import get_logger
from src.knowledge.storage import KnowledgeBaseStorage

logger = get_logger("KnowledgeBaseManager")


class KnowledgeBaseManager:
    def __init__(self, base_dir: str = "./data/knowledge_bases") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.storage = KnowledgeBaseStorage(project_root=Path(__file__).resolve().parent.parent.parent)

    def list_knowledge_bases(self) -> List[str]:
        stored = set(self.storage.list_kbs())
        disk = set(
            [p.name for p in self.base_dir.iterdir() if p.is_dir() and p.name != "__pycache__"]
        )
        return sorted(stored.union(disk))

    def get_knowledge_base_path(self, name: Optional[str] = None) -> Path:
        if name is None:
            name = self.get_default()
            if name is None:
                raise ValueError("No default knowledge base set")
        kb_dir = self.base_dir / name
        if not kb_dir.exists():
            raise ValueError(f"Knowledge base not found: {name}")
        return kb_dir

    def get_raw_path(self, name: Optional[str] = None) -> Path:
        return self.get_knowledge_base_path(name) / "raw"

    def get_content_list_path(self, name: Optional[str] = None) -> Path:
        return self.get_knowledge_base_path(name) / "content_list"

    def set_default(self, name: str) -> None:
        if name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {name}")
        self.storage.set_default_kb(name)

    def get_default(self) -> Optional[str]:
        return self.storage.get_default_kb()

    def delete(self, name: str) -> None:
        import shutil

        kb_dir = self.base_dir / name
        logger.info(f"Deleting KB '{name}': dir={kb_dir}, exists={kb_dir.exists()}")
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
            logger.info(f"Deleted directory: {kb_dir}")

        meta_path = self.storage.user_dir / f"{name}.json"
        logger.info(f"Meta path: {meta_path}, exists={meta_path.exists()}")
        if meta_path.exists():
            meta_path.unlink()
            logger.info(f"Deleted meta file: {meta_path}")

        if self.get_default() == name:
            self.storage.set_default_kb(None)
            logger.info(f"Cleared default KB")

    def delete_knowledge_base(self, name: str, confirm: bool = False) -> bool:
        if not confirm:
            raise ValueError("Deletion not confirmed")
        if name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {name}")
        self.delete(name)
        return True

    def get_info(self, name: str) -> Dict[str, Any]:
        if name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {name}")
        kb_dir = self.base_dir / name
        raw_dir = kb_dir / "raw"
        images_dir = kb_dir / "images"
        content_list_dir = kb_dir / "content_list"
        rag_storage_dir = kb_dir / "rag_storage"

        def count_files(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(1 for p in path.iterdir() if p.is_file())

        metadata = self.storage.load_kb(name)
        statistics = {
            "raw_documents": count_files(raw_dir),
            "images": count_files(images_dir),
            "content_lists": count_files(content_list_dir),
            "rag_initialized": rag_storage_dir.exists() and any(rag_storage_dir.iterdir()),
        }

        return {
            "name": name,
            "is_default": name == self.get_default(),
            "metadata": metadata,
            "statistics": statistics,
        }

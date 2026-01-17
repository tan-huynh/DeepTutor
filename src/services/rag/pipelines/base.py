from __future__ import annotations

from typing import Any, Dict, List

from src.services.rag.types import Document


class BasePipeline:
    name = "base"

    def __init__(self, kb_base_dir: str):
        self.kb_base_dir = kb_base_dir

    async def initialize(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        raise NotImplementedError

    async def add_documents(self, kb_name: str, documents: List[Document], **kwargs) -> bool:
        raise NotImplementedError

    async def search(self, query: str, kb_name: str, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.logging import get_logger
from src.services.llm import get_llm_client, get_llm_config
from src.services.rag.extras.numbered_item_extractor import extract_numbered_items_with_llm
from src.services.rag.multimodal import MultimodalProcessor
from src.services.rag.parsers.base import BaseParser, ParseResult
from src.services.rag.pipelines.base import BasePipeline
from src.services.rag.types import Document


class BaseMethod:
    method_id = "base"
    description = "Base method"
    parser_cls: Type[BaseParser]
    pipeline_cls: Type[BasePipeline]
    use_multimodal = False
    use_numbered_items = False

    def __init__(self, kb_base_dir: str, storage) -> None:
        self.logger = get_logger(f"RAGMethod:{self.method_id}")
        self.kb_base_dir = kb_base_dir
        self.storage = storage
        self.parser = self.parser_cls()
        self.pipeline = self.pipeline_cls(kb_base_dir=kb_base_dir)

    async def initialize(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        return await self._process(kb_name, file_paths, is_incremental=False)

    async def add_documents(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        return await self._process(kb_name, file_paths, is_incremental=True)

    async def search(self, query: str, kb_name: str, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        return await self.pipeline.search(query=query, kb_name=kb_name, mode=mode, **kwargs)

    async def _process(self, kb_name: str, file_paths: List[str], is_incremental: bool) -> bool:
        kb_dir = Path(self.kb_base_dir) / kb_name
        content_list_dir = kb_dir / "content_list"
        content_list_dir.mkdir(parents=True, exist_ok=True)

        parsed_results: List[ParseResult] = []
        skipped_files: List[str] = []
        for file_path in file_paths:
            if not self.parser.supports(file_path):
                skipped_files.append(file_path)
                continue
            parsed_results.append(self.parser.parse(file_path, output_dir=str(content_list_dir)))

        if skipped_files:
            self.logger.warning(
                f"Skipped {len(skipped_files)} unsupported files for {self.method_id}: "
                f"{', '.join(Path(p).suffix for p in skipped_files)}"
            )
        if not parsed_results:
            raise ValueError(
                f"No supported files for method '{self.method_id}'. "
                f"Please check file types and selected RAG provider."
            )

        documents: List[Document] = []
        all_content_items: List[dict] = []

        multimodal = None
        if self.use_multimodal:
            multimodal = MultimodalProcessor(
                llm_func=self._build_llm_func(),
                vision_func=self._build_vision_func(),
            )

        for result in parsed_results:
            content_list = result.content_list or []
            if content_list:
                all_content_items.extend(content_list)
                self._write_content_list(content_list_dir, result.document.file_path, content_list)

            if self.use_multimodal and multimodal:
                chunks = await multimodal.process_content_list(content_list)
                merged_text = "\n\n".join(chunks)
                doc = Document(
                    content=merged_text,
                    file_path=result.document.file_path,
                    metadata=result.document.metadata,
                    content_items=content_list,
                )
                documents.append(doc)
            else:
                documents.append(result.document)

        if is_incremental:
            await self.pipeline.add_documents(kb_name=kb_name, documents=documents)
        else:
            await self.pipeline.initialize(kb_name=kb_name, documents=documents)

        if self.use_numbered_items and all_content_items:
            self._extract_numbered_items(kb_name, all_content_items)

        return True

    def _write_content_list(self, content_list_dir: Path, file_path: str, content_list: List[dict]) -> None:
        file_stem = Path(file_path).stem
        output_file = content_list_dir / f"{file_stem}.json"
        output_file.write_text(json.dumps(content_list, indent=2, ensure_ascii=False), encoding="utf-8")

    def _extract_numbered_items(self, kb_name: str, content_items: List[dict]) -> None:
        llm_cfg = get_llm_config()
        numbered_items = extract_numbered_items_with_llm(
            content_items,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            batch_size=20,
            max_concurrent=5,
        )
        # Save numbered_items to kb directory (not user storage)
        kb_dir = Path(self.kb_base_dir) / kb_name
        numbered_items_file = kb_dir / "numbered_items.json"
        existing = {}
        if numbered_items_file.exists():
            try:
                existing = json.loads(numbered_items_file.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(numbered_items)
        numbered_items_file.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.logger.info(f"Saved {len(numbered_items)} numbered items to {numbered_items_file}")

    @staticmethod
    def _build_llm_func():
        """Get LLM function using unified LLM service."""
        llm_client = get_llm_client()
        return llm_client.get_model_func()

    @staticmethod
    def _build_vision_func():
        """Get vision model function using unified LLM service."""
        llm_client = get_llm_client()
        return llm_client.get_vision_model_func()


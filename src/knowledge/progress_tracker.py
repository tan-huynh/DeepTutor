#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified progress tracking for knowledge base operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.knowledge.storage import KnowledgeBaseStorage


class ProgressStage(str, Enum):
    INITIALIZING = "initializing"
    PROCESSING_DOCUMENTS = "processing_documents"
    EXTRACTING_ITEMS = "extracting_items"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressTracker:
    kb_name: str
    base_dir: Path
    task_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.storage = KnowledgeBaseStorage(project_root=Path(__file__).resolve().parent.parent.parent)

    def update(self, stage: ProgressStage, message: str, current: int = 0, total: int = 0, error: str = "") -> None:
        percent = 0
        if total > 0:
            percent = int(current / total * 100)
        if stage == ProgressStage.COMPLETED:
            percent = 100
        self.storage.update_progress(self.kb_name, stage=stage.value, percent=percent, message=message)

    def get_progress(self):
        return self.storage.load_kb(self.kb_name).get("progress")

    def clear(self) -> None:
        kb_path = self.storage.user_dir / f"{self.kb_name}.json"
        if not kb_path.exists():
            return  # KB already deleted, nothing to clear
        data = self.storage.load_kb(self.kb_name)
        data["progress"] = {}
        self.storage.save_kb(self.kb_name, data)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Migration helper for knowledge base metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.knowledge.storage import KnowledgeBaseStorage


def _map_provider_to_method(provider: str | None) -> str | None:
    if provider == "raganything":
        return "mineru-lightrag"
    if provider == "lightrag":
        return "text-lightrag"
    if provider == "llamaindex":
        return "text-llamaindex"
    return None


def migrate_all(project_root: Path | None = None) -> None:
    storage = KnowledgeBaseStorage(project_root=project_root)

    for kb_dir in storage.kb_dir.iterdir():
        if not kb_dir.is_dir():
            continue
        kb_name = kb_dir.name
        data = storage.load_kb(kb_name)
        storage.save_kb(kb_name, data)
        numbered_items_path = kb_dir / "numbered_items.json"
        if numbered_items_path.exists():
            try:
                items = json.loads(numbered_items_path.read_text(encoding="utf-8"))
                storage.merge_numbered_items(kb_name, items)
                numbered_items_path.unlink()
            except Exception:
                pass

    settings_path = storage.user_dir.parent / "settings" / "knowledge_base_configs.json"
    if settings_path.exists():
        configs = json.loads(settings_path.read_text(encoding="utf-8"))
        default_kb = configs.get("default_kb")
        if default_kb:
            storage.set_default_kb(default_kb)
        for kb_name, cfg in configs.get("configs", {}).items():
            method = _map_provider_to_method(cfg.get("rag_provider"))
            if method:
                storage.set_method(kb_name, method)


if __name__ == "__main__":
    migrate_all()


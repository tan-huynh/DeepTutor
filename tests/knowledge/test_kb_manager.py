from __future__ import annotations

import json
from pathlib import Path

import pytest
from deeptutor.knowledge.manager import KnowledgeBaseManager


def test_list_knowledge_bases_skips_empty_storage_dirs(tmp_path: Path) -> None:
    base_dir = tmp_path / "knowledge_bases"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "kb_config.json").write_text(json.dumps({"knowledge_bases": {}}), encoding="utf-8")

    empty_legacy = base_dir / "legacy-empty"
    (empty_legacy / "rag_storage").mkdir(parents=True, exist_ok=True)

    empty_llama = base_dir / "llama-empty"
    (empty_llama / "llamaindex_storage").mkdir(parents=True, exist_ok=True)

    valid_kb = base_dir / "valid-kb"
    storage_dir = valid_kb / "llamaindex_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "docstore.json").write_text("{}", encoding="utf-8")

    manager = KnowledgeBaseManager(base_dir=str(base_dir))

    assert manager.list_knowledge_bases() == ["valid-kb"]


def test_list_knowledge_bases_returns_config_entries_when_scan_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "knowledge_bases"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "kb_config.json").write_text(
        json.dumps({"knowledge_bases": {"configured-kb": {"path": "configured-kb"}}}),
        encoding="utf-8",
    )

    manager = KnowledgeBaseManager(base_dir=str(base_dir))

    def _boom(self):
        if self == base_dir:
            raise PermissionError(1, "Operation not permitted", str(base_dir))
        return []

    monkeypatch.setattr(Path, "iterdir", _boom)

    assert manager.list_knowledge_bases() == ["configured-kb"]

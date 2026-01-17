from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logging import get_logger


class KnowledgeBaseStorage:
    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.logger = get_logger("KnowledgeStorage")
        root = project_root or Path(__file__).resolve().parent.parent.parent
        self.user_dir = root / "data" / "user" / "knowledge-base"
        self.kb_dir = root / "data" / "knowledge_bases"
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self.global_path = self.user_dir / "_global.json"

    def list_kbs(self) -> List[str]:
        return sorted(
            [p.stem for p in self.user_dir.glob("*.json") if p.name != "_global.json"]
        )

    def load_global(self) -> Dict[str, Any]:
        if not self.global_path.exists():
            return {"default_kb": None}
        return json.loads(self.global_path.read_text(encoding="utf-8"))

    def save_global(self, data: Dict[str, Any]) -> None:
        self.global_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_default_kb(self) -> Optional[str]:
        return self.load_global().get("default_kb")

    def set_default_kb(self, kb_name: Optional[str]) -> None:
        data = self.load_global()
        data["default_kb"] = kb_name
        self.save_global(data)

    def load_kb(self, kb_name: str) -> Dict[str, Any]:
        kb_path = self.user_dir / f"{kb_name}.json"
        if kb_path.exists():
            return json.loads(kb_path.read_text(encoding="utf-8"))

        legacy = self._migrate_from_metadata(kb_name)
        if legacy:
            self.save_kb(kb_name, legacy)
            return legacy

        return {
            "name": kb_name,
            "method": None,
            "created_at": self._now(),
            "updated_at": self._now(),
            "description": f"Knowledge base: {kb_name}",
            "status": "unknown",
            "documents": [],
            "stats": {},
            "progress": {},
            "numbered_items": {},
        }

    def save_kb(self, kb_name: str, data: Dict[str, Any]) -> None:
        data["updated_at"] = self._now()
        kb_path = self.user_dir / f"{kb_name}.json"
        kb_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def register_kb(self, kb_name: str, method: Optional[str], description: str = "") -> None:
        data = self.load_kb(kb_name)
        data["method"] = method
        if description:
            data["description"] = description
        data["status"] = data.get("status") or "created"
        self.save_kb(kb_name, data)

    def update_progress(self, kb_name: str, stage: str, percent: int, message: str = "") -> None:
        data = self.load_kb(kb_name)
        data["progress"] = {"stage": stage, "percent": percent, "message": message}
        self.save_kb(kb_name, data)

    def add_documents(self, kb_name: str, file_paths: List[str]) -> None:
        data = self.load_kb(kb_name)
        docs = data.get("documents", [])
        for path in file_paths:
            docs.append(
                {
                    "filename": Path(path).name,
                    "path": str(path),
                    "added_at": self._now(),
                    "status": "indexed",
                }
            )
        data["documents"] = docs
        data["stats"]["total_documents"] = len(docs)
        self.save_kb(kb_name, data)

    def merge_numbered_items(self, kb_name: str, new_items: Dict[str, Any]) -> None:
        data = self.load_kb(kb_name)
        existing = data.get("numbered_items", {}) or {}
        existing.update(new_items)
        data["numbered_items"] = existing
        self.save_kb(kb_name, data)

    def set_method(self, kb_name: str, method: str) -> None:
        data = self.load_kb(kb_name)
        data["method"] = method
        self.save_kb(kb_name, data)

    def _migrate_from_metadata(self, kb_name: str) -> Optional[Dict[str, Any]]:
        metadata_file = self.kb_dir / kb_name / "metadata.json"
        if not metadata_file.exists():
            return None
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning(f"Failed to read legacy metadata: {exc}")
            return None
        try:
            metadata_file.unlink()
        except Exception:
            pass

        provider = metadata.get("rag_provider")
        method = None
        if provider == "raganything":
            method = "mineru-lightrag"
        elif provider == "lightrag":
            method = "text-lightrag"
        elif provider == "llamaindex":
            method = "text-llamaindex"

        return {
            "name": metadata.get("name", kb_name),
            "method": method,
            "created_at": metadata.get("created_at", self._now()),
            "updated_at": self._now(),
            "description": metadata.get("description", f"Knowledge base: {kb_name}"),
            "status": "migrated",
            "documents": [],
            "stats": {},
            "progress": {},
            "numbered_items": {},
        }

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


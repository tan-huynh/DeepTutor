from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.logging import get_logger
from src.services.rag.types import Document


@dataclass(frozen=True)
class ParseResult:
    document: Document
    content_list: List[dict]
    markdown: str = ""


class BaseParser:
    name = "base"
    supported_extensions: Sequence[str] = ()

    def __init__(self) -> None:
        self.logger = get_logger(f"RAGParser:{self.name}")

    def supports(self, file_path: str | Path) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions

    def parse(self, file_path: str | Path, output_dir: Optional[str] = None) -> ParseResult:
        raise NotImplementedError

    def parse_many(
        self, file_paths: Iterable[str | Path], output_dir: Optional[str] = None
    ) -> List[ParseResult]:
        results: List[ParseResult] = []
        for file_path in file_paths:
            results.append(self.parse(file_path, output_dir=output_dir))
        return results


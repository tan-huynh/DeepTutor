from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.services.rag.types import Document

from .base import BaseParser, ParseResult


class TextParser(BaseParser):
    name = "text"
    supported_extensions = (".txt", ".md")

    def parse(self, file_path: str | Path, output_dir: Optional[str] = None) -> ParseResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"TextParser does not support '{file_path.suffix}' files. "
                f"Supported: {self.supported_extensions}. "
                f"For PDF files, use 'raganything' or 'mineru-lightrag' provider."
            )

        text = file_path.read_text(encoding="utf-8")
        document = Document(
            content=text,
            file_path=str(file_path),
            metadata={"file_name": file_path.name},
        )
        return ParseResult(document=document, content_list=[], markdown=text)


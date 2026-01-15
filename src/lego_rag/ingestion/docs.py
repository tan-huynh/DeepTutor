from docx import Document as DocxDocument
from pathlib import Path
import asyncio

from .base import Ingestion_BaseLoader
from .errors import EmptyDocumentError

class Doc_Loader(Ingestion_BaseLoader):
    async def load(self, path: str):
        self._validate_path(path)

        def _read_docx():
            doc = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text

        text = await asyncio.to_thread(_read_docx)

        if not text.strip():
            raise EmptyDocumentError(f"DOCX contains no readable text: {path}")

        return [{
            "text": text,
            "source": Path(path).name,
            "metadata": {"filetype": "docx"},
        }]
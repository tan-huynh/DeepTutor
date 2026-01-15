from pypdf import PdfReader
from pathlib import Path
import asyncio

from .base import Ingestion_BaseLoader
from .errors import EmptyDocumentError

class PDFLoader(Ingestion_BaseLoader):
    """
    Loader for PDF files using pypdf.
    """
    async def load(self, path: str):
        self._validate_path(path)

        def _read_pdf():
            reader = PdfReader(path)
            docs = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                except Exception:
                    continue
                if text and text.strip():
                    docs.append({
                        "text": text,
                        "source": Path(path).name,
                        "metadata": {"page": i, "filetype": "pdf"},
                    })
            return docs

        docs = await asyncio.to_thread(_read_pdf)

        if not docs:
            raise EmptyDocumentError(f"No extractable text in PDF: {path}")

        return docs

if __name__ == '__main__':
    import asyncio
    import os

    async def debug_pdf_loader():
        print("--- Debugging PDFLoader ---")
        # Replace with a real path to test
        test_file = "debug_sample.pdf"
        
        if not os.path.exists(test_file):
            print(f"Skipping verify: '{test_file}' not found. Please create one to test.")
            return

        try:
            loader = PDFLoader()
            docs = await loader.load(test_file)
            for d in docs:
                print(f"Loaded page {d['metadata'].get('page')} from {d['source']} | Content len: {len(d['text'])}")
        except Exception as e:
            print(f"Error loading PDF: {e}")

    asyncio.run(debug_pdf_loader())
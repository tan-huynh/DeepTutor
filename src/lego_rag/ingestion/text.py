from pathlib import Path
import asyncio

from .base import Ingestion_BaseLoader
from .errors import EmptyDocumentError

class TextLoader(Ingestion_BaseLoader):
    """
    Loader for plain text (.txt) and markdown (.md) files.
    """
    async def load(self, path: str):
        self._validate_path(path)

        def _read():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        text = await asyncio.to_thread(_read)

        if not text.strip():
            raise EmptyDocumentError(f"Empty text file: {path}")

        return [{
            "text": text,
            "source": Path(path).name,
            "metadata": {"filetype": Path(path).suffix.lstrip(".")},
        }]

if __name__ == "__main__":
    import asyncio
    import os

    async def debug_text_loader():
        print("--- Debugging TextLoader ---")
        test_file = "debug_sample.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a test document for TextLoader.")
        
        try:
            loader = TextLoader()
            docs = await loader.load(test_file)
            for d in docs:
                print(f"Loaded: {d['source']} | Content: {d['text'][:20]}...")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    asyncio.run(debug_text_loader())
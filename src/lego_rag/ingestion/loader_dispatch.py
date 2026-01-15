from pathlib import Path
import asyncio

from .pdf import PDFLoader
from .text import TextLoader
from .docs import Doc_Loader
from .errors import UnsupportedFileTypeError

async def load_documents(path: str) -> list[dict]:
    """
    Dispatch load request to the appropriate loader based on file extension.
    Supported: .pdf, .docx, .txt, .md
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        loader = PDFLoader()
    elif suffix == ".docx":
        loader = Doc_Loader()
    elif suffix in [".txt", ".md"]:
        loader = TextLoader()
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

    return await loader.load(path)

async def load_documents_from_dir(dir_path: str) -> list[dict]:
    """
    Recursively load all supported documents from a directory.
    """
    tasks = []
    for p in Path(dir_path).iterdir():
        if p.is_file():
            # skip hidden / temporary files
            if p.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
                continue
            tasks.append(load_documents(str(p)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    docs = []
    for r in results:
        if isinstance(r, Exception):
            Exception("⚠️ Skipped file due to error: %s", r)
        else:
            docs.extend(r)
    return docs

if __name__ == '__main__':
    import asyncio
    import os
    
    async def debug_dispatch():
        print("--- Debugging Loader Dispatch ---")
        test_file = "debug_dispatch.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Dispatch test content.")
            
        try:
            docs = await load_documents(test_file)
            print(f"Dispatched load for {test_file}: loaded {len(docs)} chunk(s).")
            print(f"Content: {docs[0]['text']}")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    asyncio.run(debug_dispatch())
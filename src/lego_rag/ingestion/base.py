from abc import ABC, abstractmethod
from pathlib import Path

from .errors import FileReadError

class Ingestion_BaseLoader(ABC):
    @abstractmethod
    async def load(self, path: str) -> list[dict]:
        """
        Load file and return list of document dicts:
        [
          {
            "text": str,
            "source": str,       # file path or identifier
            "metadata": dict     # e.g. {"filename":..., "page":..., "author":...}
          }, ...
        ]
        """
        pass

    def _validate_path(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileReadError(f"File does not exist: {path}")
        if not p.is_file():
            raise FileReadError(f"Path is not a file: {path}")
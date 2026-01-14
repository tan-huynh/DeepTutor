from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document

class BaseRetriever(ABC):
    """
    Abstract Base Class for all retrievers.
    Enforces a standard retrieve method.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float, Optional[Document]]]:
        """
        Retrieve documents relevant to the query.

        Args:
            query (str): The query string.
            k (int): Number of documents to retrieve.

        Returns:
            List[Tuple[str, float, Optional[Document]]]: A list of tuples containing
            (chunk_id, score, Document object).
        """
        pass

"""
RAG Service
===========

Unified RAG method service for DeepTutor.
"""

from .factory import get_method, has_method, list_methods
from .service import RAGService
from .types import Chunk, Document, SearchResult

__all__ = [
    "RAGService",
    "Document",
    "Chunk",
    "SearchResult",
    "get_method",
    "list_methods",
    "has_method",
]

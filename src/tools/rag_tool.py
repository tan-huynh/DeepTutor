#!/usr/bin/env python
"""
RAG Query Tool - Pure tool wrapper for RAG operations

This module provides simple function wrappers for RAG operations.
All logic is delegated to RAGService in src/services/rag/service.py.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / "DeepTutor.env", override=False)
load_dotenv(project_root / ".env", override=False)

# Import RAGService as the single entry point
from src.services.rag.service import RAGService


async def rag_search(
    query: str,
    kb_name: Optional[str] = None,
    mode: str = "hybrid",
    method: Optional[str] = None,
    kb_base_dir: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Query knowledge base using configurable RAG pipeline.

    Args:
        query: Query question
        kb_name: Knowledge base name (optional, defaults to default knowledge base)
        mode: Query mode (e.g., "hybrid", "local", "global", "naive")
        method: RAG method to use (defaults to RAG_METHOD env var or "text-lightrag")
        kb_base_dir: Base directory for knowledge bases (for testing)
        **kwargs: Additional parameters passed to the RAG pipeline

    Returns:
        dict: Dictionary containing query results
            {
                "query": str,
                "answer": str,
                "content": str,
                "mode": str,
                "provider": str
            }

    Raises:
        ValueError: If the specified RAG pipeline is not found
        Exception: If the query fails

    Example:
        # Use default method (from .env)
        result = await rag_search("What is machine learning?", kb_name="textbook")

        # Override method
        result = await rag_search("What is ML?", kb_name="textbook", method="text-lightrag")
    """
    service = RAGService(kb_base_dir=kb_base_dir, method=method)

    try:
        return await service.search(query=query, kb_name=kb_name, mode=mode, **kwargs)
    except Exception as e:
        raise Exception(f"RAG search failed: {e}")


async def initialize_rag(
    kb_name: str,
    documents: List[str],
    method: Optional[str] = None,
    kb_base_dir: Optional[str] = None,
    **kwargs,
) -> bool:
    """
    Initialize RAG with documents.

    Args:
        kb_name: Knowledge base name
        documents: List of document file paths to index
        method: RAG method to use (defaults to RAG_METHOD env var)
        kb_base_dir: Base directory for knowledge bases (for testing)
        **kwargs: Additional arguments passed to pipeline

    Returns:
        True if successful

    Example:
        documents = ["doc1.pdf", "doc2.txt"]
        success = await initialize_rag("my_kb", documents)
    """
    service = RAGService(kb_base_dir=kb_base_dir, method=method)
    return await service.initialize(kb_name=kb_name, file_paths=documents, method=method, **kwargs)


async def delete_rag(
    kb_name: str,
    method: Optional[str] = None,
    kb_base_dir: Optional[str] = None,
) -> bool:
    """
    Delete a knowledge base.

    Args:
        kb_name: Knowledge base name
        method: RAG method to use (defaults to RAG_METHOD env var)
        kb_base_dir: Base directory for knowledge bases (for testing)

    Returns:
        True if successful

    Example:
        success = await delete_rag("old_kb")
    """
    service = RAGService(kb_base_dir=kb_base_dir, method=method)
    return await service.delete(kb_name=kb_name)


def get_available_methods() -> List[Dict]:
    """
    Get list of available RAG methods.

    Returns:
        List of pipeline information dictionaries

    Example:
        methods = get_available_methods()
        for m in methods:
            print(f\"{m['name']}: {m['description']}\")
    """
    return RAGService.list_providers()


def get_current_method() -> str:
    """Get the currently configured RAG method"""
    return RAGService.get_current_provider()


# Backward compatibility aliases
get_available_providers = get_available_methods
list_methods = RAGService.list_providers


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # List available methods
    print("Available RAG Methods:")
    for method in get_available_methods():
        print(f"  - {method['id']}: {method['description']}")
    print(f"\nCurrent method: {get_current_method()}\n")

    # Test search (requires existing knowledge base)
    result = asyncio.run(
        rag_search(
            "What is the lookup table (LUT) in FPGA?",
            kb_name="DE-all",
            mode="naive",
        )
    )

    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Method: {result.get('provider', 'unknown')}")

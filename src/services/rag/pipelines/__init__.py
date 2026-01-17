"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

from .lightrag import LightRAGPipeline
from .llamaindex import LlamaIndexPipeline

__all__ = ["LightRAGPipeline", "LlamaIndexPipeline"]

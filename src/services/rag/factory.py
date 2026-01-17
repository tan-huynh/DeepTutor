"""
Method Factory
==============

Factory for creating and managing RAG methods.
"""

from typing import Callable, Dict, List

from .methods import (
    DoclingLightRAGMethod,
    MineruLightRAGMethod,
    TextLlamaIndexMethod,
    TextLightRAGMethod,
)

_METHODS: Dict[str, Callable] = {
    "text-llamaindex": TextLlamaIndexMethod,
    "text-lightrag": TextLightRAGMethod,
    "mineru-lightrag": MineruLightRAGMethod,
    "docling-lightrag": DoclingLightRAGMethod,
}

_ALIASES = {
    "llamaindex": "text-llamaindex",
    "lightrag": "text-lightrag",
    "raganything": "mineru-lightrag",
}


def get_method(name: str, kb_base_dir: str, storage):
    method_id = _ALIASES.get(name, name)
    if method_id not in _METHODS:
        available = list(_METHODS.keys())
        raise ValueError(f"Unknown method: {method_id}. Available: {available}")
    factory = _METHODS[method_id]
    return factory(kb_base_dir=kb_base_dir, storage=storage)


def list_methods() -> List[Dict[str, str]]:
    return [
        {
            "id": "text-llamaindex",
            "name": "Text + LlamaIndex",
            "description": "Text-only pipeline using LlamaIndex vector retrieval.",
        },
        {
            "id": "text-lightrag",
            "name": "Text + LightRAG",
            "description": "Text-only pipeline using LightRAG knowledge graph.",
        },
        {
            "id": "mineru-lightrag",
            "name": "MinerU + LightRAG",
            "description": "PDF multimodal parsing via MinerU with LightRAG and numbered items.",
        },
        {
            "id": "docling-lightrag",
            "name": "Docling + LightRAG",
            "description": "Multi-format parsing via Docling with LightRAG multimodal processing.",
        },
    ]

list_methods_alias = list_methods


def has_method(name: str) -> bool:
    method_id = _ALIASES.get(name, name)
    return method_id in _METHODS

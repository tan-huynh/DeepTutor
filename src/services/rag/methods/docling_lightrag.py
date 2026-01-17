from __future__ import annotations

from src.services.rag.parsers.docling import DoclingParser
from src.services.rag.pipelines.lightrag import LightRAGPipeline

from .base import BaseMethod


class DoclingLightRAGMethod(BaseMethod):
    method_id = "docling-lightrag"
    description = "Docling multi-format + LightRAG with multimodal processing"
    parser_cls = DoclingParser
    pipeline_cls = LightRAGPipeline
    use_multimodal = True


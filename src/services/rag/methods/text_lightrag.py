from __future__ import annotations

from src.services.rag.parsers.text import TextParser
from src.services.rag.pipelines.lightrag import LightRAGPipeline

from .base import BaseMethod


class TextLightRAGMethod(BaseMethod):
    method_id = "text-lightrag"
    description = "Text-only with LightRAG"
    parser_cls = TextParser
    pipeline_cls = LightRAGPipeline


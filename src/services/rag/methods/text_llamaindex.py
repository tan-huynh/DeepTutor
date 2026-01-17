from __future__ import annotations

from src.services.rag.parsers.text import TextParser
from src.services.rag.pipelines.llamaindex import LlamaIndexPipeline

from .base import BaseMethod


class TextLlamaIndexMethod(BaseMethod):
    method_id = "text-llamaindex"
    description = "Text-only with LlamaIndex"
    parser_cls = TextParser
    pipeline_cls = LlamaIndexPipeline


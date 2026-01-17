from __future__ import annotations

from src.services.rag.parsers.mineru import MineruParser
from src.services.rag.pipelines.lightrag import LightRAGPipeline

from .base import BaseMethod


class MineruLightRAGMethod(BaseMethod):
    method_id = "mineru-lightrag"
    description = "MinerU PDF + LightRAG with multimodal processing and numbered items"
    parser_cls = MineruParser
    pipeline_cls = LightRAGPipeline
    use_multimodal = True
    use_numbered_items = True


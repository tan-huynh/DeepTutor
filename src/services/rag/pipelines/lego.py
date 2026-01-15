"""
Pipeline wrapper for the lego_rag implementation.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from src.logging import get_logger
from src.lego_rag.orchestrator import Orchestrator, Router, Judge, FusionEngine, make_rag_flow
from src.lego_rag.indexer.indexer import arun_indexing_pipeline
from src.lego_rag.retriever.pre_retriever import PreRetrievalPipeline
from src.lego_rag.retriever.retriever import RetrievalPipeline, RetrievalConfig
from src.lego_rag.retriever.post_retriever import PostRetrievalPipeline
from src.lego_rag.generation import GenerationPipeline, Generator
from settings import settings


class LegoRagPipeline:
    """
    Wrapper for the lego rag
    """
    
    name = "lego_rag"

    def __init__(self, kb_base_dir: Optional[str] = None):
        self.logger = get_logger("LegoRagPipeline")
        self.kb_base_dir = kb_base_dir or settings.lego_rag.data_path
        self._orchestrator = None

    async def initialize(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        """
        Initialize the knowledge base.
        """
        self.logger.info(f"Initializing lego_rag KB '{kb_name}' with {len(file_paths)} files.")
                
        success_count = 0
        for path in file_paths:
            try:
                self.logger.info(f"Indexing path: {path}")
                await arun_indexing_pipeline(path)
                success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to index {path}: {e}")
        
        return success_count > 0

    def _get_orchestrator(self):
        if self._orchestrator:
            return self._orchestrator
            
        # Initialize components for Orchestrator
        from src.services.llm import get_llm_client
        from src.services.embedding import get_embedding_client        
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        
        config = get_llm_client().config
        llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0,
        )
        
        # Embedding
        embed_model_path = settings.lego_rag.embedding_model_path
        embeddings = HuggingFaceEmbeddings(model_name=embed_model_path)
        
        # Initialize Pipelines
        pre = PreRetrievalPipeline(llm)
        retr_cfg = RetrievalConfig(
            vectorstore=settings.lego_rag.vector_db_path, 
            embedding_model=embeddings, 
            collection_name=settings.lego_rag.collection_name
        )
        retr = RetrievalPipeline(retr_cfg, sparse_index=None, alpha=0.6)
        post = PostRetrievalPipeline(embedding_model=embeddings, llm=llm)
        gen = Generator(llm=llm)
        gen_pipe = GenerationPipeline(gen)

        rag_flow = make_rag_flow(
            name="lego_rag_default",
            pre_retrieval_fn=pre.run,   
            retrieval_fn=retr.retrieve, 
            post_retrieval_fn=post.process,  
            generation_fn=gen_pipe.run,      
        )

        flows = {
            "lego_rag_default": rag_flow,
        }
        router = Router(flow_scores={
            "lego_rag_default": 1,
        })

        judge = Judge(confidence_threshold=0.6)
        fusion = FusionEngine()

        self._orchestrator = Orchestrator(
            flows=flows,
            router=router,
            judge=judge,
            fusion_engine=fusion,
        )
        return self._orchestrator

    async def search(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
        """
        Search using lego_rag orchestrator.
        """
        orchestrator = self._get_orchestrator()
        
        answer = await self._run_in_executor(orchestrator.run, query)
        
        return {
            "query": query,
            "answer": answer,
            "content": answer, # lego_rag orchestrator returns string answer
            "mode": "lego_rag",
            "provider": self.name
        }

    async def _run_in_executor(self, func, *args):
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)
    
    async def delete(self, kb_name: str) -> bool:
        """
        Delete knowledge base.
        """
        # lego_rag stores data in specific paths defined in settings.
        return True

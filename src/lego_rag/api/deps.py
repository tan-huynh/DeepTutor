from typing import Generator as TypeGenerator

from fastapi import Depends
from sqlalchemy.orm import Session

from src.lego_rag.retriever.pre_retriever import PreRetrievalPipeline
from src.lego_rag.retriever.retriever import RetrievalPipeline, RetrievalConfig
from src.lego_rag.retriever.post_retriever import PostRetrievalPipeline
from src.lego_rag.generation import GenerationPipeline, Generator as RAGGenerator
from src.lego_rag.orchestrator import Orchestrator, make_rag_flow, Router, Judge, FusionEngine

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from settings import settings
from .database import get_db

from src.lego_rag.logger import logger

# Singleton implementation to avoid reloading models on every request
_orchestrator_instance = None

def reset_orchestrator():
    global _orchestrator_instance
    _orchestrator_instance = None
    logger.info("Orchestrator singleton reset.")

def get_orchestrator():
    global _orchestrator_instance
    if _orchestrator_instance:
        return _orchestrator_instance

    if not settings.GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in settings.")
        # Return None or raise generic error
        raise ValueError("GROQ_API_KEY not configured")

    # Initialize components
    # Consider making models configurable
    llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.0, api_key=settings.GROQ_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

    pre = PreRetrievalPipeline(llm)
    retr_cfg = RetrievalConfig(vectorstore=settings.VECTOR_DB_PATH, embedding_model=embeddings, collection_name=settings.COLLECTION_NAME)
    retr = RetrievalPipeline(retr_cfg, sparse_index=None, alpha=0.6)
    post = PostRetrievalPipeline(embedding_model=embeddings, llm=llm)
    gen = RAGGenerator(llm=llm)
    gen_pipe = GenerationPipeline(gen)

    rag_flow = make_rag_flow(
        name="default_rag",
        pre_retrieval_fn=pre.run,   
        retrieval_fn=retr.retrieve, 
        post_retrieval_fn=post.process,  
        generation_fn=gen_pipe.run,      
    )

    flows = {
        "default_rag": rag_flow,
    }
    
    # Simple single-flow router for now
    router = Router(flow_scores={
        "default_rag": 1.0,
    })

    judge = Judge(confidence_threshold=0.6)
    fusion = FusionEngine()

    _orchestrator_instance = Orchestrator(
        flows=flows,
        router=router,
        judge=judge,
        fusion_engine=fusion,
    )
    
    return _orchestrator_instance

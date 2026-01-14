from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
import asyncio
import os
import json
from pathlib import Path
from src.logger import logger
from .database import engine, Base
from . import routes

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Lego RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

import shutil
from pathlib import Path
from fastapi import UploadFile, File, HTTPException
from settings import settings
from . import deps
from src.indexer.indexer import arun_indexing_pipeline

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # Create data directory if it doesn't exist
    data_dir = Path(settings.DATA_FOLDER_PATH)
    if data_dir.suffix: # Likely a file
        data_dir = data_dir.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / file.filename
    try:
        file.file.seek(0)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run indexing pipeline
        result = await arun_indexing_pipeline(str(file_path))
        
        # Reset orchestrator to ensure it picks up new data
        deps.reset_orchestrator()
        
        return {"message": "File ingested successfully", "result": result}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@app.post("/sync-data")
async def sync_data():
    try:
        # Index everything in the data folder
        result = await arun_indexing_pipeline(settings.DATA_FOLDER_PATH)
        deps.reset_orchestrator()
        return {"message": "Data synchronized successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-index")
async def clear_index():
    try:
        config = IndexingConfig()
        hierarchy = HierarchicalIndexer(config)
        hierarchy.clear()
        hierarchy.persist_docstore()
        
        # Also clear KG
        if Path(config.kg_path).exists():
            Path(config.kg_path).unlink()
            
        deps.reset_orchestrator()
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kb-stats")
async def get_kb_stats():
    try:
        docstore_path = settings.DOCSTORE_PATH
        # Count documents in docstore
        doc_count = 0
        chunk_count = 0
        if Path(docstore_path).exists():
            with open(docstore_path, "r", encoding="utf-8") as f:
                docstore = json.load(f)
                # In HierarchicalIndexer, docstore keys are chunk IDs (parent or child)
                # So the length of docstore is the total number of chunks.
                chunk_count = len(docstore)
                # We can't easily count "documents" (source files) without iterating metadata
                # checking for unique 'source' or 'parent_id' but for now, 
                # non-zero chunk count is enough to say it's not empty.
                doc_count = chunk_count
        
        return {
            "num_documents": doc_count,
            "num_chunks": chunk_count,
            "is_empty": doc_count == 0
        }
    except Exception as e:
        import logging
        logging.getLogger("lego_rag").error(f"KB Stats error: {e}")
        return {"num_documents": 0, "num_chunks": 0, "is_empty": True, "error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to Lego RAG API"}

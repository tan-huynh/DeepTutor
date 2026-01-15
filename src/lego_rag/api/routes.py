import json
import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Generator, Any

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import models, schemas, deps
from .database import get_db
from .deps import get_orchestrator
from src.lego_rag.orchestrator import Orchestrator
from src.lego_rag.indexer.indexer import arun_indexing_pipeline
from settings import settings

# Initialize router
router = APIRouter(prefix="/chats", tags=["chats"])
from src.lego_rag.logger import logger

@router.post("", response_model=schemas.Chat)
def create_chat(chat_in: schemas.ChatCreate, db: Session = Depends(get_db)):
    chat = models.Chat(title=chat_in.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # Create data directory if it doesn't exist
    data_dir = Path(settings.DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run indexing pipeline
        result = await arun_indexing_pipeline(str(file_path))
        
        # Reset orchestrator to ensure it picks up new data
        deps.reset_orchestrator()
        
        return {"message": "File ingested successfully", "result": result}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@router.get("", response_model=List[schemas.Chat])
def list_chats(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return db.query(models.Chat).order_by(models.Chat.created_at.desc()).offset(skip).limit(limit).all()

@router.get("/{chat_id}", response_model=schemas.Chat)
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(models.Chat).filter(models.Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.post("/{chat_id}/message")
def send_message(
    chat_id: int,
    request: schemas.ChatRequest,
    db: Session = Depends(get_db),
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    chat = db.query(models.Chat).filter(models.Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # 1. Save User Message
    user_msg = models.Message(chat_id=chat_id, role="user", content=request.query)
    db.add(user_msg)
    db.commit()

    # 2. Prepare for Streaming
    # We will collect steps and sources during the stream provided by the callback
    collected_steps = []
    collected_sources = []
    
    def event_stream() -> Generator[str, None, None]:
        # Helper to format SSE
        def sse_format(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        final_answer_container = {"text": ""}

        def callback(event_type: str, data: Any):
            if event_type == "status":
                step = {"step": data}
                collected_steps.append(step)
                yield sse_format({"type": "status", "content": data})
            elif event_type == "sources":
                # 'data' is List[Dict] (contexts)
                # We simplify for frontend if needed, but here just pass it
                nonlocal collected_sources
                collected_sources = data
                yield sse_format({"type": "sources", "content": data})
        
        try:
            # Run Orchestrator (Blocking, but we yield from callback)
            # Since orchestrator.run is synchronous, we can't 'yield' directly inside it easily 
            # unless we run it in a thread and use a queue, OR if the callback itself yields?
            # 
            # WAIT: A callback passed to a sync function cannot yield to the outer generator 
            # unless the sync function itself supports it. 
            # 
            # If Orchestrator.run is strictly sync, 'yield' inside callback won't reach here 
            # unless I use a Queue and a separate thread.
            
            # Let's use a queue-based approach for true streaming from a sync function.
            import queue
            import threading

            q = queue.Queue()
            
            def threaded_callback(event_type: str, data: Any):
                q.put({"type": event_type, "content": data})

            def worker():
                try:
                    ans = orchestrator.run(request.query, callback=threaded_callback)
                    q.put({"type": "answer", "content": ans})
                except Exception as e:
                    logger.error(f"Orchestrator error: {e}")
                    q.put({"type": "error", "content": str(e)})
                finally:
                    q.put(None) # Sentinel

            t = threading.Thread(target=worker)
            t.start()

            while True:
                item = q.get()
                if item is None:
                    break
                
                # Check for answer
                if item["type"] == "answer":
                    final_answer_container["text"] = item["content"]
                    yield sse_format(item)
                elif item["type"] == "error":
                     yield sse_format(item)
                elif item["type"] == "status":
                    collected_steps.append({"step": item["content"]})
                    yield sse_format(item)
                elif item["type"] == "sources":
                    nonlocal collected_sources
                    collected_sources = item["content"]
                    yield sse_format(item)
                else:
                    yield sse_format(item)

            t.join()

            # 3. Save Assistant Message after stream completes
            # Note: We are inside a generator, so DB session might be tricky if not careful, 
            # but 'db' is from outer scope. However, for robustness, we should do it safely.
            # ideally, we save it after the yield loop.
            
        except Exception as e:
             yield sse_format({"type": "error", "content": str(e)})

        # Save to DB (after loop)
        try:
            asst_msg = models.Message(
                chat_id=chat_id,
                role="assistant",
                content=final_answer_container["text"],
                sources=collected_sources,
                process_steps=collected_steps
            )
            # Re-acquire session if needed or use existing 'db' (it is thread-local usually, but here handled by fastapi dependency)
            # Since 'db' session is valid for the request scope, and StreamingResponse keeps request open, it should be fine.
            # But SQLite objects in different threads can be an issue.
            # The worker thread didn't touch DB. This main thread does. usage of 'db' here should be safe 
            # provided we didn't close it.
            
            db.add(asst_msg)
            db.commit()
        except Exception as e:
             logger.error(f"Failed to save history: {e}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
@router.put("/{chat_id}", response_model=schemas.Chat)
def update_chat(chat_id: int, chat_in: schemas.ChatUpdate, db: Session = Depends(get_db)):
    chat = db.query(models.Chat).filter(models.Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat.title = chat_in.title
    db.commit()
    db.refresh(chat)
    return chat

@router.delete("/{chat_id}")
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(models.Chat).filter(models.Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"message": "Chat deleted"}


from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Message Schemas ---

class MessageBase(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = []
    process_steps: Optional[List[Dict[str, Any]]] = []

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: int
    chat_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# --- Chat Schemas ---

class ChatBase(BaseModel):
    title: Optional[str] = "New Chat"

class ChatCreate(ChatBase):
    pass

class ChatUpdate(BaseModel):
    title: str

class Chat(ChatBase):
    id: int
    created_at: datetime
    messages: List[Message] = []

    class Config:
        from_attributes = True

# --- Request/Response specific schemas ---

class ChatRequest(BaseModel):
    query: str

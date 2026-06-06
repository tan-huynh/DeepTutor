"""
Co-Scientist API Router
=======================

REST streaming endpoint for the multi-agent Co-Scientist capability.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deeptutor.capabilities.co_scientist import CoScientistCapability
from deeptutor.core.context import UnifiedContext
from deeptutor.core.stream import StreamEvent, StreamEventType
from deeptutor.core.stream_bus import StreamBus
from deeptutor.logging import get_logger

logger = get_logger("CoScientistAPI")
router = APIRouter()


class CoScientistRunRequest(BaseModel):
    goal: str = ""
    topic: str = ""
    kb_name: str = "default"
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _event_to_sse(event: StreamEvent) -> dict[str, Any] | None:
    module = "co_scientist"
    if event.type == StreamEventType.PROGRESS:
        agent = event.metadata.get("agent") or event.source or module
        return {
            "type": "progress",
            "module": module,
            "data": {
                "status": event.content,
                "stage": event.stage,
                "agent": agent,
                **event.metadata,
            },
        }
    if event.type in {StreamEventType.CONTENT, StreamEventType.THINKING, StreamEventType.OBSERVATION}:
        return {
            "type": "token",
            "module": module,
            "data": {"token": event.content, "stage": event.stage},
        }
    if event.type == StreamEventType.SOURCES:
        return {
            "type": "sources",
            "module": module,
            "data": event.metadata,
        }
    if event.type == StreamEventType.RESULT:
        return {
            "type": "result",
            "module": module,
            "data": event.metadata,
        }
    if event.type == StreamEventType.ERROR:
        return {
            "type": "error",
            "module": module,
            "data": {"message": event.content, **event.metadata},
        }
    if event.type == StreamEventType.DONE:
        return {"type": "complete", "module": module}
    return None


async def _run_stream(request: CoScientistRunRequest) -> AsyncIterator[str]:
    goal = (request.goal or request.topic or "").strip()
    metadata = dict(request.metadata)
    if request.evidence:
        metadata["seed_evidence"] = request.evidence

    capability_options = {
        key: value
        for key, value in request.options.items()
        if key
        in {
            "max_hypotheses",
            "max_evidence",
            "use_web_search",
            "tournament_rounds",
            "temperature",
            "require_evidence",
        }
    }

    context = UnifiedContext(
        session_id=str(request.metadata.get("session_id") or request.metadata.get("conversation_id") or uuid.uuid4()),
        user_message=goal,
        conversation_history=request.history,
        active_capability="co_scientist",
        knowledge_bases=[request.kb_name] if request.kb_name else [],
        config_overrides={**capability_options, **request.config},
        language=str(request.options.get("language") or "en"),
        metadata=metadata,
    )
    capability = CoScientistCapability()
    bus = StreamBus()

    async def _producer() -> None:
        try:
            await capability.run(context, bus)
        except Exception as exc:
            logger.error(f"Co-Scientist run failed: {exc}")
            await bus.error(str(exc), source="co_scientist")
        finally:
            await bus.emit(StreamEvent(type=StreamEventType.DONE, source="co_scientist"))
            await bus.close()

    import asyncio

    task = asyncio.create_task(_producer())
    async for event in bus.subscribe():
        payload = _event_to_sse(event)
        if payload:
            yield _sse(payload)
    await task


@router.post("/run")
async def run_co_scientist(request: CoScientistRunRequest):
    return StreamingResponse(_run_stream(request), media_type="text/event-stream")


@router.post("")
async def run_co_scientist_default(request: CoScientistRunRequest):
    return await run_co_scientist(request)

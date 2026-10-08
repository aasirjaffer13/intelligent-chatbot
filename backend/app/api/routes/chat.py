"""Chat endpoints — thin HTTP wrappers around ChatService.

* ``POST /api/chat``      — one-shot JSON (the stable contract)
* ``POST /api/chat/stream`` — same pipeline, reply delivered as SSE events
  (Phase 10): ``delta`` chunks as they're produced, then a ``meta`` event
  carrying the exact ``ChatResponse`` contract, then ``end``.

Streaming note: the pipeline generates the reply atomically (NLP → agent
→ RAG), so the endpoint chunks the finished reply word-by-word with a
small configurable delay — a streaming *experience* without pretending
the model is token-streaming. True token streaming arrives when providers
expose incremental generation.
"""

from __future__ import annotations

import json
import logging
import re
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.exceptions import NovaError
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+\s*")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat message received (%d chars)", len(request.message))
    return chat_service.handle(request)


@router.post("/chat/stream", summary="Send a message and stream the reply (SSE)")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    from app.config import get_settings

    delay = get_settings().stream_delay_ms / 1000.0
    logger.info("streaming chat requested (%d chars)", len(request.message))

    def generate():
        try:
            response = chat_service.handle(request)
        except NovaError as exc:
            yield _sse("error", {"code": exc.code, "detail": exc.detail})
            return
        except Exception:
            logger.exception("streaming chat failed")
            yield _sse("error", {"code": "internal_error", "detail": "Internal server error"})
            return

        for piece in _WORD_RE.findall(response.response):
            yield _sse("delta", {"text": piece})
            if delay:
                time.sleep(delay)
        yield _sse("meta", response.model_dump(mode="json"))
        yield _sse("end", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # never buffer (nginx etc.)
        },
    )

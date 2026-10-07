"""POST /api/chat — thin HTTP wrapper around ChatService.

The NLP pipeline lives in ``app/services/chat_service.py``; this module only
parses the request, invokes the service, and returns the contract schema.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat message received (%d chars)", len(request.message))
    return chat_service.handle(request)

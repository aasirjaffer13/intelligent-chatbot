"""Conversation browsing for the sidebar (Phase 10).

* ``GET /api/conversations`` — newest-first list with counts + previews
* ``GET /api/conversations/{id}/messages`` — full history of one session
  (the frontend replays it into the chat pane)

Both read through ``chat_service.memory`` — the same store the chat
pipeline writes, so what the sidebar shows is exactly what memory holds.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.exceptions import NotFoundError
from app.memory import ConversationSummary
from app.schemas import (
    ConversationItem,
    ConversationListResponse,
    ConversationMessagesResponse,
    MessageItem,
)
from app.services.chat_service import chat_service

router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 140


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List conversations (sidebar)",
)
def list_conversations() -> ConversationListResponse:
    summaries = chat_service.memory.list_conversations()
    return ConversationListResponse(conversations=[_to_item(s) for s in summaries])


@router.get(
    "/conversations/{session_id}/messages",
    response_model=ConversationMessagesResponse,
    summary="Full message history of one conversation",
)
def conversation_messages(session_id: str) -> ConversationMessagesResponse:
    memory = chat_service.memory
    if not any(s.session_id == session_id for s in memory.list_conversations()):
        raise NotFoundError(f"conversation '{session_id}' not found")
    messages = memory.recent_messages(session_id, 10_000)
    return ConversationMessagesResponse(
        session_id=session_id,
        messages=[
            MessageItem(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                intent=message.intent,
                confidence=message.confidence,
            )
            for message in messages
        ],
    )


def _to_item(summary: ConversationSummary) -> ConversationItem:
    preview = summary.preview.strip()
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[: _PREVIEW_CHARS - 1].rstrip() + "…"
    return ConversationItem(
        session_id=summary.session_id,
        message_count=summary.message_count,
        preview=preview or "(empty)",
        updated_at=summary.updated_at,
    )

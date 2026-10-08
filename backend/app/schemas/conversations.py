"""Schemas for GET /api/conversations (Phase 10 sidebar)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ConversationItem(BaseModel):
    """One conversation row for the sidebar list."""

    session_id: str
    message_count: int = Field(ge=0)
    preview: str = Field(description="Last message, trimmed for the sidebar.")
    updated_at: datetime = Field(description="When the conversation last changed.")


class ConversationListResponse(BaseModel):
    conversations: list[ConversationItem]


class MessageItem(BaseModel):
    """One stored turn, in chronological order."""

    role: str = Field(description='"user" or "assistant".')
    content: str
    created_at: datetime
    intent: str | None = None
    confidence: float | None = None


class ConversationMessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]

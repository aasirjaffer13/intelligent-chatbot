"""Storage-agnostic conversation memory (Phase 6).

Two interchangeable implementations of one interface:

* ``InMemoryStore`` — dict-backed, zero dependencies. Default for dev/tests
  and the fallback when no database URL is configured.
* ``SqlAlchemyMemoryStore`` (``app.memory.db``) — the same operations on
  PostgreSQL (SQLite works too — that is how the SQL store is tested).

The chat service only ever sees the ``MemoryStore`` interface, so swapping
the backing store is a constructor argument — never a code change.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class MemoryMessage:
    """One stored turn."""

    role: str                # "user" | "assistant"
    content: str
    created_at: datetime
    intent: str | None = None
    confidence: float | None = None


class MemoryStore(ABC):
    """Operations the chat pipeline needs. All are synchronous and cheap."""

    @abstractmethod
    def get_or_create_conversation(self, session_id: str | None = None) -> str:
        """Return a usable session id, creating the conversation if needed.

        Passing None generates a fresh id — the API uses this to *create*
        sessions on first contact.
        """

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Append one turn (and bump the conversation's recency)."""

    @abstractmethod
    def recent_messages(self, session_id: str, limit: int = 12) -> list[MemoryMessage]:
        """Newest ``limit`` messages, oldest-first order (a context window)."""


def new_session_id() -> str:
    """Compact, URL-safe, collision-free session identifier."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryStore(MemoryStore):
    """Dict-backed store. Process-local: history dies with the process."""

    def __init__(self) -> None:
        self._messages: dict[str, list[MemoryMessage]] = {}

    def get_or_create_conversation(self, session_id: str | None = None) -> str:
        session_id = session_id or new_session_id()
        self._messages.setdefault(session_id, [])
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> None:
        bucket = self._messages.setdefault(session_id, [])
        bucket.append(
            MemoryMessage(
                role=role,
                content=content,
                created_at=utcnow(),
                intent=intent,
                confidence=confidence,
            )
        )

    def recent_messages(self, session_id: str, limit: int = 12) -> list[MemoryMessage]:
        if limit <= 0:
            return []
        return self._messages.get(session_id, [])[-limit:]

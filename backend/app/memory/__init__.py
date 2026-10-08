"""Conversation memory (Phase 6) — public surface.

``get_memory_store()`` chooses the backing store from configuration:

* ``NOVA_DATABASE_URL`` set (e.g. ``postgresql+psycopg://nova:nova@localhost:5432/nova``)
  -> SQL store (PostgreSQL in production, SQLite for local experiments);
* unset/unreachable -> process-local ``InMemoryStore`` (dev default, tests).

Either way the chat service talks to the same ``MemoryStore`` interface.
"""

from __future__ import annotations

import logging

from app.memory.base import (
    ConversationSummary,
    InMemoryStore,
    MemoryMessage,
    MemoryStore,
    new_session_id,
)
from app.memory.context import (
    MemoryContext,
    build_context,
    extract_name,
    is_name_question,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationSummary",
    "InMemoryStore",
    "MemoryContext",
    "MemoryMessage",
    "MemoryStore",
    "build_context",
    "extract_name",
    "get_memory_store",
    "is_name_question",
    "new_session_id",
]


def get_memory_store(database_url: str | None = None) -> MemoryStore:
    """Build the configured store. Falls back to in-memory on any failure so
    the API keeps serving (a chatbot must not 500 because the DB is down).
    """
    if database_url is None:
        from app.config import get_settings

        database_url = get_settings().database_url
    if not database_url:
        return InMemoryStore()
    try:
        from app.memory.db import SqlAlchemyMemoryStore

        store = SqlAlchemyMemoryStore(database_url)
        logger.info("memory store: SQL (%s)", _describe(database_url))
        return store
    except Exception:
        logger.exception("database unavailable — falling back to in-memory store")
        return InMemoryStore()


def _describe(database_url: str) -> str:
    """URL without credentials, safe for logs."""
    scheme, _, rest = database_url.partition("://")
    if "@" in rest:
        rest = rest.rsplit("@", 1)[1]
    return f"{scheme}://{rest}"

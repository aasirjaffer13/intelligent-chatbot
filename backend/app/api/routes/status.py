"""GET /api/status — what this process is actually running with (Phase 10).

The frontend renders this as the model/status indicator: which LLM
provider answers (or that none does), pgvector vs numpy retrieval,
memory backend, configured intent backend. Every section degrades to an
explicit "unavailable/error" value instead of failing the endpoint —
status must work precisely when other things don't.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import (
    AgentStatus,
    IntentStatus,
    LlmStatus,
    MemoryStatus,
    RagStatus,
    StatusResponse,
)

router = APIRouter(tags=["status"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=StatusResponse, summary="Runtime configuration snapshot")
def status() -> StatusResponse:
    settings = get_settings()

    # LLM provider (auto-resolve; misconfiguration reports as an error, not a 500)
    try:
        from app.llm import LLMError, get_llm_provider

        try:
            provider = get_llm_provider()
        except LLMError as exc:
            llm = LlmStatus(provider=f"error: {exc}")
        else:
            llm = LlmStatus(
                provider=provider.name if provider else "none",
                model=getattr(provider, "model", None) if provider else None,
            )
    except Exception:  # pragma: no cover - defensive
        logger.exception("status: llm probe failed")
        llm = LlmStatus(provider="unavailable")

    # RAG store mode + document count
    try:
        from app.rag import get_rag_store

        store = get_rag_store()
        rag = RagStatus(
            mode="pgvector" if store.vector_enabled else "numpy",
            documents=len(store.list_documents()),
        )
    except Exception:
        logger.exception("status: rag probe failed")
        rag = RagStatus(mode="unavailable", documents=0)

    # Memory backend straight from configuration
    url = settings.database_url
    if url.startswith(("postgresql", "postgres")):
        backend = "postgresql"
    elif url.startswith("sqlite"):
        backend = "sqlite"
    else:
        backend = "in-memory"

    return StatusResponse(
        llm=llm,
        agent=AgentStatus(max_steps=settings.agent_max_steps),
        rag=rag,
        memory=MemoryStatus(backend=backend),
        intent=IntentStatus(backend=settings.intent_backend),
    )

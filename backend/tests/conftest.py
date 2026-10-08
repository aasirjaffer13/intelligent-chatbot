"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nlp.nltk_data import ensure_nltk_data

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def nltk_data_ready() -> None:
    """Make sure NLTK corpora exist before any NLP test runs.

    If this fails (e.g. offline with no cached data), NLP tests will fail with
    visible assertion errors rather than mysterious crashes mid-pipeline.
    """
    try:
        ensure_nltk_data()
    except Exception as exc:
        logger.warning("could not ensure NLTK data: %s", exc)


@pytest.fixture()
def client(tmp_path) -> TestClient:
    """HTTP client wired to the in-process FastAPI app.

    Isolates every test: RAG store on a private SQLite file, upload dir in
    tmp, chat memory as a fresh in-memory store (no PG pollution, no state
    leaking between tests).
    """
    from app.memory import InMemoryStore
    from app.rag import get_document_service, get_rag_store
    from app.rag.service import DocumentService, RagStore
    from app.services import chat_service as chat_module
    from app.services.embedding_service import get_embedding_service

    rag_store = RagStore(f"sqlite:///{tmp_path / 'rag.db'}")
    doc_service = DocumentService(
        rag_store, get_embedding_service(), tmp_path / "uploads"
    )

    app.dependency_overrides[get_rag_store] = lambda: rag_store
    app.dependency_overrides[get_document_service] = lambda: doc_service

    # The chat route uses the module singleton — point it at this test's
    # stores too, and restore afterwards.
    previous_rag = chat_module.chat_service._rag_store
    previous_memory = chat_module.chat_service.memory
    chat_module.chat_service._rag_store = rag_store
    chat_module.chat_service.memory = InMemoryStore()

    with TestClient(app) as test_client:
        yield test_client

    chat_module.chat_service._rag_store = previous_rag
    chat_module.chat_service.memory = previous_memory
    app.dependency_overrides.clear()

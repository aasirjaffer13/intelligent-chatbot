"""Document ingestion + retrieval orchestration (Phase 7).

``DocumentService`` — upload pipeline: extract text → chunk → embed →
persist (rows + original file). Also list/delete.

``Retriever`` — query → embedding → store.search → scored chunks. Shared by
the chat pipeline (document_question routing) and future agent tools.

Both take their dependencies (store, embedder, storage dir) by injection;
``get_document_service``/``get_rag_store`` wire the production singletons
and are what FastAPI ``Depends`` and tests override.
"""

from __future__ import annotations

import logging
import re
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.rag.chunker import chunk_text
from app.rag.extractor import extract_text
from app.rag.store import DocumentInfo, RagStore, ScoredChunk

logger = logging.getLogger(__name__)

_NOVA_DIR = Path(__file__).resolve().parents[3]  # .../nova
DEFAULT_STORAGE_DIR = _NOVA_DIR / "data" / "documents"


class DocumentService:
    """Owns the upload -> chunks -> vectors flow and file storage."""

    def __init__(self, store: RagStore, embedder, storage_dir: Path | str) -> None:
        self.store = store
        self._embedder = embedder
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> DocumentInfo:
        text = extract_text(filename, data)
        chunks = chunk_text(text)
        if not chunks:
            from app.rag.extractor import EmptyDocumentError

            raise EmptyDocumentError(f"no text to chunk in {filename!r}")

        embeddings = self._embedder.encode([c.text for c in chunks])
        stored_path = self._store_file(filename, data)
        info = self.store.add_document(
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(data),
            stored_path=str(stored_path),
            chunks=[c.text for c in chunks],
            embeddings=embeddings,
        )
        logger.info(
            "ingested %r: %d chunks, %d bytes (search=%s)",
            filename,
            info.chunk_count,
            info.size_bytes,
            "pgvector" if self.store.vector_enabled else "numpy",
        )
        return info

    def list(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete(self, document_id: str) -> bool:
        stored_path = self.store.delete_document(document_id)
        if stored_path is None:
            return False
        try:
            Path(stored_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove file %s", stored_path)
        return True

    def _store_file(self, filename: str, data: bytes) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name) or "upload.bin"
        path = self.storage_dir / f"{uuid.uuid4().hex[:12]}_{safe}"
        path.write_bytes(data)
        return path


class Retriever:
    """Query-side of RAG: embed the question, fetch top-k chunks."""

    def __init__(self, store: RagStore, embedder) -> None:
        self.store = store
        self._embedder = embedder

    def search(self, query: str, *, top_k: int = 4) -> list[ScoredChunk]:
        vector = self._embedder.embed(query)
        if not vector.any():
            return []
        return self.store.search(vector, top_k=top_k)


# --- production wiring (FastAPI Depends + tests override these) -------------


@lru_cache(maxsize=1)
def get_rag_store() -> RagStore:
    """SQL store from NOVA_DATABASE_URL; falls back to a local SQLite file
    so RAG keeps working with no database configured (never raises)."""
    from app.config import get_settings

    url = get_settings().database_url
    if url:
        try:
            return RagStore(url)
        except Exception:
            logger.exception("RAG store: database unreachable — using sqlite fallback")
    fallback = DEFAULT_STORAGE_DIR / "rag_fallback.db"
    return RagStore(f"sqlite:///{fallback}")


def get_document_service() -> DocumentService:
    """Production DocumentService (dependencies fetched lazily so a missing
    embedding model fails the *request* with 503, not app startup)."""
    from app.config import get_settings
    from app.services.embedding_service import get_embedding_service

    settings = get_settings()
    storage_dir = settings.document_dir or str(DEFAULT_STORAGE_DIR)
    return DocumentService(get_rag_store(), get_embedding_service(), storage_dir)


def get_retriever() -> Retriever:
    from app.services.embedding_service import get_embedding_service

    return Retriever(get_rag_store(), get_embedding_service())

"""Document + chunk storage with vector search (Phase 7).

Dual-mode vector search, selected automatically at startup:

* **pgvector mode** — when ``CREATE EXTENSION vector`` succeeds (files
  installed + privileges), chunks get a real ``vector(384)`` column and
  search runs inside Postgres with the cosine operator ``<=>``. This is the
  production shape: vector index + relational data in one engine.
* **numpy fallback** — embeddings stored as float32 ``bytea``; search loads
  them and computes dot products in process (vectors are unit-length, so
  dot = cosine). Perfectly fine up to tens of thousands of chunks
  (50k × 384 × 4 B ≈ 77 MB) and works on *any* SQLAlchemy URL — which is
  how this mode is tested everywhere, including SQLite.

Either way the schema is the same; only the search path differs, so swapping
in pgvector later (install extension, restart) requires **zero code change**.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.memory.base import utcnow

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    stored_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ORM relationship (not just the FK) so the unit-of-work inserts the
    # parent before chunks and delete-orphan cascades on every backend.
    # Without it SQLAlchemy has no dependency graph and PostgreSQL rejects
    # the chunk inserts (SQLite only "worked" because its FK checks are
    # off by default).
    chunks: Mapped[list["ChunkRow"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkRow.chunk_index",
    )


class ChunkRow(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # float32, unit-norm

    document: Mapped["DocumentRow"] = relationship(back_populates="chunks")


@dataclass(frozen=True)
class DocumentInfo:
    """Upload metadata as seen by the API."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with its query similarity."""

    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class RagStore:
    """Documents + chunks + vector search behind one object."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self._engine = create_engine(database_url, echo=echo, future=True)
        Base.metadata.create_all(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)
        self.vector_enabled = self._try_enable_pgvector()

    # --- capability detection ------------------------------------------------

    def _try_enable_pgvector(self) -> bool:
        try:
            with self._engine.begin() as conn:
                conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
                conn.exec_driver_sql(
                    f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS "
                    f"embedding_vec vector({EMBEDDING_DIM})"
                )
            logger.info("pgvector mode ACTIVE (search runs inside Postgres)")
            return True
        except Exception as exc:
            logger.info("pgvector unavailable (%s) — using numpy cosine fallback", exc)
            return False

    # --- writes --------------------------------------------------------------

    def add_document(
        self,
        *,
        filename: str,
        content_type: str,
        size_bytes: int,
        stored_path: str | None,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> DocumentInfo:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align")
        document_id = uuid.uuid4().hex
        vectors = np.asarray(embeddings, dtype=np.float32)

        with self._sessions() as session:
            document = DocumentRow(
                id=document_id,
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                chunk_count=len(chunks),
                stored_path=stored_path,
            )
            document.chunks = [
                ChunkRow(
                    document_id=document_id,
                    chunk_index=index,
                    content=content,
                    embedding=vectors[index].tobytes(),
                )
                for index, content in enumerate(chunks)
            ]
            session.add(document)
            session.commit()
            created_at = document.created_at
        return DocumentInfo(
            id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            chunk_count=len(chunks),
            created_at=created_at,
        )

    # --- reads ---------------------------------------------------------------

    def list_documents(self) -> list[DocumentInfo]:
        with self._sessions() as session:
            rows = session.scalars(
                select(DocumentRow).order_by(DocumentRow.created_at.desc())
            ).all()
            return [_to_info(row) for row in rows]

    def delete_document(self, document_id: str) -> str | None:
        """Delete a document + its chunks. Returns stored_path (for file
        cleanup) or None when the id does not exist."""
        with self._sessions() as session:
            row = session.get(DocumentRow, document_id)
            if row is None:
                return None
            stored_path = row.stored_path
            session.delete(row)  # ORM cascade removes chunks on all backends
            session.commit()
            return stored_path

    # --- search --------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 4,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity (unit vectors in, scores out)."""
        if top_k < 1:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        if query.size != EMBEDDING_DIM:
            raise ValueError(f"query vector must have dim {EMBEDDING_DIM}")

        if self.vector_enabled:
            return self._search_pgvector(query, top_k, document_ids)
        return self._search_numpy(query, top_k, document_ids)

    def _search_pgvector(
        self, query: np.ndarray, top_k: int, document_ids: list[str] | None
    ) -> list[ScoredChunk]:
        from sqlalchemy import text

        literal = "[" + ",".join(f"{value:.6g}" for value in query) + "]"
        sql = (
            "SELECT c.document_id, d.filename, c.chunk_index, c.content, "
            "1 - (c.embedding_vec <=> CAST(:q AS vector)) AS score "
            "FROM document_chunks c "
            "JOIN documents d ON d.id = c.document_id"
        )
        params: dict = {"q": literal, "k": top_k}
        if document_ids:
            placeholders = ",".join(f":id{i}" for i in range(len(document_ids)))
            sql += f" WHERE c.document_id IN ({placeholders})"
            params.update({f"id{i}": value for i, value in enumerate(document_ids)})
        sql += " ORDER BY score DESC LIMIT :k"

        with self._sessions() as session:
            rows = session.execute(text(sql), params).all()
        return [
            ScoredChunk(
                document_id=row[0],
                filename=row[1],
                chunk_index=int(row[2]),
                content=row[3],
                score=float(row[4]),
            )
            for row in rows
        ]

    def _search_numpy(
        self, query: np.ndarray, top_k: int, document_ids: list[str] | None
    ) -> list[ScoredChunk]:
        with self._sessions() as session:
            stmt = (
                select(ChunkRow.id, DocumentRow.filename, ChunkRow)
                .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
            )
            if document_ids:
                stmt = stmt.where(ChunkRow.document_id.in_(document_ids))
            rows = session.execute(stmt).all()

            if not rows:
                return []

            embeddings = np.vstack(
                [np.frombuffer(chunk.embedding, dtype=np.float32) for _, _, chunk in rows]
            )
            # unit-norm rows -> cosine = dot product
            scores = embeddings @ query
            order = np.argsort(-scores)[:top_k]
            return [
                ScoredChunk(
                    document_id=rows[i][2].document_id,
                    filename=rows[i][1],
                    chunk_index=rows[i][2].chunk_index,
                    content=rows[i][2].content,
                    score=float(scores[i]),
                )
                for i in order
            ]


def _to_info(row: DocumentRow) -> DocumentInfo:
    return DocumentInfo(
        id=row.id,
        filename=row.filename,
        content_type=row.content_type,
        size_bytes=row.size_bytes,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )

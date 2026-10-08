"""SQLAlchemy-backed memory store (Phase 6).

Schema (created via ``create_all`` on first use — this project's dev-scale
schema management; production would manage the same tables with Alembic):

* ``conversations`` — one row per session id, ``created_at``/``updated_at``
* ``messages``       — turns with role, content, intent, confidence, timestamp

Works identically on PostgreSQL (production) and SQLite (tests) — the store
only uses portable SQLAlchemy Core/ORM constructs.
"""

from __future__ import annotations

import logging

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
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

from app.memory.base import MemoryMessage, MemoryStore, new_session_id, utcnow

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    messages: Mapped[list["MessageRow"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageRow.id",
    )


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[ConversationRow] = relationship(back_populates="messages")


class SqlAlchemyMemoryStore(MemoryStore):
    """MemoryStore on any SQLAlchemy URL (postgresql+psycopg://, sqlite:///)."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self._engine = create_engine(database_url, echo=echo, future=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)

    def _conversation(self, session: Session, session_id: str) -> ConversationRow | None:
        return session.scalar(
            select(ConversationRow).where(ConversationRow.session_id == session_id)
        )

    def get_or_create_conversation(self, session_id: str | None = None) -> str:
        session_id = session_id or new_session_id()
        with self._session_factory() as session:
            row = self._conversation(session, session_id)
            if row is None:
                session.add(ConversationRow(id=new_session_id(), session_id=session_id))
                session.commit()
                logger.info("created conversation session=%s", session_id)
            else:
                row.updated_at = utcnow()
                session.commit()
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        confidence: float | None = None,
    ) -> None:
        with self._session_factory() as session:
            conversation = self._conversation(session, session_id)
            if conversation is None:
                conversation = ConversationRow(id=new_session_id(), session_id=session_id)
                session.add(conversation)
            session.add(
                MessageRow(
                    conversation=conversation,
                    role=role,
                    content=content,
                    intent=intent,
                    confidence=confidence,
                )
            )
            conversation.updated_at = utcnow()
            session.commit()

    def recent_messages(self, session_id: str, limit: int = 12) -> list[MemoryMessage]:
        if limit <= 0:
            return []
        with self._session_factory() as session:
            conversation = self._conversation(session, session_id)
            if conversation is None:
                return []
            rows = list(conversation.messages)[-limit:]
            return [
                MemoryMessage(
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    intent=row.intent,
                    confidence=row.confidence,
                )
                for row in rows
            ]

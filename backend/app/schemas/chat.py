"""Schemas for POST /api/chat.

The response shape is defined NOW (contract-first) and stays stable across all
future phases. Phase 1 returns placeholder fields; Phases 2-9 progressively
fill them with real NLP output instead of changing the contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming chat message."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Raw user message.",
        examples=["hello, how are you?"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Conversation session id. Optional until memory lands in Phase 6.",
    )

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace-only")
        return stripped


class Entity(BaseModel):
    """A detected span of text (Phase 4 fills this with real extractions)."""

    text: str = Field(description="Surface form found in the message.")
    label: str = Field(description="Entity type, e.g. PERSON, LOCATION, DATE.")
    start: int = Field(description="Character offset of the span start.")
    end: int = Field(description="Character offset of the span end (exclusive).")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extractor confidence; rule-based methods return 1.0.",
    )


class ProcessingInfo(BaseModel):
    """Diagnostic view into the NLP pipeline for learning/debugging.

    Populated by the preprocessing pipeline (Phase 2). More fields may be
    *added* here in later phases (never renamed/removed — the contract is
    additive only).
    """

    tokens: list[str] = Field(default_factory=list, description="Final pipeline tokens.")
    normalized_text: str = Field(default="", description="Cleaned + lowercased text.")
    sentences: list[str] = Field(default_factory=list, description="Detected sentences.")


class Source(BaseModel):
    """A retrieved document chunk backing a grounded answer (Phase 7).

    Added additively to the contract: older clients ignore the field.
    """

    document_id: str = Field(description="The uploaded document this chunk came from.")
    filename: str = Field(description="Original filename, for citation display.")
    chunk_index: int = Field(description="0-based chunk position within the document.")
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity to the query.")
    quote: str = Field(description="The excerpt the answer draws from.")


class ChatResponse(BaseModel):
    """Outgoing chat payload."""

    response: str = Field(description="Generated reply.")
    intent: str = Field(description="Detected intent label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence.")
    entities: list[Entity] = Field(default_factory=list)
    sources: list[Source] = Field(
        default_factory=list,
        description="RAG citations when the reply is grounded in documents (Phase 7).",
    )
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)
    session_id: str | None = Field(
        default=None,
        description="Session id, echoed/created for conversation memory (Phase 6).",
    )

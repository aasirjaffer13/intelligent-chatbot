"""Schemas for document management (Phase 7 RAG)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentInfoResponse(BaseModel):
    """Metadata for an uploaded document."""

    id: str = Field(description="Document id.")
    filename: str = Field(description="Original filename.")
    content_type: str = Field(description="MIME type of the upload.")
    size_bytes: int = Field(description="Stored file size in bytes.")
    chunk_count: int = Field(description="Number of embedded chunks.")
    created_at: datetime = Field(description="Upload time (UTC).")

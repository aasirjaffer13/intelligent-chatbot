"""Schema for GET /api/health."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload — kept deliberately minimal and stable."""

    status: str = Field(examples=["healthy"])
    service: str = Field(examples=["nova"])

"""Schema for GET /api/status (Phase 10 model/status indicator).

Reports what is *actually configured/resolved* in this process: which
LLM provider answered last, whether vector search runs inside Postgres
or the numpy fallback, which memory backend serves sessions, and so on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LlmStatus(BaseModel):
    provider: str = Field(description="openai | huggingface | local | mock | none")
    model: str | None = Field(default=None, description="Model name, when applicable.")


class AgentStatus(BaseModel):
    max_steps: int = Field(description="Max LLM/tool rounds per turn.")


class RagStatus(BaseModel):
    mode: str = Field(description="pgvector | numpy | unavailable")
    documents: int = Field(ge=0, description="Uploaded documents stored.")


class MemoryStatus(BaseModel):
    backend: str = Field(description="postgresql | sqlite | in-memory")


class IntentStatus(BaseModel):
    backend: str = Field(description="Configured intent backend (auto|sklearn|embedding|keyword).")


class StatusResponse(BaseModel):
    llm: LlmStatus
    agent: AgentStatus
    rag: RagStatus
    memory: MemoryStatus
    intent: IntentStatus

"""LLM provider abstraction (Phase 8).

Why an abstraction: the *pipeline* owns NLP decisions (intent, entities,
memory, RAG routing); the LLM only **phrases** the reply. Swapping
OpenAI for a local model — or for no model at all — must never touch
chat_service. Every provider speaks the same two types:

    LLMRequest  (prompt + system + generation knobs)
    LLMResponse (text + provider/model metadata)

Failures raise LLMError; the chat pipeline catches it and falls back to
deterministic templates. A provider outage degrades the product, it never
breaks it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LLMRequest:
    """A single completion request, provider-agnostic."""

    prompt: str
    system: str | None = None
    max_tokens: int = 256
    temperature: float = 0.2


@dataclass(frozen=True)
class LLMResponse:
    """What came back, plus enough metadata to log/trace it."""

    text: str
    provider: str
    model: str
    latency_ms: float = 0.0


class LLMError(Exception):
    """Provider failure (auth, network, bad payload, empty completion)."""


class LLMProvider(ABC):
    """Contract every provider implements — one method, sync, raises LLMError."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a completion for ``request`` or raise :class:`LLMError`."""

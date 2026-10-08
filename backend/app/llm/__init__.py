"""LLM integration package (Phase 8).

``get_llm_provider()`` resolves the configured provider once:

* ``auto``  — OpenAI if a key is present, HuggingFace if a token is
  present, otherwise **None** (chat keeps working on templates; this is
  the default on a machine with no credentials).
* explicit ``openai`` / ``huggingface`` / ``local`` / ``mock`` — construct
  that provider (missing credentials raise LLMError at resolution, which
  the chat pipeline turns into template fallback + a log line).
* ``none`` — LLM disabled on purpose.

Secrets arrive only from the environment (``OPENAI_API_KEY``,
``HF_TOKEN`` / ``HUGGINGFACEHUB_API_TOKEN``) or their ``NOVA_``
alternates — never from source.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse
from app.llm.prompting import SYSTEM_PROMPT, build_chat_request
from app.llm.providers import (
    HuggingFaceProvider,
    LocalModelProvider,
    MockProvider,
    OpenAIProvider,
    PROVIDER_DEFAULT_MODELS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "SYSTEM_PROMPT",
    "build_chat_request",
    "get_llm_provider",
    "HuggingFaceProvider",
    "LocalModelProvider",
    "MockProvider",
    "OpenAIProvider",
]


@lru_cache
def get_llm_provider() -> LLMProvider | None:
    """Resolve the provider from settings (cached; clear on config change)."""
    from app.config import get_settings

    settings = get_settings()
    choice = settings.llm_provider

    if choice == "none":
        return None

    if choice == "auto":
        if settings.openai_api_key:
            provider: LLMProvider | None = OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                model=settings.llm_model or PROVIDER_DEFAULT_MODELS["openai"],
                timeout=settings.llm_timeout,
            )
            logger.info("LLM provider: openai (model=%s)", provider.model)
            return provider
        if settings.hf_token:
            provider = HuggingFaceProvider(
                token=settings.hf_token.get_secret_value(),
                model=settings.llm_model or PROVIDER_DEFAULT_MODELS["huggingface"],
                timeout=settings.llm_timeout,
            )
            logger.info("LLM provider: huggingface (model=%s)", provider.model)
            return provider
        logger.info(
            "LLM provider: none (no OPENAI_API_KEY / HF_TOKEN found) — "
            "using deterministic template replies"
        )
        return None

    if choice == "openai":
        return OpenAIProvider(
            api_key=(
                settings.openai_api_key.get_secret_value()
                if settings.openai_api_key
                else ""
            ),
            model=settings.llm_model or PROVIDER_DEFAULT_MODELS["openai"],
            timeout=settings.llm_timeout,
        )
    if choice == "huggingface":
        return HuggingFaceProvider(
            token=(
                settings.hf_token.get_secret_value() if settings.hf_token else ""
            ),
            model=settings.llm_model or PROVIDER_DEFAULT_MODELS["huggingface"],
            timeout=settings.llm_timeout,
        )
    if choice == "local":
        return LocalModelProvider(
            model=settings.llm_model or PROVIDER_DEFAULT_MODELS["local"]
        )
    if choice == "mock":
        return MockProvider()

    raise LLMError(f"unknown llm_provider: {choice!r}")  # validator should prevent

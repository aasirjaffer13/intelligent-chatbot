"""Concrete LLM providers (Phase 8).

* **OpenAIProvider** — hosted chat completions over HTTP (httpx).
* **HuggingFaceProvider** — hosted text-generation Inference API over HTTP.
* **LocalModelProvider** — `transformers` pipeline on this machine;
  the model loads lazily on the first call, never at import time.
* **MockProvider** — deterministic offline double for tests and demos.

All HTTP providers accept an injectable ``transport`` so tests run fully
offline via ``httpx.MockTransport``. API keys are passed in, never logged,
never part of a repr.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import httpx

from app.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
_DEFAULT_LOCAL_MODEL = "distilgpt2"


class OpenAIProvider(LLMProvider):
    """POST {base}/chat/completions with a bearer key."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_OPENAI_MODEL,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("OpenAIProvider requires an API key")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def __repr__(self) -> str:  # never leak the key
        return f"OpenAIProvider(model={self.model!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        started = perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000

        if response.status_code != 200:
            raise LLMError(
                f"OpenAI API returned {response.status_code}: {response.text[:200]}"
            )
        try:
            body: dict[str, Any] = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected OpenAI response shape: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("OpenAI returned an empty completion")
        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


class HuggingFaceProvider(LLMProvider):
    """Text-generation Inference API: ``inputs`` + ``parameters`` in, text out."""

    name = "huggingface"

    def __init__(
        self,
        *,
        token: str,
        model: str = _DEFAULT_HF_MODEL,
        base_url: str = "https://api-inference.huggingface.co",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token:
            raise LLMError("HuggingFaceProvider requires a token")
        self._token = token
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def __repr__(self) -> str:
        return f"HuggingFaceProvider(model={self.model!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        text_in = (
            f"{request.system}\n\n{request.prompt}" if request.system else request.prompt
        )
        payload = {
            "inputs": text_in,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": max(request.temperature, 1e-5),
                "return_full_text": False,
            },
        }
        started = perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/models/{self.model}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"HuggingFace request failed: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000

        if response.status_code != 200:
            raise LLMError(
                f"HuggingFace API returned {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError("HuggingFace returned non-JSON") from exc
        if isinstance(body, dict) and "error" in body:
            raise LLMError(f"HuggingFace error: {body['error']}")

        # Inference API returns [{"generated_text": "..."}] (or a bare dict).
        try:
            generated = (
                body[0]["generated_text"] if isinstance(body, list) else body["generated_text"]
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected HuggingFace response shape: {exc}") from exc
        if not isinstance(generated, str) or not generated.strip():
            raise LLMError("HuggingFace returned an empty completion")
        return LLMResponse(
            text=generated.strip(),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


class LocalModelProvider(LLMProvider):
    """`transformers` text-generation pipeline running on this machine.

    The (potentially large) model download happens on the first
    ``complete()`` call — constructing the provider is free, and a missing
    model or library surfaces as LLMError (-> template fallback), not a
    crash at startup.
    """

    name = "local"

    def __init__(self, *, model: str = _DEFAULT_LOCAL_MODEL) -> None:
        self.model = model
        self._pipeline: Any = None

    def __repr__(self) -> str:
        return f"LocalModelProvider(model={self.model!r}, loaded={self._pipeline is not None})"

    def _load(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as exc:
                raise LLMError("transformers is not installed") from exc
            try:
                self._pipeline = hf_pipeline("text-generation", model=self.model)
            except Exception as exc:  # model download/load failures
                raise LLMError(f"failed to load local model '{self.model}': {exc}") from exc
        return self._pipeline

    def complete(self, request: LLMRequest) -> LLMResponse:
        text_in = (
            f"{request.system}\n\n{request.prompt}" if request.system else request.prompt
        )
        started = perf_counter()
        try:
            outputs = self._load()(
                text_in,
                max_new_tokens=request.max_tokens,
                temperature=max(request.temperature, 1e-5),
                do_sample=request.temperature > 0,
                return_full_text=False,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"local generation failed: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000

        try:
            text = outputs[0]["generated_text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected local pipeline output: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("local model returned an empty completion")
        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


class MockProvider(LLMProvider):
    """Deterministic offline provider.

    Records every request so tests can assert on prompt contents. Used for
    tests, demos, and ``NOVA_LLM_PROVIDER=mock`` — never for real traffic.
    """

    name = "mock"

    def __init__(self, *, reply: str = "This is a mock LLM reply.") -> None:
        self.reply = reply
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text=self.reply, provider=self.name, model=self.name)


PROVIDER_DEFAULT_MODELS = {
    "openai": _DEFAULT_OPENAI_MODEL,
    "huggingface": _DEFAULT_HF_MODEL,
    "local": _DEFAULT_LOCAL_MODEL,
}

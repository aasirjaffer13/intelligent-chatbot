"""Phase 8 tests: LLM provider abstraction, prompting, factory, chat wiring.

Everything runs offline: HTTP providers are exercised through
``httpx.MockTransport``; the factory is driven by monkeypatched
environment variables; chat integration injects ``MockProvider``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.llm import (
    HuggingFaceProvider,
    LLMError,
    LLMRequest,
    LocalModelProvider,
    MockProvider,
    OpenAIProvider,
    SYSTEM_PROMPT,
    build_chat_request,
    get_llm_provider,
)
from app.memory import InMemoryStore
from app.memory.base import MemoryMessage
from app.services.chat_service import ChatService

# --- shared helpers ----------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_llm_caches():
    """Settings + provider factory must reflect this test's environment."""
    from app.config import get_settings

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_provider.cache_clear()


def _msg(role: str, content: str) -> MemoryMessage:
    return MemoryMessage(role=role, content=content, created_at=datetime.now(timezone.utc))


# --- prompting ---------------------------------------------------------------


class TestBuildChatRequest:
    def test_prompt_contains_pipeline_facts(self) -> None:
        request = build_chat_request(
            message="remind me at 3pm",
            intent_label="reminder",
            confidence=0.87,
            entity_texts=["TIME=3pm"],
            messages=[_msg("user", "hello"), _msg("assistant", "Hi!")],
            known_name="Aasir",
        )
        assert request.system == SYSTEM_PROMPT
        assert "Detected intent: reminder (confidence 0.87)" in request.prompt
        assert "TIME=3pm" in request.prompt
        assert "User's name (from memory): Aasir" in request.prompt
        assert "user: hello" in request.prompt
        assert "NOVA: Hi!" in request.prompt
        assert "User message: remind me at 3pm" in request.prompt

    def test_minimal_prompt(self) -> None:
        request = build_chat_request(
            message="hi",
            intent_label="greeting",
            confidence=0.9,
            entity_texts=[],
            messages=[],
            known_name=None,
        )
        assert "Entities: none" in request.prompt
        assert "Recent conversation:" not in request.prompt
        assert "User's name" not in request.prompt

    def test_history_is_capped(self) -> None:
        messages = [_msg("user", f"turn {i}") for i in range(20)]
        request = build_chat_request(
            message="now", intent_label="small_talk", confidence=0.5,
            entity_texts=[], messages=messages, known_name=None,
        )
        assert "turn 19" in request.prompt
        assert "turn 10" not in request.prompt  # only the last 8 lines survive
        assert "turn 0" not in request.prompt

    def test_generation_knobs_are_carried(self) -> None:
        request = build_chat_request(
            message="x", intent_label="greeting", confidence=0.5,
            entity_texts=[], messages=[], known_name=None,
            max_tokens=99, temperature=0.0,
        )
        assert request.max_tokens == 99
        assert request.temperature == 0.0


# --- mock provider -----------------------------------------------------------


class TestMockProvider:
    def test_records_requests_and_replies(self) -> None:
        provider = MockProvider(reply="canned")
        response = provider.complete(LLMRequest(prompt="hello"))
        assert response.text == "canned"
        assert response.provider == "mock"
        assert provider.calls[0].prompt == "hello"


# --- OpenAI ------------------------------------------------------------------


def _openai_json(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestOpenAIProvider:
    def test_success_sends_auth_model_and_messages(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.read()
            return httpx.Response(200, json=_openai_json("  Hello from GPT.  "))

        provider = OpenAIProvider(
            api_key="sk-test-123",
            model="gpt-4o-mini",
            transport=httpx.MockTransport(handler),
        )
        response = provider.complete(
            LLMRequest(prompt="hi", system="be brief", max_tokens=42, temperature=0.0)
        )

        assert seen["path"] == "/v1/chat/completions"
        assert seen["auth"] == "Bearer sk-test-123"
        import json as _json

        body = _json.loads(seen["body"])
        assert body["model"] == "gpt-4o-mini"
        assert body["max_tokens"] == 42
        assert body["temperature"] == 0.0
        assert body["messages"][0] == {"role": "system", "content": "be brief"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}

        assert response.text == "Hello from GPT."
        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"

    def test_http_error_raises_llm_error(self) -> None:
        provider = OpenAIProvider(
            api_key="sk-wrong",
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")),
        )
        with pytest.raises(LLMError, match="401"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_malformed_response_raises(self) -> None:
        provider = OpenAIProvider(
            api_key="sk-test",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []})),
        )
        with pytest.raises(LLMError, match="shape"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_empty_completion_raises(self) -> None:
        provider = OpenAIProvider(
            api_key="sk-test",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_openai_json("   "))
            ),
        )
        with pytest.raises(LLMError, match="empty"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_network_failure_raises_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        provider = OpenAIProvider(api_key="sk-test", transport=httpx.MockTransport(handler))
        with pytest.raises(LLMError, match="request failed"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_missing_key_rejected_at_construction(self) -> None:
        with pytest.raises(LLMError, match="API key"):
            OpenAIProvider(api_key="")

    def test_repr_never_leaks_the_key(self) -> None:
        provider = OpenAIProvider(api_key="sk-super-secret-42")
        assert "sk-super-secret-42" not in repr(provider)
        assert "gpt-4o-mini" in repr(provider)


# --- HuggingFace -------------------------------------------------------------


class TestHuggingFaceProvider:
    def test_success_payload_and_parse(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = request.read()
            return httpx.Response(200, json=[{"generated_text": " generated text "}])

        provider = HuggingFaceProvider(
            token="hf_abc",
            model="org/model",
            transport=httpx.MockTransport(handler),
        )
        response = provider.complete(
            LLMRequest(prompt="hi", system="rules", max_tokens=7, temperature=0.3)
        )

        assert seen["path"] == "/models/org/model"
        assert seen["auth"] == "Bearer hf_abc"
        import json as _json

        body = _json.loads(seen["body"])
        assert body["inputs"].startswith("rules")  # system folded in
        assert body["parameters"]["max_new_tokens"] == 7
        assert body["parameters"]["return_full_text"] is False

        assert response.text == "generated text"
        assert response.provider == "huggingface"

    def test_error_dict_raises(self) -> None:
        provider = HuggingFaceProvider(
            token="hf_abc",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"error": "model is loading"})
            ),
        )
        with pytest.raises(LLMError, match="model is loading"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_non_200_raises(self) -> None:
        provider = HuggingFaceProvider(
            token="hf_abc",
            transport=httpx.MockTransport(lambda r: httpx.Response(503, text="busy")),
        )
        with pytest.raises(LLMError, match="503"):
            provider.complete(LLMRequest(prompt="hi"))

    def test_missing_token_rejected(self) -> None:
        with pytest.raises(LLMError, match="token"):
            HuggingFaceProvider(token="")


# --- local -------------------------------------------------------------------


class TestLocalModelProvider:
    def test_construction_is_lazy(self) -> None:
        provider = LocalModelProvider(model="distilgpt2")
        assert provider._pipeline is None  # nothing downloaded at import/construct
        assert "loaded=False" in repr(provider)


# --- factory -----------------------------------------------------------------


def _scrub_key_env(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                 "NOVA_OPENAI_API_KEY", "NOVA_HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NOVA_LLM_PROVIDER", raising=False)


class TestFactory:
    def test_auto_without_keys_resolves_to_none(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        assert get_llm_provider() is None

    def test_auto_with_openai_key(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_auto_prefers_openai_over_huggingface(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("HF_TOKEN", "hf-from-env")
        assert isinstance(get_llm_provider(), OpenAIProvider)

    def test_auto_with_hf_token_only(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "hf-from-env")
        assert isinstance(get_llm_provider(), HuggingFaceProvider)

    def test_explicit_openai_without_key_raises(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("NOVA_LLM_PROVIDER", "openai")
        with pytest.raises(LLMError, match="API key"):
            get_llm_provider()

    def test_explicit_none_disables_llm(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("NOVA_LLM_PROVIDER", "none")
        assert get_llm_provider() is None

    def test_explicit_mock(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("NOVA_LLM_PROVIDER", "mock")
        assert isinstance(get_llm_provider(), MockProvider)

    def test_explicit_local(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("NOVA_LLM_PROVIDER", "local")
        provider = get_llm_provider()
        assert isinstance(provider, LocalModelProvider)
        assert provider._pipeline is None  # still lazy

    def test_settings_reject_unknown_provider(self) -> None:
        from app.config import Settings

        with pytest.raises(ValidationError, match="llm_provider"):
            Settings(llm_provider="parrot")

    def test_api_key_is_secret_in_settings(self, monkeypatch) -> None:
        _scrub_key_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-99")
        from app.config import Settings

        settings = Settings()
        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-super-secret-99"
        assert "sk-super-secret-99" not in repr(settings)


# --- chat service integration ------------------------------------------------


class TestChatServiceWithLLM:
    def test_llm_phrases_the_reply(self) -> None:
        mock = MockProvider(reply="Phrased by the LLM.")
        service = ChatService(memory=InMemoryStore(), llm=mock)

        from app.schemas import ChatRequest

        response = service.handle(ChatRequest(message="hello there"))

        assert response.response == "Phrased by the LLM."
        assert response.intent == "greeting"
        assert len(mock.calls) == 1
        assert "hello there" in mock.calls[0].prompt
        assert "Detected intent: greeting" in mock.calls[0].prompt
        assert mock.calls[0].system == SYSTEM_PROMPT

    def test_time_intent_never_reaches_the_llm(self) -> None:
        mock = MockProvider(reply="LLM time")
        service = ChatService(memory=InMemoryStore(), llm=mock)

        from app.schemas import ChatRequest

        response = service.handle(ChatRequest(message="what time is it"))

        assert mock.calls == []
        assert "timezone" in response.response  # deterministic clock answer

    def test_name_flow_never_reaches_the_llm(self) -> None:
        mock = MockProvider(reply="LLM name")
        service = ChatService(memory=InMemoryStore(), llm=mock)

        from app.schemas import ChatRequest

        hello = service.handle(ChatRequest(message="My name is Zephyr"))
        assert "Nice to meet you, Zephyr" in hello.response

        recalled = service.handle(
            ChatRequest(message="what is my name?", session_id=hello.session_id)
        )
        assert "Zephyr" in recalled.response
        assert mock.calls == []  # memory answers come from stored history, not a model

    def test_document_questions_go_to_rag_not_the_llm(
        self, tmp_path
    ) -> None:
        from app.rag import RagStore

        mock = MockProvider(reply="LLM answer")
        service = ChatService(
            memory=InMemoryStore(),
            llm=mock,
            rag_store=RagStore(f"sqlite:///{tmp_path / 'rag.db'}"),  # offline
        )

        from app.schemas import ChatRequest

        response = service.handle(
            ChatRequest(message="what does the document say about vacation days")
        )

        assert response.intent == "document_question"
        assert response.response != "LLM answer"
        assert "upload" in response.response.lower()  # RAG refusal: no documents
        assert response.sources == []
        assert mock.calls == []

    def test_llm_error_falls_back_to_templates(self) -> None:
        class FailingProvider:
            name = "failing"

            def complete(self, request: LLMRequest):
                raise LLMError("provider is down")

        failing = ChatService(memory=InMemoryStore(), llm=FailingProvider())
        control = ChatService(memory=InMemoryStore(), llm=None)

        from app.schemas import ChatRequest

        broken = failing.handle(ChatRequest(message="hello there"))
        template = control.handle(ChatRequest(message="hello there"))

        assert broken.response == template.response
        assert broken.response  # chat never breaks, even when the LLM does

    def test_empty_llm_reply_falls_back_to_templates(self) -> None:
        empty = ChatService(memory=InMemoryStore(), llm=MockProvider(reply="   "))
        control = ChatService(memory=InMemoryStore(), llm=None)

        from app.schemas import ChatRequest

        a = empty.handle(ChatRequest(message="hello there"))
        b = control.handle(ChatRequest(message="hello there"))
        assert a.response == b.response


# --- API integration ---------------------------------------------------------


class TestChatEndpointWithLLM:
    def test_endpoint_uses_injected_provider(self, client) -> None:
        from app.schemas import ChatRequest  # noqa: F401  (contract import)
        from app.services import chat_service as chat_module

        mock = MockProvider(reply="Hello from the injected provider.")
        previous = chat_module.chat_service._llm
        previous_resolved = chat_module.chat_service._llm_resolved
        previous_agent = chat_module.chat_service._agent
        previous_agent_resolved = chat_module.chat_service._agent_resolved
        chat_module.chat_service._llm = mock
        chat_module.chat_service._llm_resolved = True
        chat_module.chat_service._agent = None
        chat_module.chat_service._agent_resolved = False  # rebuild around mock
        try:
            resp = client.post("/api/chat", json={"message": "hello there"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["response"] == "Hello from the injected provider."
            assert body["intent"] == "greeting"
            assert body["sources"] == []
        finally:
            chat_module.chat_service._llm = previous
            chat_module.chat_service._llm_resolved = previous_resolved
            chat_module.chat_service._agent = previous_agent
            chat_module.chat_service._agent_resolved = previous_agent_resolved

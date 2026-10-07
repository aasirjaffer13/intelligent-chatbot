from __future__ import annotations

from app.schemas import ChatRequest
from app.services.chat_service import ChatService


def test_handle_populates_processing_info() -> None:
    response = ChatService().handle(ChatRequest(message="Hello, how are you?"))

    assert response.processing.tokens == ["hello"]
    assert response.processing.normalized_text == "hello, how are you?"
    assert response.processing.sentences == ["hello, how are you?"]


def test_handle_keeps_contract_placeholders() -> None:
    response = ChatService().handle(ChatRequest(message="hello"))

    assert response.intent == "unknown"
    assert response.confidence == 0.0
    assert response.entities == []


def test_handle_echoes_session_id() -> None:
    response = ChatService().handle(
        ChatRequest(message="hello there", session_id="session-42")
    )

    assert response.session_id == "session-42"


def test_reply_reports_token_count() -> None:
    response = ChatService().handle(ChatRequest(message="hello, how are you?"))

    assert "1 token" in response.response
    assert "hello" in response.response


def test_reply_truncates_long_token_previews() -> None:
    message = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"
    response = ChatService().handle(ChatRequest(message=message))

    assert "11 tokens" in response.response
    assert "…" in response.response
    assert "kilo" not in response.response  # beyond the 10-token preview


def test_reply_when_no_content_tokens_remain() -> None:
    response = ChatService().handle(ChatRequest(message="!!! ???"))

    assert "no content tokens" in response.response
    assert response.processing.tokens == []

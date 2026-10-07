from __future__ import annotations

from app.schemas import ChatRequest
from app.services.chat_service import ChatService


def test_handle_classifies_and_resplies() -> None:
    response = ChatService().handle(ChatRequest(message="hello, how are you?"))

    assert response.intent == "greeting"
    assert response.confidence > 0.35
    assert response.response  # template-based greeting, not a stub
    assert response.processing.tokens == ["hello"]
    assert response.processing.normalized_text == "hello, how are you?"
    assert response.processing.sentences == ["hello, how are you?"]


def test_handle_entities_empty_until_phase_4() -> None:
    response = ChatService().handle(ChatRequest(message="hello"))
    assert response.entities == []


def test_handle_echoes_session_id() -> None:
    response = ChatService().handle(
        ChatRequest(message="hello there", session_id="session-42")
    )
    assert response.session_id == "session-42"


def test_time_intent_returns_real_clock() -> None:
    response = ChatService().handle(ChatRequest(message="what time is it"))

    assert response.intent == "time"
    assert "It's" in response.response
    assert ":" in response.response  # HH:MM


def test_unknown_intent_mentions_low_confidence() -> None:
    response = ChatService().handle(
        ChatRequest(message="qwertyuiop zxcvbn mnbvcxz 98765")
    )

    assert response.intent == "unknown"
    assert response.confidence < 0.6
    assert "intent confidence" in response.response


def test_punctuation_only_input_survives() -> None:
    response = ChatService().handle(ChatRequest(message="!!! ???"))

    assert response.processing.tokens == []
    assert response.response  # still produces a coherent reply
    assert response.intent == "unknown"


def test_reply_generation_is_deterministic() -> None:
    svc = ChatService()
    first = svc.handle(ChatRequest(message="thanks a lot")).response
    second = svc.handle(ChatRequest(message="thanks a lot")).response
    assert first == second

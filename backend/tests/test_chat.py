from __future__ import annotations

from fastapi.testclient import TestClient


def test_chat_returns_full_contract(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "hello, how are you?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"]
    assert body["intent"] == "unknown"
    assert body["confidence"] == 0.0
    assert body["entities"] == []
    # Phase 2: the preprocessing pipeline fills `processing`
    assert body["processing"]["tokens"] == ["hello"]
    assert body["processing"]["normalized_text"] == "hello, how are you?"
    assert body["processing"]["sentences"] == ["hello, how are you?"]
    assert "hello" in body["response"]


def test_chat_rejects_missing_message(client: TestClient) -> None:
    response = client.post("/api/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_blank_message(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "   "})

    assert response.status_code == 422

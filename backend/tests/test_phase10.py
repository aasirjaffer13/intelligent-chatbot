"""Phase 10 backend tests: SSE streaming, conversation browsing, status."""

from __future__ import annotations

import json


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert name is not None and data is not None, f"malformed SSE block: {block!r}"
        events.append((name, data))
    return events


class TestStreamEndpoint:
    def test_stream_contract(self, client) -> None:
        with client.stream(
            "POST", "/api/chat/stream", json={"message": "hello there"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

        events = parse_sse(body)
        names = [name for name, _ in events]
        assert names[0] == "delta"
        assert "meta" in names
        assert names[-1] == "end"
        assert names.count("meta") == 1

        deltas = "".join(data["text"] for name, data in events if name == "delta")
        meta = next(data for name, data in events if name == "meta")
        # the streamed chunks reassemble into EXACTLY the contract reply
        assert meta["response"] == deltas
        assert meta["intent"] == "greeting"
        assert meta["confidence"] > 0
        assert meta["session_id"]

    def test_stream_carries_sources_for_document_questions(self, client) -> None:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "what does the document say about vacation days"},
        ) as response:
            body = "".join(response.iter_text())
        meta = next(data for name, data in parse_sse(body) if name == "meta")
        assert meta["intent"] == "document_question"
        assert isinstance(meta["sources"], list)
        assert meta["processing"]["tokens"]

    def test_stream_validation_error_is_json(self, client) -> None:
        response = client.post("/api/chat/stream", json={"message": ""})
        assert response.status_code == 422


class TestConversationsApi:
    def test_empty_list(self, client) -> None:
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert response.json() == {"conversations": []}

    def test_list_and_history_roundtrip(self, client) -> None:
        first = client.post("/api/chat", json={"message": "hello there"}).json()
        session_a = first["session_id"]
        client.post(
            "/api/chat", json={"message": "My name is Zephyr", "session_id": session_a}
        )
        second = client.post("/api/chat", json={"message": "tell me a joke"}).json()
        session_b = second["session_id"]

        listing = client.get("/api/conversations").json()["conversations"]
        assert len(listing) == 2
        by_id = {item["session_id"]: item for item in listing}
        assert set(by_id) == {session_a, session_b}
        assert by_id[session_a]["message_count"] == 4
        assert by_id[session_b]["message_count"] == 2
        assert by_id[session_b]["preview"]  # last message, non-empty
        # most recently updated first
        assert listing[0]["session_id"] == session_b

        history = client.get(f"/api/conversations/{session_a}/messages")
        assert history.status_code == 200
        payload = history.json()
        assert payload["session_id"] == session_a
        assert len(payload["messages"]) == 4
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert payload["messages"][1]["content"] == first["response"]

    def test_unknown_conversation_404_envelope(self, client) -> None:
        response = client.get("/api/conversations/deadbeef/messages")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert "deadbeef" in error["detail"]


class TestStatusEndpoint:
    def test_status_shape(self, client) -> None:
        response = client.get("/api/status")
        assert response.status_code == 200
        body = response.json()

        assert set(body) == {"llm", "agent", "rag", "memory", "intent"}
        assert body["llm"]["provider"] in {
            "openai",
            "huggingface",
            "local",
            "mock",
            "none",
        }
        assert body["agent"]["max_steps"] >= 1
        assert body["rag"]["mode"] in {"pgvector", "numpy", "unavailable"}
        assert body["rag"]["documents"] >= 0
        assert body["memory"]["backend"] in {"postgresql", "sqlite", "in-memory"}
        assert body["intent"]["backend"] in {"auto", "sklearn", "embedding", "keyword"}

    def test_status_does_not_leak_secrets(self, client) -> None:
        body = client.get("/api/status").text
        assert "sk-" not in body
        assert "hf_" not in body

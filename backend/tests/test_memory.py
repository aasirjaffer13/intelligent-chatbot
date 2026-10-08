from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.memory import (
    InMemoryStore,
    build_context,
    extract_name,
    get_memory_store,
    is_name_question,
)
from app.memory.db import SqlAlchemyMemoryStore
from app.schemas import ChatRequest
from app.services.chat_service import ChatService


class TestInMemoryStore:
    def test_get_or_create_generates_id(self) -> None:
        store = InMemoryStore()
        created = store.get_or_create_conversation(None)
        assert created
        assert store.get_or_create_conversation(created) == created

    def test_add_and_recent_roundtrip(self) -> None:
        store = InMemoryStore()
        store.get_or_create_conversation("s1")
        store.add_message("s1", "user", "hello", "greeting", 0.9)
        store.add_message("s1", "assistant", "hi there", "greeting", 0.9)
        recent = store.recent_messages("s1")
        assert [m.role for m in recent] == ["user", "assistant"]
        assert recent[0].content == "hello"
        assert recent[0].intent == "greeting"
        assert recent[0].confidence == 0.9

    def test_recent_respects_limit(self) -> None:
        store = InMemoryStore()
        for i in range(10):
            store.add_message("s1", "user", f"msg{i}")
        recent = store.recent_messages("s1", limit=3)
        assert [m.content for m in recent] == ["msg7", "msg8", "msg9"]

    def test_zero_limit_returns_empty_not_everything(self) -> None:
        store = InMemoryStore()
        store.add_message("s1", "user", "hello")
        assert store.recent_messages("s1", limit=0) == []

    def test_unknown_session_returns_empty(self) -> None:
        assert InMemoryStore().recent_messages("nope") == []


class TestSqlAlchemyStore:
    """Same contract as InMemoryStore, on a real SQL engine (SQLite here;
    the store only uses portable constructs, so PostgreSQL behaves the same)."""

    @pytest.fixture()
    def store(self, tmp_path) -> SqlAlchemyMemoryStore:
        return SqlAlchemyMemoryStore(f"sqlite:///{tmp_path / 'mem.db'}")

    def test_contract_parity(self, store: SqlAlchemyMemoryStore) -> None:
        session_id = store.get_or_create_conversation(None)
        assert store.get_or_create_conversation(session_id) == session_id
        store.add_message(session_id, "user", "hello", "greeting", 0.8)
        store.add_message(session_id, "assistant", "hi", "greeting", 0.8)
        recent = store.recent_messages(session_id)
        assert [m.role for m in recent] == ["user", "assistant"]

    def test_persists_across_instances(self, tmp_path) -> None:
        url = f"sqlite:///{tmp_path / 'persist.db'}"
        first = SqlAlchemyMemoryStore(url)
        session_id = first.get_or_create_conversation("persist-s1")
        first.add_message(session_id, "user", "remember me")

        second = SqlAlchemyMemoryStore(url)  # same file, new store
        recent = second.recent_messages("persist-s1")
        assert [m.content for m in recent] == ["remember me"]

    def test_sessions_are_isolated(self, store: SqlAlchemyMemoryStore) -> None:
        store.add_message("alpha", "user", "secretA")
        store.add_message("beta", "user", "secretB")
        assert [m.content for m in store.recent_messages("alpha")] == ["secretA"]
        assert [m.content for m in store.recent_messages("beta")] == ["secretB"]


class TestContextBuilding:
    def test_extract_name_statements(self) -> None:
        assert extract_name("My name is Zephyr") == "Zephyr"
        assert extract_name("i am quinn") == "Quinn"
        assert extract_name("I'm Ada") == "Ada"
        assert extract_name("call me Sam") == "Sam"

    def test_extract_name_rejects_predicates(self) -> None:
        assert extract_name("I'm fine") is None
        assert extract_name("I am good") is None
        assert extract_name("I am here") is None
        assert extract_name("hello there") is None

    def test_name_questions(self) -> None:
        assert is_name_question("what is my name?")
        assert is_name_question("what's my name")
        assert is_name_question("do you remember my name")
        assert not is_name_question("what is the weather")

    def test_build_context_takes_newest_name(self) -> None:
        store = InMemoryStore()
        store.add_message("s", "user", "My name is OldName")
        store.add_message("s", "assistant", "noted")
        store.add_message("s", "user", "actually, My name is NewName")
        context = build_context(store.recent_messages("s"), "hello")
        assert context.known_name == "NewName"
        assert context.turn_count == 3  # 2 stored user turns + current

    def test_build_context_uses_current_message(self) -> None:
        context = build_context([], "My name is Solo")
        assert context.known_name == "Solo"
        assert context.turn_count == 1


class TestChatServiceMemory:
    def test_name_recalled_from_real_history(self) -> None:
        """Roadmap's acceptance test: statement turn 1, question turn 2."""
        svc = ChatService(memory=InMemoryStore())
        first = svc.handle(ChatRequest(message="My name is Zephyr", session_id="s-name"))
        assert "Zephyr" in first.response

        second = svc.handle(ChatRequest(message="what is my name?", session_id="s-name"))
        assert "Zephyr" in second.response
        assert "from" in second.response or "earlier" in second.response

    def test_name_not_leaked_across_sessions(self) -> None:
        svc = ChatService(memory=InMemoryStore())
        svc.handle(ChatRequest(message="My name is Zephyr", session_id="s-a"))
        other = svc.handle(ChatRequest(message="what is my name?", session_id="s-b"))
        assert "Zephyr" not in other.response

    def test_session_id_created_when_absent(self) -> None:
        svc = ChatService(memory=InMemoryStore())
        first = svc.handle(ChatRequest(message="hello"))
        assert first.session_id  # server-created
        # following up with the returned id continues the same conversation
        svc.handle(ChatRequest(message="My name is Echo", session_id=first.session_id))
        second = svc.handle(
            ChatRequest(message="what is my name?", session_id=first.session_id)
        )
        assert "Echo" in second.response

    def test_stateless_reply_unchanged(self) -> None:
        svc = ChatService(memory=InMemoryStore())
        response = svc.handle(ChatRequest(message="hello"))
        assert response.intent == "greeting"
        assert response.response

    def test_zero_window_disables_history(self) -> None:
        svc = ChatService(memory=InMemoryStore(), window=0)
        svc.handle(ChatRequest(message="My name is Ghost", session_id="s0"))
        second = svc.handle(ChatRequest(message="what is my name?", session_id="s0"))
        assert "Ghost" not in second.response


class TestMemoryFactory:
    def test_empty_url_gives_in_memory_store(self) -> None:
        assert isinstance(get_memory_store(""), InMemoryStore)

    def test_sql_url_gives_sql_store(self, tmp_path) -> None:
        store = get_memory_store(f"sqlite:///{tmp_path / 'factory.db'}")
        assert isinstance(store, SqlAlchemyMemoryStore)

    def test_bad_url_degrades_to_in_memory(self) -> None:
        store = get_memory_store("postgresql+psycopg://nobody:wrong@127.0.0.1:1/nope")
        assert isinstance(store, InMemoryStore)


class TestChatApiMemory:
    def test_api_session_flow(self, client: TestClient) -> None:
        first = client.post("/api/chat", json={"message": "My name is River"})
        assert first.status_code == 200
        session_id = first.json()["session_id"]
        assert session_id

        second = client.post(
            "/api/chat", json={"message": "what is my name?", "session_id": session_id}
        )
        assert "River" in second.json()["response"]

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.memory import InMemoryStore
from app.rag import (
    DocumentService,
    EmptyDocumentError,
    RagStore,
    UnsupportedFileType,
    chunk_text,
    compose_grounded_answer,
    extract_text,
)
from app.rag.answerer import NO_DOCUMENTS_REPLY, NOT_FOUND_REPLY
from app.services.embedding_service import get_embedding_service
from app.services.chat_service import ChatService

def _build_pdf(text: str) -> bytes:
    """Minimal single-page PDF with *computed* xref offsets (valid enough
    for pypdf to extract the text — the point of the fixture)."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


MINI_PDF = _build_pdf(
    "The kubernetes scheduler places pods across nodes using affinity rules."
)


class TestChunker:
    def test_offsets_are_exact(self) -> None:
        text = " ".join(f"word{i}" for i in range(250))
        for chunk in chunk_text(text, chunk_words=80, overlap_words=15):
            assert text[chunk.start : chunk.end] == chunk.text

    def test_overlap_produces_shared_words(self) -> None:
        text = " ".join(f"w{i}" for i in range(200))
        chunks = chunk_text(text, chunk_words=50, overlap_words=10)
        assert len(chunks) > 1
        first_tail = set(chunks[0].text.split()[-10:])
        second_head = set(chunks[1].text.split()[:10])
        assert first_tail & second_head  # overlap is real

    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("just a few words here")
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_empty_and_whitespace(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_indexes_sequential(self) -> None:
        text = " ".join(f"w{i}" for i in range(500))
        chunks = chunk_text(text, chunk_words=50, overlap_words=5)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_overlap_ge_chunk_words_is_clamped(self) -> None:
        chunks = chunk_text("one two three four five", chunk_words=3, overlap_words=99)
        assert chunks  # clamped to 2, step 1 — still terminates


class TestExtractor:
    def test_plain_text(self) -> None:
        assert extract_text("notes.txt", "hello world".encode()) == "hello world"

    def test_markdown(self) -> None:
        assert extract_text("readme.md", "# Title\ntext".encode()) == "# Title\ntext"

    def test_pdf_extracts_page_text(self) -> None:
        text = extract_text("doc.pdf", MINI_PDF)
        assert "kubernetes scheduler" in text

    def test_unsupported_type(self) -> None:
        with pytest.raises(UnsupportedFileType):
            extract_text("image.png", b"\x89PNG binary junk")

    def test_empty_document(self) -> None:
        with pytest.raises(EmptyDocumentError):
            extract_text("blank.txt", b"   \n ")


class TestRagStore:
    @pytest.fixture()
    def store(self, tmp_path) -> RagStore:
        return RagStore(f"sqlite:///{tmp_path / 'store.db'}")

    @pytest.fixture(autouse=True)
    def _embedder(self):
        self.embedder = get_embedding_service()

    def test_numpy_mode_on_sqlite(self, store: RagStore) -> None:
        assert store.vector_enabled is False

    def test_add_list_delete_roundtrip(self, store: RagStore) -> None:
        info = store.add_document(
            filename="a.txt",
            content_type="text/plain",
            size_bytes=10,
            stored_path=None,
            chunks=["alpha beta", "gamma delta"],
            embeddings=self.embedder.encode(["alpha beta", "gamma delta"]),
        )
        assert info.chunk_count == 2
        assert [d.filename for d in store.list_documents()] == ["a.txt"]
        assert store.delete_document(info.id) is None  # no stored file
        assert store.list_documents() == []
        assert store.delete_document("missing") is None

    def test_search_ranks_relevant_chunk_first(self, store: RagStore) -> None:
        contents = [
            "the cat sat on the mat",
            "quantum physics equations describe particle behavior",
            "baking bread requires flour water yeast and patience",
        ]
        store.add_document(
            filename="mixed.txt",
            content_type="text/plain",
            size_bytes=100,
            stored_path=None,
            chunks=contents,
            embeddings=self.embedder.encode(contents),
        )
        hits = store.search(self.embedder.embed("baking bread with yeast"), top_k=2)
        assert hits[0].content == contents[2]
        assert hits[0].score > hits[1].score
        assert -1.0 <= hits[0].score <= 1.0

    def test_search_empty_store(self, store: RagStore) -> None:
        assert store.search(self.embedder.embed("anything"), top_k=3) == []

    def test_search_document_filter(self, store: RagStore) -> None:
        one = store.add_document(
            filename="one.txt", content_type="text/plain", size_bytes=1,
            stored_path=None, chunks=["apples and oranges"],
            embeddings=self.embedder.encode(["apples and oranges"]),
        )
        store.add_document(
            filename="two.txt", content_type="text/plain", size_bytes=1,
            stored_path=None, chunks=["cars and trucks"],
            embeddings=self.embedder.encode(["cars and trucks"]),
        )
        hits = store.search(self.embedder.embed("apples fruit"), top_k=5, document_ids=[one.id])
        assert hits and all(h.document_id == one.id for h in hits)


class TestGroundedAnswer:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.rag.store import ScoredChunk

        self.ScoredChunk = ScoredChunk
        self.embedder = get_embedding_service()

    def _hit(self, content: str, score: float) -> "object":
        return self.ScoredChunk(
            document_id="doc1", filename="guide.txt",
            chunk_index=0, content=content, score=score,
        )

    def test_no_documents_refusal(self) -> None:
        answer = compose_grounded_answer("q", [], min_score=0.35, has_documents=False)
        assert answer.reply == NO_DOCUMENTS_REPLY
        assert not answer.grounded and not answer.sources

    def test_weak_scores_refuse(self) -> None:
        answer = compose_grounded_answer(
            "quantum", [self._hit("some text", 0.1)],
            min_score=0.35, has_documents=True,
        )
        assert answer.reply == NOT_FOUND_REPLY
        assert not answer.grounded

    def test_grounded_reply_quotes_document(self) -> None:
        content = "The kubernetes scheduler places pods across nodes. Affinity rules guide placement."
        answer = compose_grounded_answer(
            "how does kubernetes place pods", [self._hit(content, 0.72)],
            min_score=0.35, has_documents=True,
        )
        assert answer.grounded
        assert "guide.txt" in answer.reply
        assert "kubernetes" in answer.reply.lower()
        assert len(answer.sources) == 1
        assert answer.sources[0].score == 0.72
        assert answer.sources[0].quote  # verbatim excerpt present

    def test_sources_capped_at_three(self) -> None:
        hits = [self._hit("text", 0.9 - i * 0.05) for i in range(5)]
        answer = compose_grounded_answer(
            "text", hits, min_score=0.35, has_documents=True
        )
        assert len(answer.sources) == 3


class TestDocumentService:
    def test_ingest_stores_file_and_chunks(self, tmp_path) -> None:
        store = RagStore(f"sqlite:///{tmp_path / 'svc.db'}")
        service = DocumentService(store, get_embedding_service(), tmp_path / "files")
        info = service.ingest(
            "notes.txt",
            ("kubernetes networking guide text " * 20).encode(),
            "text/plain",
        )
        assert info.chunk_count >= 1
        assert any(p.name.endswith("notes.txt") for p in (tmp_path / "files").iterdir())

        service.delete(info.id)
        assert service.list() == []
        assert not any(p.name.endswith("notes.txt") for p in (tmp_path / "files").iterdir())


class TestChatRagIntegration:
    @pytest.fixture()
    def svc(self, tmp_path) -> ChatService:
        store = RagStore(f"sqlite:///{tmp_path / 'chat.db'}")
        service = DocumentService(store, get_embedding_service(), tmp_path / "up")
        service.ingest(
            "handbook.txt",
            (
                "Vacation policy: employees accrue 20 paid vacation days per year. "
                "Unused days roll over up to 5 days into the next calendar year. "
                "Requests longer than one week need manager approval."
            ).encode(),
            "text/plain",
        )
        return ChatService(memory=InMemoryStore(), rag_store=store)

    def test_document_question_answered_with_citations(self, svc: ChatService) -> None:
        from app.schemas import ChatRequest

        response = svc.handle(
            ChatRequest(message="what does the document say about vacation days")
        )
        assert response.intent == "document_question"
        assert response.sources, "expected RAG citations"
        assert "handbook.txt" in response.response
        assert "20" in response.response or "vacation" in response.response.lower()

    def test_unanswerable_question_refuses(self, svc: ChatService) -> None:
        from app.schemas import ChatRequest

        response = svc.handle(
            ChatRequest(message="what does the document say about pension investment")
        )
        assert response.intent == "document_question"
        assert response.sources == []  # nothing grounded -> no citations
        assert "couldn't find" in response.response.lower() or "won't guess" in response.response

    def test_non_document_intent_has_no_sources(self, svc: ChatService) -> None:
        from app.schemas import ChatRequest

        response = svc.handle(ChatRequest(message="hello"))
        assert response.sources == []


class TestDocumentApi:
    def _upload(self, client: TestClient, name: str, content: bytes, mime: str):
        return client.post(
            "/api/documents/upload",
            files={"file": (name, content, mime)},
        )

    def test_upload_list_delete_flow(self, client: TestClient) -> None:
        resp = self._upload(
            client, "guide.txt", "docker networking basics for containers".encode(), "text/plain"
        )
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        assert doc["filename"] == "guide.txt"
        assert doc["chunk_count"] >= 1

        listing = client.get("/api/documents").json()
        assert any(d["id"] == doc["id"] for d in listing)

        deleted = client.delete(f"/api/documents/{doc['id']}")
        assert deleted.status_code == 204
        assert all(d["id"] != doc["id"] for d in client.get("/api/documents").json())
        assert client.delete(f"/api/documents/{doc['id']}").status_code == 404

    def test_unsupported_upload_415(self, client: TestClient) -> None:
        resp = self._upload(client, "photo.png", b"\x89PNG\x00\x00", "image/png")
        assert resp.status_code == 415

    def test_empty_upload_400(self, client: TestClient) -> None:
        resp = self._upload(client, "blank.txt", b"   ", "text/plain")
        assert resp.status_code == 400

    def test_chat_end_to_end_rag(self, client: TestClient) -> None:
        upload = self._upload(
            client,
            "security.txt",
            (
                "Password policy: all passwords must be at least 12 characters. "
                "Passwords expire every 90 days and must not be reused."
            ).encode(),
            "text/plain",
        )
        assert upload.status_code == 201

        chat = client.post(
            "/api/chat", json={"message": "what does the document say about password rules"}
        )
        assert chat.status_code == 200
        body = chat.json()
        assert body["intent"] == "document_question"
        assert body["sources"], body
        assert body["sources"][0]["filename"] == "security.txt"
        assert "12" in body["response"] or "password" in body["response"].lower()

    def test_document_question_without_uploads_refuses(self, client: TestClient) -> None:
        chat = client.post(
            "/api/chat", json={"message": "what does the document say about refunds"}
        )
        body = chat.json()
        assert body["intent"] == "document_question"
        assert body["sources"] == []
        assert "upload" in body["response"].lower()

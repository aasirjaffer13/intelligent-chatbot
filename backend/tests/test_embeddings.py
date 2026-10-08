from __future__ import annotations

import numpy as np
import pytest

import app.nlp.intent as intent_module
from app.nlp.intent import (
    EmbeddingIntentClassifier,
    classify_intent,
)
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.similarity_service import (
    SimilarityService,
    cosine,
    cosine_similarity_matrix,
)


@pytest.fixture(scope="module")
def embedder() -> EmbeddingService:
    return get_embedding_service()


@pytest.fixture(scope="module")
def embedding_classifier() -> EmbeddingIntentClassifier:
    """Process-cached embedding classifier (embeds all 489 patterns once)."""
    classifier = intent_module.get_embedding_intent_classifier()
    assert classifier is not None, "sentence-transformers model must be available"
    return classifier


class TestEmbeddingService:
    def test_encode_shape(self, embedder: EmbeddingService) -> None:
        vectors = embedder.encode(["hello world", "second sentence"])
        assert vectors.shape == (2, embedder.dim)

    def test_encode_rows_are_unit_norm(self, embedder: EmbeddingService) -> None:
        vectors = embedder.encode(["semantic vectors are normalized", "another one"])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_empty_batch(self, embedder: EmbeddingService) -> None:
        assert embedder.encode([]).shape == (0, embedder.dim)

    def test_embed_empty_text_is_zero_vector(self, embedder: EmbeddingService) -> None:
        assert np.all(embedder.embed("   ") == 0.0)

    def test_deterministic(self, embedder: EmbeddingService) -> None:
        first = embedder.embed("the same text twice")
        second = embedder.embed("the same text twice")
        assert np.allclose(first, second, atol=1e-6)

    def test_paraphrase_beats_unrelated(self, embedder: EmbeddingService) -> None:
        paraphrase = embedder.embed("I can't remember my login password")
        related = embedder.embed("I forgot my password")
        unrelated = embedder.embed("the stock market rallied today")
        assert cosine(paraphrase, related) > cosine(paraphrase, unrelated) + 0.3


class TestSimilarityMath:
    def test_cosine_identical_is_one(self) -> None:
        v = np.array([0.3, 0.4, 0.5])
        assert cosine(v, v) == pytest.approx(1.0)

    def test_cosine_zero_vector(self) -> None:
        assert cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_matrix_scores_match_pairwise(self, embedder: EmbeddingService) -> None:
        query = embedder.embed("what is the weather like")
        matrix = embedder.encode(["will it rain tomorrow", "tell me a joke"])
        scores = cosine_similarity_matrix(query, matrix)
        for i, row in enumerate(matrix):
            assert scores[i] == pytest.approx(cosine(query, row), abs=1e-5)

    def test_empty_matrix(self) -> None:
        assert cosine_similarity_matrix(np.ones(4), np.zeros((0, 4))).size == 0


class TestSimilarityService:
    def test_ranks_paraphrase_first(self, embedder: EmbeddingService) -> None:
        service = SimilarityService(embedder)
        matches = service.rank(
            "I forgot my password",
            [
                "the stock market rose today",
                "i cannot remember my login password",
                "what is the weather like",
                "reset my login credentials",
            ],
        )
        assert matches[0].text == "i cannot remember my login password"
        assert matches[0].score > 0.5
        assert matches[-1].score < 0.3
        assert all(-1.0 <= m.score <= 1.0 for m in matches)

    def test_empty_candidates(self, embedder: EmbeddingService) -> None:
        assert SimilarityService(embedder).rank("hello", []) == []

    def test_top_k_limits_results(self, embedder: EmbeddingService) -> None:
        service = SimilarityService(embedder)
        matches = service.rank("hello", ["hi", "bye", "thanks"], top_k=2)
        assert len(matches) == 2

    def test_scores_sorted_descending(self, embedder: EmbeddingService) -> None:
        matches = SimilarityService(embedder).rank(
            "what time is it", ["noon", "goodbye", "current clock time"]
        )
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)


class TestEmbeddingIntentClassifier:
    def test_known_intents(self, embedding_classifier) -> None:
        for message, expected in [
            ("hey, what's up?", "greeting"),
            ("what time is it", "time"),
            ("who are you", "identity"),
            ("thanks a lot", "thanks"),
            ("what can you do", "capabilities"),
            ("tell me a joke", "small_talk"),
        ]:
            result = embedding_classifier.predict(message)
            assert result.label == expected, message
            assert result.method == "embedding"

    def test_paraphrase_tf_idf_cannot_see(self, embedding_classifier) -> None:
        # zero keyword overlap with "forgot password" patterns
        result = embedding_classifier.predict("i cannot access my account i forgot the password")
        assert result.label == "password_help"
        assert result.confidence > 0.5

    def test_gibberish_is_unknown(self, embedding_classifier) -> None:
        result = embedding_classifier.predict("zxcvbn mnbvcxz qwe rty 9876")
        assert result.label == "unknown"

    def test_confidence_bounded(self, embedding_classifier) -> None:
        result = embedding_classifier.predict("random text at all")
        assert 0.0 <= result.confidence <= 1.0


class TestBackendSelection:
    def test_default_backend_is_not_embedding(self) -> None:
        # auto chain stays sklearn-first; embeddings are opt-in
        result = classify_intent("hey, what's up?")
        assert result.method in {"sklearn", "tfidf"}

    def test_embedding_backend(self, embedding_classifier) -> None:
        result = classify_intent("what time is it", backend="embedding")
        assert result.method == "embedding"
        assert result.label == "time"

    def test_keyword_backend(self) -> None:
        result = classify_intent("please reset my password", backend="keyword")
        assert result.method == "keyword_fallback"
        assert result.label == "password_help"

    def test_unknown_backend_degrades_to_auto(self) -> None:
        result = classify_intent("hey, what's up?", backend="does-not-exist")
        assert result.method in {"sklearn", "tfidf", "keyword_fallback"}

    def test_embedding_backend_unavailable_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(intent_module, "get_embedding_intent_classifier", lambda: None)
        result = classify_intent("hello", backend="embedding")
        assert result.method in {"sklearn", "tfidf"}
        assert result.label == "greeting"

    @pytest.mark.parametrize("backend", ["auto", "embedding", "keyword", "sklearn"])
    def test_empty_input_short_circuits(self, backend: str) -> None:
        result = classify_intent("   ", backend=backend)
        assert result.label == "unknown"
        assert result.confidence == 0.0
        assert result.method == "empty"

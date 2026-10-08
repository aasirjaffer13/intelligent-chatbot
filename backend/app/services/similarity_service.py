"""Semantic similarity over embeddings (Phase 5; reused by RAG in Phase 7).

Pure vector math + a thin service:

* vectors are L2-normalized by ``EmbeddingService``, so cosine similarity
  reduces to the dot product;
* ``SimilarityService`` takes an injected embedder — tests pass fakes,
  production passes the sentence-transformers singleton;
* ``rank`` returns candidates ordered best-first with scores in [-1, 1].

TF-IDF similarity (Phase 3) measured *shared vocabulary*; embedding
similarity measures *shared meaning* — that is the entire point of Phase 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors; 0.0 if either is a zero vector."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of one query against every row of a matrix (unit rows -> dot)."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # zero rows stay zero-score instead of NaN
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return (matrix @ query) / (norms.ravel() * query_norm)


@dataclass(frozen=True)
class SimilarityMatch:
    """One ranked candidate."""

    text: str
    score: float
    index: int


class SimilarityService:
    """Rank string candidates by semantic similarity to a query."""

    def __init__(self, embedder=None) -> None:
        # Injected for tests; production uses the shared singleton lazily.
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from app.services.embedding_service import get_embedding_service

            self._embedder = get_embedding_service()
        return self._embedder

    def rank(self, query: str, candidates: list[str], top_k: int | None = None) -> list[SimilarityMatch]:
        """Best-first ordering of candidates for the query."""
        if not candidates:
            return []

        query_vec = self.embedder.embed(query)
        matrix = self.embedder.encode(list(candidates))
        scores = cosine_similarity_matrix(query_vec, matrix)

        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [
            SimilarityMatch(text=candidates[i], score=float(scores[i]), index=int(i))
            for i in order
        ]

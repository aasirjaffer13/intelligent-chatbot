"""Sentence embeddings (Phase 5).

Wraps a sentence-transformers model behind a tiny service:

* ``encode`` returns **L2-normalized** vectors, so cosine similarity is a
  plain dot product everywhere downstream (similarity_service, RAG Phase 7).
* The model loads lazily on first use (once per process) — startup stays
  fast and a fresh clone without the model cache works; the download happens
  on first encode when network is available.
* If the model cannot load, callers catch the exception and degrade (the
  intent layer falls back to its TF-IDF chain, RAG falls back to keyword
  search). The API never fails *because* of an embedding model.

Default model: all-MiniLM-L6-v2 (~80 MB, 384 dims, trained on >1B pairs).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Encode text into fixed-size semantic vectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # renamed in newer sentence-transformers; support both
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("embedding model loaded: %s (%d dims)", model_name, self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Vectorize a batch. Returns shape (len(texts), dim), unit-norm rows."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine == dot product downstream
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        """Vectorize a single string. Empty text -> zero vector (not the model)."""
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Load the model once per process. Raises if the model is unavailable —
    callers are expected to handle the failure (see module docstring).
    """
    return EmbeddingService()

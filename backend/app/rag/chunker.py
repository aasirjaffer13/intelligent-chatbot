"""Text chunking for RAG (Phase 7).

Why chunks? Embedding models have a token limit (~256 words for MiniLM in
practice) and retrieval granularity needs to be *smaller than a document* —
you want the paragraph that answers the question, not the whole PDF. Chunks
become the unit of embedding, retrieval, and citation.

Strategy: **sliding word window with overlap**. Overlap exists because an
answer can straddle a boundary; without it, the sentence that answers the
question may be cut in half and neither half retrieves well.

Alternatives (documented in docs/07_rag.md): sentence windows, recursive
character splitting, semantic/similarity chunking, structure-aware splitting
(markdown headers, PDF sections).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit with char offsets into the ORIGINAL text."""

    index: int
    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    *,
    chunk_words: int = 80,
    overlap_words: int = 15,
) -> list[Chunk]:
    """Split ``text`` into overlapping word windows.

    Offsets index the input string exactly: ``text[c.start:c.end] == c.text``.
    """
    if chunk_words < 1:
        raise ValueError("chunk_words must be >= 1")
    tokens = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not tokens:
        return []

    overlap = min(overlap_words, chunk_words - 1)
    step = chunk_words - overlap

    chunks: list[Chunk] = []
    start_tok = 0
    while start_tok < len(tokens):
        end_tok = min(start_tok + chunk_words, len(tokens))
        char_start = tokens[start_tok][0]
        char_end = tokens[end_tok - 1][1]
        chunk = Chunk(
            index=len(chunks),
            text=text[char_start:char_end],
            start=char_start,
            end=char_end,
        )
        # A final window identical to the previous one adds nothing.
        if not chunks or chunk.text != chunks[-1].text:
            chunks.append(chunk)
        if end_tok == len(tokens):
            break
        start_tok += step
    return chunks

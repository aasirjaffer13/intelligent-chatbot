"""Grounded answer composition (Phase 7).

The heart of "no hallucinated document facts": the answer is **assembled
from retrieved chunks**, never from the model's imagination. Two refusal
paths, both explicit:

* no documents uploaded     -> tell the user to upload one
* documents but weak scores -> "not found in the uploaded documents"

Retrieval quality gate: the best chunk must clear ``min_score`` (cosine).
Below that, we prefer honest ignorance over a confidently wrong answer —
the same philosophy as intent's ``unknown`` fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.store import ScoredChunk
from app.schemas import Source

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

NO_DOCUMENTS_REPLY = (
    "I don't have any documents to search yet — upload a PDF, .txt or .md "
    "file and ask me about it."
)
NOT_FOUND_REPLY = (
    "I couldn't find that in the uploaded documents, so I won't guess. "
    "Try rephrasing, or upload a document that covers it."
)


@dataclass
class GroundedAnswer:
    reply: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = False  # True only when reply rests on retrieved context


def compose_grounded_answer(
    query: str,
    hits: list[ScoredChunk],
    *,
    min_score: float,
    has_documents: bool,
) -> GroundedAnswer:
    """Turn retrieval hits into an answer + citations, or an honest refusal."""
    if not has_documents:
        return GroundedAnswer(reply=NO_DOCUMENTS_REPLY, grounded=False)

    eligible = [hit for hit in hits if hit.score >= min_score]
    if not eligible:
        return GroundedAnswer(reply=NOT_FOUND_REPLY, grounded=False)

    best = eligible[0]
    quote = _best_sentence(best.content, query)
    reply = (
        f"According to {best.filename} (part {best.chunk_index + 1}): "
        f"“{quote}”"
    )
    if len(eligible) > 1:
        reply += f"\n\n(+ {len(eligible) - 1} related passage"
        reply += "s" if len(eligible) > 2 else ""
        reply += " cited below)"

    sources = [
        Source(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            score=round(hit.score, 4),
            quote=_best_sentence(hit.content, query),
        )
        for hit in eligible[:3]
    ]
    return GroundedAnswer(reply=reply, sources=sources, grounded=True)


def _best_sentence(chunk_text: str, query: str) -> str:
    """Pick the sentence most lexically similar to the query.

    Deliberately *extractive and cheap*: the sentence must appear verbatim in
    the document — that is what makes the quote trustworthy. (Phase 8 can
    layer an LLM to *phrase* the answer, but the quote stays verbatim.)
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(chunk_text) if s.strip()]
    if not sentences:
        return chunk_text.strip()

    query_terms = {t for t in re.findall(r"[a-z0-9']+", query.lower()) if len(t) > 2}
    if not query_terms:
        return sentences[0]

    def overlap(sentence: str) -> int:
        return sum(1 for t in re.findall(r"[a-z0-9']+", sentence.lower()) if t in query_terms)

    return max(sentences, key=overlap)

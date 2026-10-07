"""Sentence and word tokenization.

Tokenization = splitting raw text into meaningful units. It is the first
conceptual cut in every NLP pipeline: nothing downstream (counting, stopwords,
stemming, intent, entities) can run on a raw string.

Two implementations live here on purpose:

* ``naive_word_tokenize`` — a 3-line regex you can fully understand. It shows
  the *idea* of tokenization and, more importantly, its failure modes.
* ``word_tokenize`` / ``sentence_tokenize`` — NLTK's rule-based tokenizers
  (Treebank/Punkt), which handle punctuation, contractions and sentence
  boundaries properly.

Docs: docs/nlp/02_tokenization.md
"""

from __future__ import annotations

import logging
import re

from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize

logger = logging.getLogger(__name__)

# Fallback sentence splitter: split after . ! ? when followed by whitespace.
# Naive by design — it misfires on abbreviations ("Dr. Smith") and decimals
# ("3.14"), which is exactly why Punkt's trained model exists.
_FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NAIVE_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences (Punkt model; regex fallback if missing)."""
    if not text or not text.strip():
        return []
    try:
        return [s.strip() for s in nltk_sent_tokenize(text) if s.strip()]
    except LookupError:
        _warn_once(
            "punkt_tab",
            "NLTK 'punkt_tab' model missing — using naive regex sentence split. "
            "Fix: python -m app.nlp.nltk_data",
        )
        return [p.strip() for p in _FALLBACK_SENTENCE_RE.split(text.strip()) if p.strip()]


def word_tokenize(text: str) -> list[str]:
    """Split text into word/punctuation tokens (NLTK Treebank rules).

    ``preserve_line=True`` skips sentence splitting, so no model data is
    needed: TreebankWordTokenizer is purely rule-based.
    """
    if not text or not text.strip():
        return []
    return nltk_word_tokenize(text, preserve_line=True)


def naive_word_tokenize(text: str) -> list[str]:
    """Teaching implementation: lowercase + regex on word characters.

    Kept for comparison (tests, notebook, docs). Notice what it gets wrong:
    punctuation glued to words is silently dropped, contractions stay whole,
    URLs become garbage tokens.
    """
    return _NAIVE_WORD_RE.findall(text.lower())

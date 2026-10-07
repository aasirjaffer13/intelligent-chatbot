"""Text preprocessing pipeline.

Pipeline order (this order matters and is documented in docs/nlp/01_preprocessing.md):

    raw text
      -> clean_text()        strip URLs / HTML / control chars, collapse whitespace
      -> lowercase           (optional, part of cleaning)
      -> sentence_tokenize() sentences for diagnostics
      -> word_tokenize()     tokens
      -> strip_punctuation() drop pure-punctuation tokens   (optional)
      -> remove_stopwords()  drop function words            (optional)
      -> stem OR lemmatize   reduce word forms              (optional, exclusive)

KEY PRINCIPLE: every step is individually switchable via ``PreprocessConfig``.
Preprocessing is NOT universally good — models trained on natural text
(transformers, spaCy, embeddings) often perform *worse* on heavily preprocessed
input. We preprocess aggressively only for classical bag-of-words methods
(TF-IDF intent classification, Phase 3).

Docs: docs/nlp/01_preprocessing.md
"""

from __future__ import annotations

import logging
import re
import string

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pydantic import BaseModel, Field, model_validator

from app.nlp.tokenizer import sentence_tokenize, word_tokenize

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\r\n\t\f\v]+")
_WHITESPACE_RE = re.compile(r"\s+")

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


class PreprocessConfig(BaseModel):
    """Switches for each pipeline step. Defaults = the Phase 2 demo pipeline."""

    lowercase: bool = Field(default=True)
    remove_urls: bool = Field(default=True)
    remove_html: bool = Field(default=True)
    remove_punctuation: bool = Field(default=True)
    remove_stopwords: bool = Field(default=True)
    stemming: bool = Field(default=False, description="Porter suffix stripping.")
    lemmatization: bool = Field(default=False, description="WordNet dictionary reduction.")
    language: str = Field(default="english")

    @model_validator(mode="after")
    def _stemming_and_lemma_exclusive(self) -> "PreprocessConfig":
        if self.stemming and self.lemmatization:
            raise ValueError(
                "stemming and lemmatization are mutually exclusive — pick one "
                "reduction strategy per pipeline run"
            )
        return self


class PreprocessResult(BaseModel):
    """Everything the pipeline produced — used by services, tests and logs."""

    original_text: str
    normalized_text: str = Field(description="Cleaned + lowercased text (before token filters).")
    sentences: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(description="Final tokens after all enabled steps.")
    removed_stopwords: list[str] = Field(
        default_factory=list,
        description="Tokens dropped as stopwords (diagnostics/learning).",
    )


# --- individual steps (each pure, each independently testable) ---------------


def clean_text(text: str, config: PreprocessConfig) -> str:
    """String-level normalization: URLs, HTML, control chars, whitespace, case."""
    result = text
    if config.remove_html:
        result = _HTML_RE.sub(" ", result)
    if config.remove_urls:
        result = _URL_RE.sub(" ", result)
    result = _CONTROL_RE.sub(" ", result)
    result = _WHITESPACE_RE.sub(" ", result).strip()
    if config.lowercase:
        result = result.lower()
    return result


def strip_punctuation(tokens: list[str]) -> list[str]:
    """Drop tokens made entirely of punctuation (``!``, ``,``, ``--``).

    Tokens containing letters are kept even if they include punctuation marks
    (``n't``, ``'s``, ``e-mail``) — those carry meaning.
    """
    return [
        token
        for token in tokens
        if token and not all(char in string.punctuation for char in token)
    ]


def get_stopwords(language: str = "english") -> frozenset[str]:
    """Stopword list, cached. Returns an empty set (with a loud log) if the
    NLTK corpus is unavailable — the pipeline then simply removes nothing
    instead of crashing the whole API."""
    try:
        return frozenset(nltk_stopwords.words(language))
    except (LookupError, OSError):
        _warn_once(
            f"stopwords_{language}",
            f"NLTK stopwords corpus '{language}' missing — stopword removal "
            "disabled. Fix: python -m app.nlp.nltk_data",
        )
        return frozenset()


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    """Keep only tokens that are not function words."""
    stopwords = get_stopwords(language)
    return [token for token in tokens if token not in stopwords]


_stemmer: PorterStemmer | None = None
_lemmatizer: WordNetLemmatizer | None = None


def stem_tokens(tokens: list[str]) -> list[str]:
    """Porter stemming: chop inflectional suffixes by rule (``studies -> studi``)."""
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return [_stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """WordNet lemmatization: dictionary lookup to a base form (``studies -> study``).

    Default POS is noun — a known limitation demonstrated in the docs/notebook
    (``running`` as a verb needs ``pos='v'``)."""
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    try:
        return [_lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        _warn_once(
            "wordnet",
            "NLTK 'wordnet' corpus missing — lemmatization disabled. "
            "Fix: python -m app.nlp.nltk_data",
        )
        return list(tokens)


# --- the pipeline ------------------------------------------------------------


def preprocess(text: str, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Run the full preprocessing pipeline over one message."""
    config = config or PreprocessConfig()

    normalized = clean_text(text, config)
    sentences = sentence_tokenize(normalized)
    tokens = word_tokenize(normalized)

    if config.remove_punctuation:
        tokens = strip_punctuation(tokens)

    removed: list[str] = []
    if config.remove_stopwords:
        stopwords = get_stopwords(config.language)
        kept: list[str] = []
        for token in tokens:
            if token in stopwords:
                removed.append(token)
            else:
                kept.append(token)
        tokens = kept

    if config.stemming:
        tokens = stem_tokens(tokens)
    elif config.lemmatization:
        tokens = lemmatize_tokens(tokens)

    return PreprocessResult(
        original_text=text,
        normalized_text=normalized,
        sentences=sentences,
        tokens=tokens,
        removed_stopwords=removed,
    )

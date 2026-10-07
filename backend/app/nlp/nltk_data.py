"""NLTK corpus/model bootstrap.

NLTK ships *code* with pip, but its trained models and word lists (called
"resources") live separately. This module makes that explicit instead of
hiding a network download inside application code:

    python -m app.nlp.nltk_data      # manual one-time setup

The FastAPI lifespan also calls ``ensure_nltk_data()`` on startup so a first
run "just works" — it only touches the network when a resource is actually
missing, and logs loudly (never silently) when it cannot.
"""

from __future__ import annotations

import logging

import nltk

logger = logging.getLogger(__name__)

# resource name (as passed to nltk.download) -> path used by nltk.data.find
# NOTE: zip-based corpora must include the .zip suffix here — nltk.data.find
# does not try it automatically.
REQUIRED_RESOURCES: dict[str, str] = {
    "punkt_tab": "tokenizers/punkt_tab",      # sentence tokenization (directory)
    "stopwords": "corpora/stopwords",         # English stopword list (directory)
    "wordnet": "corpora/wordnet.zip",         # lemmatization dictionary (zip)
    "omw-1.4": "corpora/omw-1.4.zip",         # multilingual synonyms for wordnet
}


def find_missing_resources() -> list[str]:
    """Return the names of resources not present in the local nltk_data dir."""
    missing: list[str] = []
    for name, path in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)
    return missing


def ensure_nltk_data() -> list[str]:
    """Download any missing resources. Returns the names it downloaded.

    Idempotent: with everything present this does zero network I/O.
    Raises RuntimeError if a missing resource cannot be downloaded (offline).
    """
    missing = find_missing_resources()
    if not missing:
        return []

    downloaded: list[str] = []
    for name in missing:
        logger.info("downloading NLTK resource '%s' ...", name)
        ok = nltk.download(name, quiet=True, raise_on_error=True)
        if ok:
            downloaded.append(name)

    if find_missing_resources():  # something still missing after attempts
        raise RuntimeError(
            "NLTK resources still missing after download. "
            "Run manually: python -m app.nlp.nltk_data"
        )
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        fetched = ensure_nltk_data()
        if fetched:
            print(f"Downloaded NLTK resources: {', '.join(fetched)}")
        else:
            print("All NLTK resources already present.")
    except Exception as exc:  # pragma: no cover - manual CLI path
        print(f"FAILED: {exc}")
        raise SystemExit(1) from exc

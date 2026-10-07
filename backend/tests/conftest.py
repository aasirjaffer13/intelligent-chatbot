"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nlp.nltk_data import ensure_nltk_data

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def nltk_data_ready() -> None:
    """Make sure NLTK corpora exist before any NLP test runs.

    If this fails (e.g. offline with no cached data), NLP tests will fail with
    visible assertion errors rather than mysterious crashes mid-pipeline.
    """
    try:
        ensure_nltk_data()
    except Exception as exc:
        logger.warning("could not ensure NLTK data: %s", exc)


@pytest.fixture()
def client() -> TestClient:
    """HTTP client wired to the in-process FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

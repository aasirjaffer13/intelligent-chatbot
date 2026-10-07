"""Intent classification.

Three implementations behind one function (``classify_intent``), following the
project's swap-in rule — simple first, advanced later:

1. ``TfidfCosineClassifier``  — baseline: TF-IDF over training patterns,
   predict = highest cosine similarity. No training artifacts needed.
2. ``SklearnIntentClassifier`` — trained model loaded from artifacts
   (Logistic Regression / Naive Bayes / Linear SVM — chosen by the evaluation
   in ``train_intent_model.py``). Requires ``python -m app.nlp.train_intent_model``.
3. ``KeywordFallbackClassifier`` — last-resort heuristics so the API never
   crashes when artifacts are missing (fresh clone before training).

Anything below UNKNOWN_THRESHOLD becomes the ``unknown`` intent — the
low-confidence fallback required by the spec.

Docs: docs/nlp/03_classification.md
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]          # .../backend
NOVA_DIR = BACKEND_DIR.parent                               # .../nova
DATASET_PATH = NOVA_DIR / "data" / "intents" / "intents.json"
ARTIFACT_DIR = BACKEND_DIR / "artifacts" / "intent"

UNKNOWN = "unknown"
DEFAULT_UNKNOWN_THRESHOLD = 0.35


class IntentResult(BaseModel):
    """What the classifier decided."""

    label: str = Field(description="Intent tag, or 'unknown'.")
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="tfidf | sklearn | keyword_fallback")


class IntentDataset:
    """Parsed intents file (tags, patterns, responses)."""

    def __init__(self, raw: dict) -> None:
        self.version: str = raw.get("version", "?")
        self.intents: list[dict] = raw["intents"]

    @property
    def tags(self) -> list[str]:
        return [intent["tag"] for intent in self.intents]

    def responses_for(self, tag: str) -> list[str]:
        for intent in self.intents:
            if intent["tag"] == tag:
                return intent["responses"]
        return []


def load_dataset(path: Path = DATASET_PATH) -> IntentDataset:
    with path.open(encoding="utf-8") as handle:
        return IntentDataset(json.load(handle))


# --- implementation 1: TF-IDF + cosine baseline ------------------------------


class TfidfCosineClassifier:
    """Nearest-pattern classification in TF-IDF space.

    Fit: vectorize every training pattern once.
    Predict: vectorize the query, cosine against all patterns, aggregate the
    best score per intent label. Confidence = that best score (cosine of
    non-negative TF-IDF vectors already lives in [0, 1]).
    """

    def __init__(self, patterns: list[str], labels: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            # stop_words=None on purpose: who/what/how distinguish intents
            # (see train_intent_model.VECTORIZER_PARAMS notes)
            lowercase=True,
            ngram_range=(1, 2),          # unigrams + bigrams catch "what's up"
            sublinear_tf=True,           # 1 + log(tf): dampens raw counts
        )
        self.pattern_matrix = self.vectorizer.fit_transform(patterns)
        self.labels = np.asarray(labels)

    def predict(self, text: str) -> IntentResult:
        query = self.vectorizer.transform([text])
        sims = cosine_similarity(query, self.pattern_matrix).ravel()
        best_per_label: dict[str, float] = {}
        for label, score in zip(self.labels, sims):
            if score > best_per_label.get(label, -1.0):
                best_per_label[label] = float(score)
        label, confidence = max(best_per_label.items(), key=lambda kv: kv[1])
        if confidence < DEFAULT_UNKNOWN_THRESHOLD:
            return IntentResult(label=UNKNOWN, confidence=confidence, method="tfidf")
        return IntentResult(label=label, confidence=min(confidence, 1.0), method="tfidf")


# --- implementation 2: trained scikit-learn model ----------------------------


class TfidfCosineModel:
    """Sklearn-compatible baseline adapter (operates on TF-IDF *vectors*).

    Lets the TF-IDF + cosine baseline be saved/loaded through the exact same
    artifact interface as sklearn models. ``predict_proba`` returns raw
    cosine similarities in [0, 1] — scores, not normalized probabilities;
    the serving wrapper only uses argmax + max-score thresholding, which is
    identical to what cross-validation evaluated.
    """

    def __init__(self, pattern_matrix, pattern_labels, threshold: float = DEFAULT_UNKNOWN_THRESHOLD):
        self.pattern_matrix = pattern_matrix
        self.pattern_labels = np.asarray(pattern_labels)
        self.threshold = threshold
        self.classes_ = sorted(set(self.pattern_labels.tolist()))

    def predict_proba(self, X) -> np.ndarray:
        sims = cosine_similarity(X, self.pattern_matrix)  # (n_queries, n_patterns)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for col, label in enumerate(self.pattern_labels):
            idx = self.classes_.index(label)
            np.maximum(proba[:, idx], sims[:, col], out=proba[:, idx])
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        best = np.argmax(proba, axis=1)
        return np.asarray([
            self.classes_[i] if proba[row, i] >= self.threshold else UNKNOWN
            for row, i in enumerate(best)
        ])


class SklearnIntentClassifier:
    """Wraps the vectorizer + model saved by train_intent_model.py."""

    def __init__(self, vectorizer: TfidfVectorizer, model, meta: dict) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.meta = meta
        self.threshold: float = float(meta.get("unknown_threshold", DEFAULT_UNKNOWN_THRESHOLD))

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "SklearnIntentClassifier | None":
        vectorizer_path = artifact_dir / "vectorizer.joblib"
        model_path = artifact_dir / "model.joblib"
        meta_path = artifact_dir / "meta.json"
        if not (vectorizer_path.exists() and model_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            joblib.load(vectorizer_path),
            joblib.load(model_path),
            meta,
        )

    def predict(self, text: str) -> IntentResult:
        vector = self.vectorizer.transform([text])
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            classes = list(self.model.classes_)
            best = int(np.argmax(probabilities))
            label, confidence = classes[best], float(probabilities[best])
        else:
            # e.g. uncalibrated LinearSVC: squash decision function into (0,1)
            scores = self.model.decision_function(vector)[0]
            exp = np.exp(scores - np.max(scores))
            probabilities = exp / exp.sum()
            classes = list(self.model.classes_)
            best = int(np.argmax(probabilities))
            label, confidence = classes[best], float(probabilities[best])

        if label != UNKNOWN and confidence < self.threshold:
            return IntentResult(label=UNKNOWN, confidence=confidence, method="sklearn")
        return IntentResult(label=label, confidence=min(confidence, 1.0), method="sklearn")


# --- implementation 3: keyword fallback (no artifacts) -----------------------

_KEYWORD_RULES: dict[str, list[str]] = {
    "greeting": ["hi", "hello", "hey", "greetings", "morning", "howdy"],
    "goodbye": ["bye", "goodbye", "goodnight", "later", "farewell", "leave"],
    "thanks": ["thanks", "thank", "appreciate", "thx"],
    "help": ["help", "stuck", "assist", "support", "how do i"],
    "identity": ["who are you", "what are you", "your name", "are you"],
    "capabilities": ["what can you do", "features", "capable", "able to do"],
    "weather": ["weather", "rain", "forecast", "umbrella", "temperature"],
    "time": ["what time", "current time", "the date", "what day"],
    "password_help": ["password", "log in", "login", "locked out", "credentials"],
    "document_question": ["document", "pdf", "file", "uploaded", "summarize"],
    "small_talk": ["how are you", "joke", "bored", "chat", "funny"],
}


class KeywordFallbackClassifier:
    """Deterministic keyword rules — used only when artifacts are missing."""

    def predict(self, text: str) -> IntentResult:
        lowered = text.lower()
        best_label, hits = UNKNOWN, 0
        for label, keywords in _KEYWORD_RULES.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > hits:
                best_label, hits = label, score
        if best_label == UNKNOWN:
            return IntentResult(label=UNKNOWN, confidence=0.0, method="keyword_fallback")
        confidence = min(0.35 + 0.15 * hits, 0.8)
        return IntentResult(label=best_label, confidence=confidence, method="keyword_fallback")


# --- public API --------------------------------------------------------------


@lru_cache(maxsize=1)
def get_intent_classifier():
    """Load the best available classifier once, lazily."""
    try:
        trained = SklearnIntentClassifier.load()
    except Exception:
        logger.exception("failed to load intent artifacts — falling back")
        trained = None

    if trained is not None:
        logger.info(
            "intent classifier: sklearn (%s, trained %s)",
            trained.meta.get("model_type", "?"),
            trained.meta.get("trained_at", "?"),
        )
        return trained

    dataset = load_dataset()
    patterns: list[str] = []
    labels: list[str] = []
    for intent in dataset.intents:
        patterns.extend(intent["patterns"])
        labels.extend([intent["tag"]] * len(intent["patterns"]))
    logger.warning(
        "intent artifacts not found at %s — using TF-IDF baseline. "
        "Train with: python -m app.nlp.train_intent_model",
        ARTIFACT_DIR,
    )
    return TfidfCosineClassifier(patterns, labels)


def classify_intent(text: str) -> IntentResult:
    """Classify one message. Empty/whitespace input short-circuits to unknown."""
    if not text or not text.strip():
        return IntentResult(label=UNKNOWN, confidence=0.0, method="empty")
    try:
        return get_intent_classifier().predict(text.strip())
    except Exception:
        logger.exception("intent classification failed — returning unknown")
        return IntentResult(label=UNKNOWN, confidence=0.0, method="error")

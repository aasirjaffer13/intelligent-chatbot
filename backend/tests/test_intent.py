from __future__ import annotations

import json

import pytest

from app.nlp.intent import (
    ARTIFACT_DIR,
    DATASET_PATH,
    KeywordFallbackClassifier,
    TfidfCosineClassifier,
    classify_intent,
    load_dataset,
)

EXPECTED_TAGS = {
    "greeting", "goodbye", "thanks", "help", "identity", "capabilities",
    "small_talk", "weather", "time", "password_help", "document_question",
    "unknown",
}


class TestDataset:
    def test_all_expected_intents_present(self) -> None:
        dataset = load_dataset()
        assert set(dataset.tags) == EXPECTED_TAGS

    def test_every_intent_has_enough_diverse_patterns(self) -> None:
        dataset = load_dataset()
        for intent in dataset.intents:
            assert len(intent["patterns"]) >= 15, intent["tag"]
            assert len(set(intent["patterns"])) == len(intent["patterns"]), intent["tag"]
            assert all(p.strip() for p in intent["patterns"]), intent["tag"]

    def test_every_intent_has_responses(self) -> None:
        dataset = load_dataset()
        for intent in dataset.intents:
            assert intent["responses"], intent["tag"]
            assert all(r.strip() for r in intent["responses"]), intent["tag"]


class TestTrainedClassifier:
    """Spec-mandated examples must classify correctly with the saved artifacts."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("hey, what's up?", "greeting"),
            ("I can't remember my login password", "password_help"),
            ("what time is it", "time"),
            ("who are you", "identity"),
            ("thanks a lot", "thanks"),
            ("what can you do", "capabilities"),
            ("what does the document say about gradient descent", "document_question"),
            ("what's the weather like", "weather"),
        ],
    )
    def test_known_intents(self, message: str, expected: str) -> None:
        result = classify_intent(message)
        assert result.label == expected
        assert result.confidence > 0.35
        assert result.method == "sklearn"

    def test_gibberish_falls_back_to_unknown(self) -> None:
        result = classify_intent("qwerty zxcvbn mnbvc xyz 12345")
        assert result.label == "unknown"

    @pytest.mark.parametrize("message", ["", "   ", "\n\t"])
    def test_empty_input_is_unknown_with_zero_confidence(self, message: str) -> None:
        result = classify_intent(message)
        assert result.label == "unknown"
        assert result.confidence == 0.0

    def test_artifacts_metadata_is_valid(self) -> None:
        meta = json.loads((ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))
        assert set(meta["labels"]) == EXPECTED_TAGS
        assert 0.0 < meta["unknown_threshold"] <= 1.0
        assert meta["model_type"]
        assert (ARTIFACT_DIR / "model.joblib").exists()
        assert (ARTIFACT_DIR / "vectorizer.joblib").exists()

    def test_ambiguous_input_returns_plausible_class(self) -> None:
        # "how are you" is genuinely greeting/small_talk — either is defensible.
        result = classify_intent("how are you")
        assert result.label in {"greeting", "small_talk"}


class TestTfidfBaseline:
    """The baseline classifier, tested in isolation on a tiny corpus."""

    def _classifier(self) -> TfidfCosineClassifier:
        patterns = [
            "reset my password", "i forgot my login password",
            "what is the weather today", "will it rain tomorrow",
            "hello there", "hi my friend",
        ]
        labels = ["password_help", "password_help", "weather", "weather", "greeting", "greeting"]
        return TfidfCosineClassifier(patterns, labels)

    def test_nearest_pattern_wins(self) -> None:
        clf = self._classifier()
        assert clf.predict("i forgot my password").label == "password_help"
        assert clf.predict("is it going to rain").label == "weather"
        assert clf.predict("hello").label == "greeting"

    def test_unrelated_text_is_unknown(self) -> None:
        clf = self._classifier()
        assert clf.predict("purple submarine refrigerator sale").label == "unknown"

    def test_confidence_is_bounded(self) -> None:
        clf = self._classifier()
        result = clf.predict("what is the weather outside")
        assert 0.0 <= result.confidence <= 1.0


class TestKeywordFallback:
    def test_detects_obvious_keywords(self) -> None:
        clf = KeywordFallbackClassifier()
        assert clf.predict("please reset my password now").label == "password_help"
        assert clf.predict("what is the weather").label == "weather"

    def test_gibberish_is_unknown(self) -> None:
        result = KeywordFallbackClassifier().predict("zxcvbnmasd qwe")
        assert result.label == "unknown"
        assert result.confidence == 0.0

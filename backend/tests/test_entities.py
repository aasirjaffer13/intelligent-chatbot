from __future__ import annotations

import pytest

from app.nlp.entities import (
    ENTITY_LABELS,
    RuleBasedEntityExtractor,
    SpacyEntityExtractor,
    get_entity_extractor,
)
from app.schemas import ChatRequest
from app.services.chat_service import ChatService


def assert_valid_spans(text: str, spans) -> None:
    """Invariants every extractor must satisfy."""
    for span in spans:
        assert span.label in ENTITY_LABELS, span
        assert 0 <= span.start < span.end <= len(text), span
        assert span.text == text[span.start : span.end], span
        assert 0.0 <= span.confidence <= 1.0, span
    # pairwise non-overlapping
    ordered = sorted(spans, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start, (a, b)


class TestFactory:
    def test_returns_available_extractor(self) -> None:
        extractor = get_entity_extractor()
        assert extractor.name in {"rules", "hybrid"}
        assert hasattr(extractor, "extract")

    def test_empty_and_blank_inputs(self) -> None:
        extractor = get_entity_extractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_hybrid_keeps_provenance_when_model_present(self) -> None:
        extractor = get_entity_extractor()
        if extractor.name != "hybrid":
            import pytest as _pytest

            _pytest.skip("spaCy model not installed — rules-only mode")
        spans = extractor.extract("Dr. Smith ships 42 units to Paris at 3:30pm")
        assert spans
        assert all(s.source in {"rules", "spacy"} for s in spans)

    def test_hybrid_prefers_rule_label_for_ambiguous_time(self) -> None:
        # spaCy's small model calls "3:30pm" CARDINAL in isolation; the
        # hybrid must still surface it as TIME.
        extractor = get_entity_extractor()
        if extractor.name != "hybrid":
            import pytest as _pytest

            _pytest.skip("spaCy model not installed — rules-only mode")
        spans = extractor.extract("the 3:30pm slot")
        assert any(s.label == "TIME" and s.text == "3:30pm" for s in spans)


class TestActiveExtractor:
    """Behavior of whichever extractor production uses."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.extractor = get_entity_extractor()

    def test_full_sentence_all_five_types(self) -> None:
        text = "Dr. Smith meets Ada in Paris at 3:30pm tomorrow with 42 files"
        spans = self.extractor.extract(text)
        # Guaranteed in BOTH modes (rules-only and hybrid):
        # rules cover TIME/DATE, either layer covers PERSON/LOCATION/NUMBER
        assert {e.label for e in spans} == set(ENTITY_LABELS)
        assert_valid_spans(text, spans)

    def test_number_with_commas(self) -> None:
        text = "I paid 1,250.50 dollars on 2026-10-08"
        spans = self.extractor.extract(text)
        assert any(s.label == "NUMBER" and "1,250.50" in s.text for s in spans)
        assert any(s.label == "DATE" and s.text == "2026-10-08" for s in spans)
        assert_valid_spans(text, spans)

    def test_plain_text_has_no_entities(self) -> None:
        assert self.extractor.extract("the quick brown fox jumps") == []
        assert self.extractor.extract("what can you do") == []


class TestRuleBased:
    """Deterministic unit tests — exact spans, independent of any model."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.extractor = RuleBasedEntityExtractor()

    def test_titled_person(self) -> None:
        spans = self.extractor.extract("Dr. Smith called")
        person = [s for s in spans if s.label == "PERSON"]
        assert [s.text for s in person] == ["Dr. Smith"]
        assert person[0].confidence == 1.0

    def test_time_patterns(self) -> None:
        for text, expected in [
            ("call at 3:30pm", "3:30pm"),
            ("call at 9:15", "9:15"),
            ("meet at noon", "noon"),
        ]:
            spans = self.extractor.extract(text)
            assert any(s.label == "TIME" and s.text == expected for s in spans), text

    def test_date_patterns(self) -> None:
        for text, expected in [
            ("due 2026-10-08", "2026-10-08"),
            ("due October 8", "October 8"),
            ("see you tomorrow", "tomorrow"),
            ("next monday works", "next monday"),
        ]:
            spans = self.extractor.extract(text)
            assert any(s.label == "DATE" and s.text == expected for s in spans), text

    def test_gazetteer_location(self) -> None:
        spans = self.extractor.extract("flights to Tokyo and paris")
        locations = [s.text for s in spans if s.label == "LOCATION"]
        assert locations == ["Tokyo", "paris"]

    def test_number_wins_loses_to_time_and_date(self) -> None:
        # "3:30" must not also be reported as NUMBER, "2026-10-08" not split up
        text = "the 3:30 train on 2026-10-08 cost 25"
        spans = self.extractor.extract(text)
        texts = [s.text for s in spans]
        assert "3:30" in [s.text for s in spans if s.label == "TIME"]
        assert "2026-10-08" in [s.text for s in spans if s.label == "DATE"]
        assert not any(s.label == "NUMBER" and "2026" in s.text for s in spans)
        assert any(s.label == "NUMBER" and s.text == "25" for s in spans)
        assert_valid_spans(text, spans)


class TestSpacy:
    """Statistical extractor: label normalization + learned context."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.extractor = SpacyEntityExtractor()

    def test_labels_normalized(self) -> None:
        text = "Ada spent 50 dollars in Paris on 2026-10-08 at noon"
        spans = self.extractor.extract(text)
        assert spans, "expected entities"
        for span in spans:
            assert span.label in ENTITY_LABELS, span.label
        assert_valid_spans(text, spans)

    def test_context_beats_regex(self) -> None:
        # "Ada" has no title — rules can only catch it via the first-name list,
        # spacy catches it from context alone.
        spans = self.extractor.extract("Ada pushed the commit")
        assert any(s.label == "PERSON" and s.text == "Ada" for s in spans)

    def test_source_marked(self) -> None:
        spans = self.extractor.extract("Paris at noon")
        assert all(s.source == "spacy" for s in spans)


class TestChatServiceIntegration:
    def test_entities_populated_in_response(self) -> None:
        response = ChatService().handle(
            ChatRequest(message="meet me in Paris at noon tomorrow")
        )
        labels = {e.label for e in response.entities}
        assert response.entities
        assert {"LOCATION", "TIME", "DATE"} <= labels
        for entity in response.entities:
            assert entity.text
            assert entity.start < entity.end

    def test_message_without_entities(self) -> None:
        response = ChatService().handle(ChatRequest(message="hello there"))
        assert response.entities == []
        assert response.intent == "greeting"

    def test_offsets_reference_original_message(self) -> None:
        message = "shipping to Berlin costs 99"
        response = ChatService().handle(ChatRequest(message=message))
        for entity in response.entities:
            assert message[entity.start : entity.end] == entity.text

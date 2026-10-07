"""Entity extraction (Phase 4).

Two extractors behind ONE interface:

* ``RuleBasedEntityExtractor`` — regex + gazetteers. Deterministic, zero
  dependencies beyond the stdlib, always available. Rules encode *language
  patterns* (dates look like dates, titles precede names), so precision is
  high on the phenomena they cover.
* ``SpacyEntityExtractor`` — statistical NER from ``en_core_web_sm`` behind
  the same ``extract()`` method. Learned context beats regex where rules
  break down (e.g. "Ada" with no title), at the cost of a model dependency.

``get_entity_extractor()`` composes both into a **hybrid** extractor when the
spaCy model loads (precise TIME/DATE rules first, statistical context for the
rest), else falls back to pure rules — the caller never changes. Labels are
normalized to the project's ontology: PERSON, LOCATION, DATE, TIME, NUMBER.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

ENTITY_LABELS = ("PERSON", "LOCATION", "DATE", "TIME", "NUMBER")


@dataclass(frozen=True)
class EntitySpan:
    """One extracted mention. Offsets index the ORIGINAL message text."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    source: str = "rules"

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


class EntityExtractor:
    """Interface both extractors implement (structural typing)."""

    name: str = "abstract"

    def extract(self, text: str) -> list[EntitySpan]:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Rule-based extraction
# --------------------------------------------------------------------------

# Priority: lower number wins overlap resolution. TIME/DATE must claim
# "3:30pm"/"2026-10-08" before the NUMBER regex would slice them apart.
_PRIORITY = {"TIME": 1, "DATE": 2, "PERSON": 3, "LOCATION": 4, "NUMBER": 5}

_TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?"               # 3:30, 3:30 pm, 3:30p.m.
    r"|\b\d{1,2}\s?(?:[ap])\.?m\.?\b"                   # 9 am, 9 a.m.
    r"|\b(?:noon|midnight)\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"                          # ISO 2026-10-08
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may"
    r"|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b"  # Oct 8
    r"|\b(?:next|last|this)?\s*(?:monday|tuesday|wednesday|thursday"
    r"|friday|saturday|sunday)\b"                      # next monday
    r"|\b(?:today|tomorrow|yesterday)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"               # 1,250.50
    r"|\b\d+(?:\.\d+)?\b"                              # 42, 3.14
)

_TITLED_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
)

# Sentence-initial capitals are the classic false-positive trap: only
# mid-sentence Capitalized pairs count as PERSON candidates.
_CAPS_PAIR_RE = re.compile(r"(?<!^)(?<!\.\s)(?<=[a-z,]\s)([A-Z][a-z]+\s+[A-Z][a-z]+)")

# Small gazetteers — precision over recall. Unknown names/places are a job
# for the statistical model, not a regex.
_FIRST_NAMES = {
    "ada", "alice", "bob", "carol", "david", "emma", "frank", "grace",
    "henry", "isabel", "james", "kate", "leo", "maria", "nina", "oliver",
    "priya", "quinn", "rachel", "sam", "tina", "victor", "wendy",
}
_NAME_RE = re.compile(r"\b(" + "|".join(sorted(_FIRST_NAMES)) + r")\b", re.IGNORECASE)

_CITIES = {
    "amsterdam", "athens", "auckland", "bangkok", "barcelona", "beijing",
    "berlin", "bogota", "boston", "brussels", "budapest", "buenos aires",
    "cairo", "capetown", "chicago", "copenhagen", "dubai", "dublin",
    "florence", "helsinki", "hong kong", "istanbul", "jakarta", "kuala lumpur",
    "lisbon", "london", "los angeles", "madrid", "manila", "melbourne",
    "mexico city", "milan", "moscow", "mumbai", "munich", "nairobi",
    "new york", "oslo", "paris", "prague", "rio de janeiro", "rome",
    "san francisco", "san jose", "santiago", "seattle", "seoul", "shanghai",
    "singapore", "stockholm", "sydney", "taipei", "tokyo", "toronto",
    "valencia", "vancouver", "venice", "vienna", "warsaw", "zurich",
}
_COUNTRIES = {
    "argentina", "australia", "austria", "belgium", "brazil", "canada",
    "chile", "china", "colombia", "denmark", "egypt", "finland", "france",
    "germany", "greece", "hungary", "india", "indonesia", "ireland", "italy",
    "japan", "kenya", "korea", "malaysia", "mexico", "morocco", "netherlands",
    "new zealand", "nigeria", "norway", "peru", "philippines", "poland",
    "portugal", "russia", "singapore", "spain", "sweden", "switzerland",
    "thailand", "turkey", "ukraine", "united kingdom",
    "united states", "vietnam",
}
_GAZETTEER = _CITIES | _COUNTRIES
_GAZETTEER_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(_GAZETTEER, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Priority order the patterns claim spans in.
_PATTERN_ORDER = (
    ("TIME", _TIME_RE, 1.0),
    ("DATE", _DATE_RE, 1.0),
    ("PERSON", _TITLED_NAME_RE, 1.0),
    ("LOCATION", _GAZETTEER_RE, 1.0),
    ("PERSON", _NAME_RE, 0.85),
    ("PERSON", _CAPS_PAIR_RE, 0.7),
    ("NUMBER", _NUMBER_RE, 1.0),
)


def _overlaps(a_start: int, a_end: int, taken: list[tuple[int, int]]) -> bool:
    return any(a_start < e and s < a_end for s, e in taken)


class RuleBasedEntityExtractor(EntityExtractor):
    """Regex + gazetteer extraction. Deterministic and dependency-free."""

    name = "rules"

    def extract(self, text: str) -> list[EntitySpan]:
        if not text or not text.strip():
            return []

        taken: list[tuple[int, int]] = []
        spans: list[EntitySpan] = []

        for label, pattern, confidence in _PATTERN_ORDER:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if _overlaps(start, end, taken):
                    continue
                taken.append((start, end))
                spans.append(
                    EntitySpan(
                        text=text[start:end],
                        label=label,
                        start=start,
                        end=end,
                        confidence=confidence,
                        source=self.name,
                    )
                )

        spans.sort(key=lambda s: s.start)
        return spans


# --------------------------------------------------------------------------
# spaCy extraction (statistical, same interface)
# --------------------------------------------------------------------------

# spaCy's ontology -> project ontology. Anything unmapped (ORG, PRODUCT...)
# is intentionally dropped: the contract promises a small, predictable set.
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "CARDINAL": "NUMBER",
    "QUANTITY": "NUMBER",
    "ORDINAL": "NUMBER",
    "MONEY": "NUMBER",
    "PERCENT": "NUMBER",
}

_SPACY_CONFIDENCE = 0.9  # statistical model, not calibrated — documented heuristic


class SpacyEntityExtractor(EntityExtractor):
    """Statistical NER via en_core_web_sm, labels normalized to ENTITY_LABELS."""

    name = "spacy"

    def __init__(self) -> None:
        import spacy

        self._nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> list[EntitySpan]:
        if not text or not text.strip():
            return []
        doc = self._nlp(text)
        spans = [
            EntitySpan(
                text=ent.text,
                label=_SPACY_LABEL_MAP[ent.label_],
                start=ent.start_char,
                end=ent.end_char,
                confidence=_SPACY_CONFIDENCE,
                source=self.name,
            )
            for ent in doc.ents
            if ent.label_ in _SPACY_LABEL_MAP
        ]
        spans.sort(key=lambda s: s.start)
        return spans


# --------------------------------------------------------------------------
# Hybrid extraction — composition of both, best available recall/precision
# --------------------------------------------------------------------------

class HybridEntityExtractor(EntityExtractor):
    """Rules for regex-shaped phenomena, spaCy for context, merged.

    Layering rationale:

    1. Rule TIME/DATE claim their spans first — they are precise patterns
       and spaCy's small model is unreliable here ("3:30pm" sometimes comes
       back as CARDINAL, "tomorrow" can be missed entirely).
    2. spaCy fills in the rest (PERSON/LOCATION from *context*, not just a
       wordlist) minus overlaps with layer 1.
    3. Remaining rule spans (gazetteer LOCATION, titled PERSON, NUMBER)
       fill any gap neither layer covered.

    Each span keeps its provenance in ``span.source`` ("rules"/"spacy").
    """

    name = "hybrid"

    def __init__(self, rules: RuleBasedEntityExtractor, spacy: SpacyEntityExtractor) -> None:
        self._rules = rules
        self._spacy = spacy

    def extract(self, text: str) -> list[EntitySpan]:
        if not text or not text.strip():
            return []

        rule_spans = self._rules.extract(text)
        rule_time_date = [s for s in rule_spans if s.label in ("TIME", "DATE")]
        taken: list[tuple[int, int]] = [(s.start, s.end) for s in rule_time_date]

        merged = list(rule_time_date)
        for span in self._spacy.extract(text):
            if not _overlaps(span.start, span.end, taken):
                merged.append(span)
                taken.append((span.start, span.end))

        for span in rule_spans:
            if span.label in ("TIME", "DATE"):
                continue
            if not _overlaps(span.start, span.end, taken):
                merged.append(span)
                taken.append((span.start, span.end))

        merged.sort(key=lambda s: s.start)
        return merged


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_entity_extractor() -> EntityExtractor:
    """Best available extractor: hybrid (spaCy + rules) if the model loads,
    else pure rules.

    Cached: the model loads once per process. Failures degrade silently to
    the rule extractor — the API never breaks because a model is missing.
    """
    try:
        return HybridEntityExtractor(RuleBasedEntityExtractor(), SpacyEntityExtractor())
    except Exception:
        return RuleBasedEntityExtractor()


def extract_entities(text: str) -> list[EntitySpan]:
    """Convenience wrapper used by the chat service."""
    return get_entity_extractor().extract(text)

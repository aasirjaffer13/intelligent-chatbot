# NLP Concept: Named Entity Recognition (NER)

> **Project phase:** 4 · **Code:** `backend/app/nlp/entities.py` ·
> **Notebook:** `04_ner.ipynb`

---

## 1. What is it?

NER finds and classifies **spans of text that refer to real-world things** —
people, places, dates, times, quantities — and labels each with a type:

```
"Dr. Smith meets Ada in Paris at 3:30pm tomorrow with 42 files"
 └─PERSON────────┘        └LOC──┘  └──TIME───┘ └DATE──┘   └NUM─┘
```

Where intent classification answers *what does the user want*, NER answers
*which specific things are they talking about* — the slots that turn a
generic reply into a specific one ("see you **tomorrow** in **Paris**").

## 2. Why it exists?

- **Grounding:** a chatbot that ignores entities gives the same reply to
  "remind me at 3pm" and "remind me at 6pm".
- **Structure:** entities are the bridge from free text to structured data
  (a calendar entry, a search query, an API call — Phase 9 tools).
- **Efficiency:** extracting "October 8" costs microseconds; asking an LLM to
  re-read the message for every field costs money and latency.

## 3. The task, precisely — BIO tagging

Sequence labeling frames NER as a per-token tag:

| Token | Dr. | Smith | meets | Paris |
|---|---|---|---|---|
| BIO tag | B-PERSON | I-PERSON | O | B-LOCATION |

- `B-` begin a new entity, `I-` inside (continue) it, `O` outside.
- Multi-token entities ("**New York**" = B-LOCATION, I-LOCATION) are why
  naive per-word classifiers fail — the model must know *boundaries*.
- Our extractor instead returns **character offsets** (`start`, `end`) over
  the original message — the same information, API-friendly (offsets let the
  frontend highlight spans in place).

## 4. Rule-based NER (what we built first)

Rules encode *what entities look like*:

| Type | Signal | Rule |
|---|---|---|
| TIME | digits-colon-digits, am/pm, noon | `\b\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?` |
| DATE | ISO, month-day, weekdays, today/tomorrow | regex alternation |
| NUMBER | int/decimal/comma-grouped digits | `\b\d{1,3}(?:,\d{3})+...\|\b\d+(?:\.\d+)?\b` |
| PERSON | titles (`Dr. Smith`) + first-name/capitalized-pair gazetteers | patterns + wordlists |
| LOCATION | city/country gazetteer, case-insensitive | wordlist |

**Overlap resolution:** patterns claim spans in priority order
(TIME → DATE → PERSON → LOCATION → NUMBER) so `3:30pm` never also becomes
`3` + `:30` + `pm` NUMBER fragments, and `2026-10-08` is not sliced into
`2026` and `10` and `08`.

**Pros:** deterministic, auditable, instant, no model, confidence = 1.0 for
covered phenomena.
**Cons:** wordlists never finish (unknown names/places missed); "Dr. Smith"
needs a title rule someone wrote; context-blind ("Apple" the fruit vs the
company — we don't even try ORG).

## 5. Statistical NER (spaCy)

A trained model (we ship `en_core_web_sm`, ~12 MB) labels tokens from
**context**, not spelling:

- `Ada pushed the commit` → `Ada` is PERSON even with no title and absent
  from any list — the verb pattern taught it.
- Trained on OntoNotes-style annotated data via a neural tagger
  (transition-based parser head reused for BIO decoding).

Its weaknesses on small models: `3:30pm` sometimes becomes CARDINAL
(NUMBER) without sentence context; rare spellings misfire; latency ~1–3 ms
vs ~0.1 ms for rules; the label set is its own ontology (PERSON, GPE,
CARDINAL, MONEY…) which **we map to ours**: GPE/LOC/FAC → LOCATION,
CARDINAL/QUANTITY/ORDINAL/MONEY/PERCENT → NUMBER.

## 6. Why we compose them — the hybrid

Neither layer alone is reliable, so `HybridEntityExtractor` stacks them:

1. **Rule TIME/DATE claim first** — regexes are precise where spaCy wobbles.
2. **spaCy fills** PERSON/LOCATION/NUMBER contextually, skipping overlaps.
3. **Remaining rules** (gazetteers, titled names) cover anything left.

Each span keeps provenance (`source: rules|spacy`) so you can debug which
layer produced it. This is the classic production pattern: *rules for what
you can formalize, models for what you can't, composition for the rest.*

## 7. Evaluation

Span-level (not token-level) metrics:

- **Exact match:** predicted span must equal gold span (text + offsets + label).
- **Precision** = correct predictions / all predictions
- **Recall** = correct / all gold spans · **F1** = harmonic mean
- Partial-match scoring (overlap counts half) exists but exact is what we
  report — false boundaries are real errors for offset-consuming UIs.

## 8. How we implemented it

- `entities.py` — `EntitySpan` contract, `RuleBasedEntityExtractor`,
  `SpacyEntityExtractor`, `HybridEntityExtractor`, `get_entity_extractor()`
  (cached factory; **degrades to rules if the model is missing** — the API
  never 500s because of a dependency).
- `chat_service.py` runs extraction after intent and maps spans onto the
  stable `Entity` schema (`text`, `label`, `start`, `end`, `confidence`).

Example (`/api/chat`):

```json
{"message": "meet me in Paris at noon tomorrow"}
...
"entities": [
  {"text": "Paris",    "label": "LOCATION", "start": 8,  "end": 13, "confidence": 0.9},
  {"text": "noon",     "label": "TIME",     "start": 17, "end": 21, "confidence": 1.0},
  {"text": "tomorrow", "label": "DATE",     "start": 22, "end": 30, "confidence": 1.0}
]
```

## 9. Limitations

1. **Closed label set** — no ORG, PRODUCT, EVENT; "remind me to call Mom"
   loses `Mom` (no title, not in gazetteer, spaCy may catch it — may not).
2. **No coreference** — "I spoke to Dr. Smith. He said…" — `He` is not
   linked back to Smith.
3. **Ambiguity is out of scope** — "Apple shipped 2 million units": is
   Apple LOCATION? No — but our labels have no ORG bucket, so it's dropped
   rather than mislabeled (deliberate: a wrong label is worse than none).
4. **Exact-match evaluation is brutal** — one boundary off (`Dr. Smith` vs
   `Smith`) scores 0 for that span.
5. **Gazetteers rot** — cities go by many names (Bombay/Mumbai…).

## 10. How modern systems improve it

- **Transformer NER** (BERT-CRF, spaCy `trf`): ~90–93% F1 vs ~85% for small
  models, at 10–50× the compute.
- **LLM extraction:** one prompt handles arbitrary label sets and few-shot
  examples — but is nondeterministic and slow; fine for offline pipelines,
  not for a latency budget.
- **Entity linking** (NER + knowledge base): resolving "Paris" to
  `wikidata:Q90` disambiguates Paris-Texas from Paris-France.
- **Our architecture's answer:** keep rules+spaCy for the routing-critical
  fast path; let Phase 8's LLM provider handle open-ended extraction *only
  when a tool call actually needs it*.

---

*Previous: [03_classification.md](03_classification.md) · Next: embeddings (Phase 5)*

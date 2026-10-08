# NOVA — Roadmap

Nine build phases (V1–V9) plus a frontend polish phase. Each phase is complete
only when it **runs, is tested, and is documented** — then we stop and review.

> Mapping to the original vision: V1–V9 ≙ Phases 1–9 below; Phase 10 is the
> production-grade UI pass layered on top.

---

## Phase 1 — Project foundation ✅
*V1 in spirit: the skeleton everything else hangs on.*

- [x] Monorepo layout: `backend/`, `frontend/`, `data/`, `notebooks/`, `docs/`
- [x] FastAPI app factory, typed config (`pydantic-settings`), CORS, logging
- [x] Typed exceptions + JSON error handlers
- [x] `GET /api/health`
- [x] `POST /api/chat` — validated **stub** with the final response contract
- [x] Pydantic schemas: `ChatRequest`, `ChatResponse`, `Entity`, `ProcessingInfo`
- [x] pytest suite (health, chat contract, validation)
- [x] React + Vite + Tailwind chat UI: bubbles, typing indicator, health pill,
      error banner, suggestion chips, responsive layout
- [x] Dev proxy `/api → :8000`, normalized API errors in `services/api.js`
- [x] Docs: README, ARCHITECTURE, ROADMAP, NLP_LEARNING

**Exit criteria met:** backend verified, frontend verified, tests green.

---

## Phase 2 — NLP preprocessing pipeline (rule-based NLP) ✅
*V1*

- [x] `nlp/preprocessing.py`: lowercase, cleaning, stopwords, stemming, lemmatization
- [x] `nlp/tokenizer.py`: sentence + word tokenization (NLTK + naive baseline)
- [x] `nlp/nltk_data.py`: explicit corpus bootstrap (`python -m app.nlp.nltk_data`)
- [x] `services/chat_service.py`: route → pipeline → response
- [x] Populated `processing.tokens` / `processing.normalized_text` / `processing.sentences`
- [x] Unit tests: 46 tests across tokenizer/preprocessing/service/API
- [x] Docs: `docs/nlp/01_preprocessing.md`, `docs/nlp/02_tokenization.md`
- [x] Notebook: `01_text_preprocessing.ipynb`

**Exit criteria met:** every preprocessing step independently switchable and
tested; pipeline integrated into `/api/chat` without changing the contract.

## Phase 3 — Intent classification
*V2 + V3*

- [x] Intent dataset in `data/intents/` (12 intents incl. greeting, goodbye,
      thanks, help, identity, capabilities, weather, time, small_talk,
      password_help, document_question, unknown — 489 patterns, v1.3)
- [x] **First:** TF-IDF + cosine similarity (understand the math)
- [x] **Then:** scikit-learn classifier (Multinomial NB / Linear SVM / Logistic)
- [x] Evaluation: accuracy, precision, recall, F1, confusion matrix; compare both
      (stratified 5-fold CV; Linear SVM calibrated selected — macro F1 0.757 ± 0.027)
- [x] `nlp/train_intent_model.py` — trains once, **saves artifacts** (never
      retrain on server start)
- [x] Docs: TF-IDF equation, cosine similarity, classification metrics
      (`docs/nlp/03_classification.md`, `docs/reports/intent_model.md`)
- [x] Notebooks: `02_tf_idf.ipynb`, `03_intent_classification.ipynb`

**Exit criteria met:** artifact-trained classifier behind a stable
`classify_intent()` API with baseline/keyword/unknown fallbacks; intent and
confidence populated in `/api/chat` responses; 70 tests passing.

## Phase 4 — Entity extraction
*V3*

- [x] Rule-based NER: PERSON, LOCATION, DATE, TIME, NUMBER (regex + gazetteers)
- [x] `EntityExtractor` abstraction → spaCy NER drops in unchanged
      (plus a hybrid: precise TIME/DATE rules first, spaCy context for the rest)
- [x] Entities returned in `ChatResponse.entities` (offsets into original text)
- [x] Docs: NER, BIO tagging, sequence labeling; rule-based vs statistical
      (`docs/nlp/04_entity_extraction.md`)
- [x] Notebook: `04_ner.ipynb`

**Exit criteria met:** all five labels extracted with valid non-overlapping
offsets; factory degrades to rules when the spaCy model is missing; 88 tests
passing.

## Phase 5 — Semantic search (embeddings)
*V4*

- [x] `services/embedding_service.py` + `services/similarity_service.py`
- [x] sentence-transformers; demonstrate paraphrase matching
      ("I forgot my password" ≈ "I can't remember my login password", 0.788)
- [x] Intent detection gains an embedding-based path alongside TF-IDF
      (`EmbeddingIntentClassifier`, opt-in via `NOVA_INTENT_BACKEND` /
      `classify_intent(backend="embedding")`)
- [x] Docs: word/sentence embeddings, vector space, cosine similarity vs TF-IDF
      (`docs/nlp/05_embeddings.md`)
- [x] Notebook: `05_embeddings.ipynb`

**Exit criteria met:** paraphrase ranks 0.788 vs unrelated 0.001; embedding
backend classifies all spec examples correctly with unknown fallback; 115
tests passing.

## Phase 6 — Conversation memory
*V5 support layer*

- [x] PostgreSQL + SQLAlchemy: `Conversation`, `Message` (timestamps, session id)
      (SQL store on portable SQLAlchemy; in-memory fallback + SQLite-tested)
- [x] Session-scoped recent-context window injected into responses
      (`NOVA_MEMORY_WINDOW`, default 12 messages)
- [x] "My name is Aasir" → later "What is my name?" works **from real stored
      history** (no hardcoding — proven with Zephyr/Echo/River in tests)
- [x] Memory abstraction so the store can be swapped (in-memory → Postgres)
      (`MemoryStore` ABC; `get_memory_store()` picks from `NOVA_DATABASE_URL`)

**Exit criteria met:** 137 tests passing; name recall + session isolation +
auto session creation verified end-to-end through the API.

## Phase 7 — RAG ✅
*V5–V6*

- [x] Document upload (PDF/text) → extract → chunk → embed → store
      (dual-mode: pgvector when `CREATE EXTENSION vector` succeeds, numpy
      cosine fallback otherwise — verified on SQLite **and** PostgreSQL 18)
- [x] Similarity search → top-k chunks → grounded answer + **source citations**
      (extractive quoted sentence, `sources[]` in `ChatResponse`, refusal
      below `NOVA_RAG_MIN_SCORE` instead of guessing)
- [x] Endpoints: `POST /api/documents/upload`, `GET /api/documents`,
      `DELETE /api/documents/{id}`
- [x] Strict separation: document-derived knowledge vs general conversation;
      refuse to answer beyond retrieved context (no hallucinated document facts)
- [x] Docs: chunking strategies, vector databases, grounding (`docs/07_rag.md`)
- [x] Notebook: `07_rag.ipynb`

**Exit criteria met:** 166 tests passing; PostgreSQL acceptance passed
(ingest → list → top-k retrieval 0.367 → grounded quoted answer → delete
with zero orphaned chunks); unanswerable questions refused with no citations.

## Phase 8 — LLM integration ✅
*V6*

- [x] `LLMProvider` interface: `OpenAIProvider`, `HuggingFaceProvider`,
      `LocalModelProvider` (+ `MockProvider` for offline tests, and **no
      provider** when `auto` finds no credentials — templates stay the default)
- [x] Keys exclusively via environment variables (`SecretStr`, never
      logged/repr'd; `OPENAI_API_KEY`, `HF_TOKEN` / `NOVA_*` alternates)
- [x] LLM used for *response generation* over NLP pipeline outputs —
      the pipeline still decides intent/entities; time, name recall and
      RAG quotes never reach the model; every failure falls back to templates
- [x] Docs: prompt construction, provider abstraction
      (`docs/08_llm_integration.md`)

**Exit criteria met:** 200 tests passing (35 offline LLM tests incl.
MockTransport auth/payload checks, factory auto-selection with scrubbed
env, chat integration proving LLM → template fallback and routing
priority).

## Phase 9 — Tools + agent architecture ✅
*V7 + V8 + V9*

- [x] Tool registry: calculator, current_time, document_search, web_search,
      weather — register a class, it appears in the next prompt; duplicate
      names rejected; errors returned as observations, never raised
- [x] Loop: user → LLM → tool decision → tool → result → LLM → final answer
      (ReAct-style, bounded by `NOVA_AGENT_MAX_STEPS`, forgiving JSON
      parsing, full step trace in `AgentResult.steps`)
- [x] Tools are modular: `Tool` ABC + `ToolRegistry.register()`;
      calculator uses an AST whitelist (never `eval`); HTTP tools are
      offline-testable via injectable transports
- [x] Conversation memory (V7) feeds the agent loop — the prompt builder
      includes the memory window; time/name/RAG routes still bypass the loop
- [x] Docs: tool calling, ReAct-style agents, when agents help vs hurt
      (`docs/09_agents.md`)

**Exit criteria met:** 249 tests passing (49 offline agent tests:
tool→observation→final round trip, unknown/crashing tools, max-steps
rail, deterministic bypass, template fallback on provider failure).

## Phase 10 — Frontend production pass ✅
- [x] Streaming responses (`POST /api/chat/stream`, SSE `delta`/`meta`/`end`
      events; browser parses the stream over fetch and repaints the
      assistant bubble chunk by chunk)
- [x] Markdown + fenced code blocks with copy buttons (react-markdown +
      remark-gfm, hand-styled so it follows the theme)
- [x] Sidebar with conversation list, new-conversation button, and
      conversation replay (`GET /api/conversations` + `…/{id}/messages`)
- [x] Dark/light mode toggle (one `light` class on `<html>` re-themes the
      whole app: Tailwind v4 utilities compile to `var(--color-*)`, so
      `index.css` mirrors the palette under `:root.light`)
- [x] Loading/error/empty states (history spinner, stream cursor,
      sidebar skeletons, offline badge, dismissible error banner)
- [x] RAG source citation chips (file · chunk · score) + document
      upload/delete UI in the sidebar
- [x] Model/status indicator (`GET /api/status`: LLM provider/model,
      RAG mode + doc count, memory backend, agent step budget)

**Exit criteria met:** 257 tests passing (8 new: SSE contract with exact
delta reassembly, RAG sources over the stream, conversations list/history
round trip + 404 envelope, status shape without secret leakage); frontend
production build clean; live smoke test of all four endpoints against
uvicorn + PostgreSQL.

---

## Rules of engagement

1. One phase at a time — no skipping ahead "just because it's easy".
2. Every phase: working code → tests → run it → document the NLP concept → stop.
3. Simple readable implementation **before** any advanced model.
4. New dependencies must justify themselves (documented in `requirements.txt`).

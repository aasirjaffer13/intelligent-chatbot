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

- [ ] Rule-based NER: PERSON, LOCATION, DATE, TIME, NUMBER (regex + gazetteers)
- [ ] `EntityExtractor` abstraction → spaCy NER drops in unchanged
- [ ] Entities returned in `ChatResponse.entities`
- [ ] Docs: NER, BIO tagging, sequence labeling; rule-based vs statistical
- [ ] Notebook: `05_ner.ipynb`

## Phase 5 — Semantic search (embeddings)
*V4*

- [ ] `services/embedding_service.py` + `services/similarity_service.py`
- [ ] sentence-transformers; demonstrate paraphrase matching
      ("I forgot my password" ≈ "I can't remember my login password")
- [ ] Intent detection gains an embedding-based path alongside TF-IDF
- [ ] Docs: word/sentence embeddings, vector space, cosine similarity vs TF-IDF
- [ ] Notebook: `04_embeddings.ipynb`

## Phase 6 — Conversation memory
*V5 support layer*

- [ ] PostgreSQL + SQLAlchemy: `Conversation`, `Message` (timestamps, session id)
- [ ] Session-scoped recent-context window injected into responses
- [ ] "My name is Aasir" → later "What is my name?" works **from real stored
      history** (no hardcoding)
- [ ] Memory abstraction so the store can be swapped (in-memory → Postgres)

## Phase 7 — RAG
*V5–V6*

- [ ] Document upload (PDF/text) → extract → chunk → embed → store
      (pgvector)
- [ ] Similarity search → top-k chunks → grounded answer + **source citations**
- [ ] Endpoints: `POST /api/documents/upload`, `GET /api/documents`,
      `DELETE /api/documents/{id}`
- [ ] Strict separation: document-derived knowledge vs general conversation;
      refuse to answer beyond retrieved context (no hallucinated document facts)
- [ ] Docs: chunking strategies, vector databases, grounding
- [ ] Notebook: `07_rag.ipynb`

## Phase 8 — LLM integration
*V6*

- [ ] `LLMProvider` interface: `OpenAIProvider`, `HuggingFaceProvider`,
      `LocalModelProvider`
- [ ] Keys exclusively via environment variables
- [ ] LLM used for *response generation* over NLP pipeline outputs —
      the pipeline still decides intent/entities
- [ ] Docs: prompt construction, provider abstraction

## Phase 9 — Tools + agent architecture
*V7 + V8 + V9*

- [ ] Tool registry: calculator, current_time, document_search, web_search, weather
- [ ] Loop: user → LLM → tool decision → tool → result → LLM → final answer
- [ ] Tools are modular: register a class, it becomes available
- [ ] Conversation memory (V7) feeds the agent loop
- [ ] Docs: tool calling, ReAct-style agents, when agents help vs hurt

## Phase 10 — Frontend production pass
- [ ] Streaming responses, markdown + code blocks with copy buttons
- [ ] Sidebar with conversation list, new-conversation button
- [ ] Dark/light mode toggle, full loading/error/empty states
- [ ] RAG source citations + document upload UI, model/status indicator

---

## Rules of engagement

1. One phase at a time — no skipping ahead "just because it's easy".
2. Every phase: working code → tests → run it → document the NLP concept → stop.
3. Simple readable implementation **before** any advanced model.
4. New dependencies must justify themselves (documented in `requirements.txt`).

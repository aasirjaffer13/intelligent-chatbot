# NOVA — Architecture

This document explains how the system is layered **today (Phases 1–10
complete)** — the original Phase 1 plan grew in place, never by rewrite.

---

## 1. Layering

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite + Tailwind)                         │
│  components (presentational) ← hooks (state) ← services/api │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP/JSON + SSE streams
                           │  (dev: Vite proxy /api → :8000)
┌──────────────────────────▼──────────────────────────────────┐
│  API layer — app/api/                                       │
│  routes/ parse requests, call services, return schemas      │
├─────────────────────────────────────────────────────────────┤
│  Services — app/services/                                  │
│  orchestration: preprocess → intent → entities → respond    │
├─────────────────────────────────────────────────────────────┤
│  NLP — app/nlp/                                             │
│  preprocessing · tokenizer · intent · entities · similarity │
├──────────────┬───────────────┬──────────────┬───────────────┤
│ memory/      │ rag/          │ models/      │ llm/          │
│ [Phase 6]    │ [Phase 7]     │ [Phase 6]    │ [Phase 8]     │
├──────────────┼───────────────┼──────────────┼───────────────┤
│ tools/       │ agent/        │              │               │
│ [Phase 9]    │ [Phase 9]     │              │               │
├──────────────┴───────────────┴──────────────┴───────────────┤
│  Core — app/core/  (logging, typed exceptions)              │
│  Schemas — app/schemas/ (Pydantic: the public contract)     │
│  Config — app/config.py (typed .env, NOVA_ prefix)          │
└─────────────────────────────────────────────────────────────┘
```

**Dependency rule:** arrows point downward only. `api` knows about `services`,
`services` knows about `nlp`, never the reverse. The NLP layer never imports
FastAPI, so every NLP function is unit-testable with a plain string.

---

## 2. Request flow today (Phase 2)

```
POST /api/chat {"message": "hello, how are you?"}
        │
        ▼
ChatRequest (Pydantic) ── validates: non-blank, ≤4000 chars
        │
        ▼
route: chat.py ── thin: parse → call service → return
        │
        ▼
services/chat_service.ChatService.handle()
        │
        ├── nlp/preprocessing.preprocess()
        │      clean → lowercase → sentence/word tokenize
        │      → strip punctuation → remove stopwords → (stem|lemma)
        │
        └── _generate_reply()   [Phase 3: per-intent templates
                                 Phase 8: LLM generation
                                 Phase 9: agent tool loop]
        │
        ▼
ChatResponse (Pydantic) ── stable contract for every future phase
        │
        ▼
JSON to frontend → useChat() → message bubble
```

Phase 3 inserts intent classification *inside* `ChatService.handle()`.
Request/response schemas stay identical, so the frontend does not change.

---

## 3. The swap-in pattern (how the project evolves)

Every NLP component follows one rule:

> **Simple implementation first, advanced model later, same function signature.**

Example — intent detection:

```python
# Phase 3 (first implementation)
def detect_intent(text: str) -> IntentResult:      # TF-IDF + cosine similarity

# Phase 3 (upgrade)
def detect_intent(text: str) -> IntentResult:      # scikit-learn classifier
       # same signature → caller never changes

# Phase 5 (upgrade)
def detect_intent(text: str) -> IntentResult:      # embedding similarity
```

The caller (`chat_service`) is written against the signature once. Upgrades
happen *inside* the module. This is the Strategy pattern applied to NLP —
and it is why the `ChatResponse` schema was fully defined in Phase 1.

The same pattern applies to:

| Component | Module | First version | Later version |
| --- | --- | --- | --- |
| Tokenization | `nlp/tokenizer.py` | whitespace/regex | spaCy, HF tokenizers (Phase 5) |
| Preprocessing | `nlp/preprocessing.py` | regex + NLTK | model-specific pipelines |
| Intent | `nlp/intent.py` | TF-IDF + cosine | sklearn → transformer fine-tune |
| Entities | `nlp/entities.py` | regex/lookup rules | spaCy NER (Phase 4) |
| Similarity | `nlp/similarity.py` | TF-IDF vectors | sentence-transformers (Phase 5) |
| Response gen | `services/chat_service.py` | template/rule | LLM via `LLMProvider` (Phase 8) |

---

## 4. Frontend architecture

```
ChatPage (composition root)
├── Sidebar           — new chat, conversation list (replay), document
│                       upload/delete, mobile overlay (useDocuments)
├── ChatHeader        — menu, logo, runtime pill (GET /api/status),
│                       health pill (useHealth), theme toggle, new chat
├── MessageList       — scroll area, empty state, history spinner, auto-scroll
│   └── MessageBubble — user (right, indigo) vs NOVA (left, slate)
│       ├── MarkdownContent  — react-markdown + GFM (tables, lists, quotes)
│       │   └── CodeBlock    — fenced code with Copy button
│       └── source chips     — file · chunk · score (RAG citations)
├── error banner      — dismissible, fed by useChat().error
└── ChatInput         — auto-resizing textarea, Enter to send

hooks/useChat.js      — state machine + SSE streaming (delta → meta swap)
hooks/useTheme.js     — dark/light, one `light` class on <html>
hooks/useDocuments.js — upload/delete document state
hooks/useHealth.js    — polls GET /api/health every 30s
services/api.js       — fetch wrapper, error normalization,
                        streamChat() SSE parser (fetch + ReadableStream)
index.css             — palette vars; :root.light mirrors the ramp so
                        every Tailwind class re-themes with one class
```

Components never call `fetch` directly. When the API grows, only
`services/api.js` and the hooks change.

**Why the Vite proxy?** In dev, the browser calls `localhost:5173/api/...`,
Vite forwards to `localhost:8000`. Same-origin → zero CORS configuration.
In production, set `VITE_API_BASE_URL` (or serve both from one origin).

---

## 5. Configuration & secrets

- `backend/app/config.py` — `pydantic-settings`, env prefix `NOVA_`, reads `.env`.
- Never import `os.environ` directly; always go through `get_settings()`.
- `.env` is gitignored; `.env.example` documents every variable.
- Phase 7+ secrets (DB URL, API keys) follow the same pattern.

---

## 6. Error handling & logging

- `core/logging.py` — one formatter, called once in `create_app()`.
- `core/exceptions.py` — raise `NovaError` subclasses in services/NLP; handlers
  return `{"error": {"code", "detail"}}` and log the stack trace server-side.
- Unexpected exceptions → generic 500 JSON; details never leak to the client.

---

## 7. Testing strategy

| Level | Tool | What it covers |
| --- | --- | --- |
| API contract | `fastapi.testclient` + pytest | Status codes, exact JSON shapes (`tests/`) |
| NLP unit tests | pytest | Each pure function: tokens, stems, intents (Phase 2+) |
| Training evaluation | training script metrics | accuracy/P/R/F1/confusion matrix (Phase 3) |
| LLM providers | pytest + `httpx.MockTransport` | auth, payloads, error paths, template fallback (Phase 8) |
| Agent loop | pytest + scripted provider | tool execution, observations, step rail (Phase 9) |
| Streaming + browsing | pytest + `client.stream` | SSE delta reassembly = contract, conversations round trip, status shape (Phase 10) |

Tests run without a database, network or model downloads — they must always be
fast and offline.

---

## 8. Infrastructure status

| Piece | Arrives | Notes |
| --- | --- | --- |
| PostgreSQL + SQLAlchemy | Phase 6 ✅ | conversations, messages; RAG chunks/documents on the same engine (in-memory/SQLite fallback when no URL) |
| pgvector | Phase 7 ⚙️ | schema + query path built (dual-mode); extension pending on this machine → numpy cosine fallback until installed |
| LLM providers | Phase 8 ✅ | OpenAI / HuggingFace / local transformers behind one interface; `auto` degrades to templates without keys |
| Redis (optional) | later | caching / rate limiting |
| Model artifacts (`backend/artifacts/`) | Phase 3 ✅ | trained models saved, never retrained on startup |

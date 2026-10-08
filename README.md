# NOVA — Intelligent NLP Chatbot

**NOVA** is a production-quality chatbot built as a long-term learning project.
The architecture evolves incrementally through nine phases — from rule-based NLP
all the way to autonomous agents — **without hiding the NLP behind an LLM API**.

> Every component starts as a simple implementation you can read and understand,
> then gets replaced by a more advanced model behind the *same interface*.

**Current status: Phase 5 — Semantic Search** (Phases 1-4 complete)

---

## Why this project exists

| Goal | How |
| --- | --- |
| Learn NLP for real | Implement each concept by hand first (rules → TF-IDF → ML → embeddings → transformers → RAG → agents) |
| Be portfolio-quality | Clean layering, type hints, Pydantic contracts, tests, logging, docs |
| Not be an LLM wrapper | LLM integration only arrives in Phase 8, behind a `LLMProvider` abstraction |
| Stay extensible | Each NLP component is swappable behind a stable function signature |

Full phase breakdown: [docs/ROADMAP.md](docs/ROADMAP.md) ·
Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Learning docs: [docs/NLP_LEARNING.md](docs/NLP_LEARNING.md)

---

## Project structure

```
nova/
├── backend/                  FastAPI application
│   ├── app/
│   │   ├── main.py           App factory, CORS, error handlers, lifespan
│   │   ├── config.py         Typed settings from .env (pydantic-settings)
│   │   ├── api/              HTTP layer — routers only
│   │   │   ├── router.py     Single aggregation point for all routes
│   │   │   └── routes/       health.py, chat.py, ...
│   │   ├── core/             logging.py, exceptions.py
│   │   ├── models/           SQLAlchemy entities (Phase 6+)
│   │   ├── schemas/          Pydantic request/response contracts
│   │   ├── services/         Business logic / orchestration (Phase 2+)
│   │   ├── nlp/              NLP components (Phase 2+)
│   │   ├── memory/           Conversation memory (Phase 6)
│   │   └── rag/              Retrieval-augmented generation (Phase 7)
│   ├── tests/                pytest suite (API contract tests)
│   ├── requirements.txt      Each dependency documented with a reason
│   └── .env.example
├── frontend/                 React + Vite + Tailwind chat UI
│   ├── src/
│   │   ├── components/       ChatHeader, MessageList, MessageBubble, ChatInput, ...
│   │   ├── pages/            ChatPage
│   │   ├── hooks/            useChat (state machine), useHealth (status polling)
│   │   ├── services/         api.js — fetch wrapper, error normalization
│   │   └── App.jsx
│   ├── vite.config.js        Dev proxy: /api → localhost:8000
│   └── .env.example
├── data/                     intents/ (Phase 3), documents/ (Phase 7)
├── notebooks/                Educational experiments (start Phase 2)
├── docs/                     ARCHITECTURE, ROADMAP, NLP_LEARNING
└── README.md
```

---

## Getting started

### Prerequisites

- Python **3.11+** (tested on 3.13)
- Node.js **20.19+ / 22+ / 24** (tested on 24)
- Git

### 1. Install the backend

```powershell
cd nova\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env         # optional in Phase 1, needed later
```

### 2. Run the backend

```powershell
uvicorn app.main:app --reload
```

- API: <http://localhost:8000>
- Interactive docs (OpenAPI/Swagger): <http://localhost:8000/docs>

> **NLTK corpora:** downloaded automatically on first startup (~2 MB, logged),
> or manually via `python -m app.nlp.nltk_data`.

### 3. Install the frontend

```powershell
cd ..\frontend
npm install
Copy-Item .env.example .env         # optional; empty VITE_API_BASE_URL = use proxy
```

### 4. Run the frontend

```powershell
npm run dev
```

Open <http://localhost:5173>. The header status pill turns **Connected** once it
reaches the backend.

### 5. Run the tests

```powershell
cd ..\backend
.\.venv\Scripts\python.exe -m pytest
```

---

## API endpoints (Phase 1)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness probe — `{"status":"healthy","service":"nova"}` |
| `POST` | `/api/chat` | Send a message (Phase 1: validated stub; NLP from Phase 2) |
| `GET` | `/` | Service info |
| `GET` | `/docs` | Swagger UI |

### Example

```powershell
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"hello, how are you?"}'
```

```json
{
  "response": "Preprocessed your message into 4 tokens: cats, sleeping, visit, info.",
  "intent": "unknown",
  "confidence": 0.0,
  "entities": [],
  "processing": {
    "tokens": ["cats", "sleeping", "visit", "info"],
    "normalized_text": "the cats are sleeping! visit for more info.",
    "sentences": ["the cats are sleeping!", "visit for more info."]
  },
  "session_id": null
}
```

The **response shape is the full Phase 2+ contract** and will stay stable —
future phases fill `intent`, `entities` and `processing` with real NLP output
instead of changing the schema.

---

## Phase 1 engineering decisions (and why)

| Decision | Reason |
| --- | --- |
| `pydantic-settings` with `NOVA_` env prefix | Typed config, no secrets in source, no variable-name collisions |
| Contract-first `ChatResponse` schema | Frontend never needs rewiring when real NLP lands in Phases 2–4 |
| `/api` proxy in Vite | Dev requests are same-origin → no CORS setup in development |
| Route → service → NLP layering | Business logic stays testable without HTTP |
| `NovaError` exception handlers | Services raise typed errors; clients get JSON, logs get stack traces |
| Minimal Phase 1 dependency set | No SQLAlchemy/pgvector/ML libs until their phase needs them (see `requirements.txt` for per-dependency rationale) |
| `create_app()` factory | Tests get a clean app instance; future phases can inject config/providers |

---

## Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| **1** | Project foundation: FastAPI + React, health check, contracts | ✅ Done |
| **2** | NLP preprocessing pipeline (tokenize, normalize, stem, lemmatize) | ✅ Done |
| **3** | Intent classification: TF-IDF → classical ML, evaluation metrics | ✅ Done |
| **4** | Entity extraction: rules → spaCy NER behind one abstraction | ✅ Done |
| **5** | Semantic search with sentence embeddings | ✅ Done |
| 6 | Conversation memory (sessions in PostgreSQL) | ⬜ Next |
| 7 | RAG: upload documents, grounded answers with citations | ⬜ |
| 8 | LLM integration via `LLMProvider` abstraction | ⬜ |
| 9 | Tool calling + agent architecture | ⬜ |
| 10 | Production-grade frontend polish (streaming, markdown, themes) | ⬜ |

Details: [docs/ROADMAP.md](docs/ROADMAP.md)

---

## Engineering rules

1. Clean modular code — no giant files.
2. Type hints everywhere in Python; Pydantic at every boundary.
3. API / business logic / NLP / database / model code stay separated.
4. Secrets only via `.env` (gitignored) — `.env.example` documents the shape.
5. Proper exception handling + structured logging from day one.
6. Unit tests for NLP functions, API tests for endpoints.
7. Every new dependency must state *why* it is introduced.
8. No deprecated libraries when a maintained alternative exists.
9. **Learning mode:** every NLP concept gets documentation (what / why / math /
   implementation / limitations / how modern systems improve) — see
   [docs/NLP_LEARNING.md](docs/NLP_LEARNING.md).

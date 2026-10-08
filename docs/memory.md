# Conversation Memory (Phase 6)

> **Code:** `backend/app/memory/` (`base.py`, `context.py`, `db.py`) ·
> **Config:** `NOVA_DATABASE_URL`, `NOVA_MEMORY_WINDOW` ·
> **Tests:** `backend/tests/test_memory.py`

---

## 1. What is it?

Persistent per-session storage of the conversation's turns, plus a
**context window** — the slice of recent history injected into reply
generation. It is what lets a chatbot say *"Your name is Zephyr — you told
me earlier"* instead of pretending every message arrives in a vacuum.

## 2. Why it exists?

- **Continuity:** references ("what about tomorrow?", "you said…") require
  prior turns.
- **Identity:** user preferences/names declared once must survive to later
  turns — *from stored data, never hardcoded.*
- **Phase 7+ dependency:** RAG and the agent loop (Phases 7/9) both read
  conversation history; without a store they would re-invent one badly.

## 3. Architecture — one interface, two stores

```
ChatService ──► MemoryStore (ABC)
                  ├─ InMemoryStore      dict-backed, process-local
                  └─ SqlAlchemyMemoryStore ──► PostgreSQL (prod) / SQLite (tests)
```

The service **only** sees the interface (`get_or_create_conversation`,
`add_message`, `recent_messages`), so swapping stores is a constructor
argument. `get_memory_store()` picks from config:

| `NOVA_DATABASE_URL` | store |
|---|---|
| unset / empty | `InMemoryStore` |
| `postgresql+psycopg://…` | SQL store on Postgres |
| unreachable URL | logs error, **degrades to in-memory** (never 500s) |

### Schema

```sql
conversations(id PK, session_id UNIQUE, created_at, updated_at)
messages(id PK, conversation_id FK CASCADE, role, content,
         intent, confidence, created_at)
```

Schema is created with `create_all` on first use — honest for this dev-scale
project; a production deployment would manage the identical tables with
Alembic migrations.

### Local PostgreSQL setup (this machine)

```sql
-- as superuser
CREATE ROLE nova LOGIN;
CREATE DATABASE nova OWNER nova;
```

```ini
# backend/.env (gitignored)
NOVA_DATABASE_URL=postgresql+psycopg://nova@localhost:5432/nova
```

Localhost uses `trust` auth in `pg_hba.conf` on the dev machine (backup kept
as `pg_hba.conf.bak-nova`); with password auth, embed the password in the
URL or use `.pgpass`. Without any URL the app silently uses the in-memory
store — nothing breaks.

## 4. The request lifecycle

```
POST /api/chat  {message, session_id?}
  1. get_or_create_conversation(session_id)   # None -> server generates uuid
  2. recent_messages(session, window=12)      # history BEFORE this turn
  3. build_context(recent, message)           # known_name, turn_count
  4. NLP pipeline (preprocess/intent/entities)
  5. reply = memory-aware generation
  6. persist user turn + assistant turn
  7. respond with session_id (client reuses it)
```

Order matters: context is built **before** the current turn is written, so
`turn_count` reflects turns *seen*, not messages double-counted.

## 5. Name recall — worked example

Turn 1: `"My name is Zephyr"`
→ `extract_name()` matches `my name is ([A-Za-z…])`, reply
`"Nice to meet you, Zephyr!"`, both turns stored.

Turn 2: `"what is my name?"`
→ `build_context()` scans stored **user** messages newest-last for name
statements → `known_name = "Zephyr"` → reply
`"Your name is Zephyr — you told me earlier in our conversation."`

Nothing about Zephyr exists in source code — swap the name, the flow works
(proven by tests using `Zephyr`, `Echo`, `River`).

**Design details:**
- newest declaration wins (a user correcting their name is honored);
- predicate blacklist: `"I'm fine"` must not record *fine* as a name;
- question patterns: `what is my name`, `what's my name`, `who am I`,
  `do you remember/know my name`, `what do you call me`;
- without history the answer honestly says *"I don't know your name yet"*.

## 6. Session semantics

- `session_id` omitted → server **creates** one and returns it; the client
  (frontend `useChat` hook) sends it back on every later request.
- `clearChat` in the UI drops the id → the next message starts a *new*
  conversation (old one remains stored server-side).
- sessions are fully isolated — history never leaks across ids.
- `NOVA_MEMORY_WINDOW=0` disables context (messages still stored).

## 7. Trade-offs & limitations

| choice | why |
|---|---|
| window = 12 messages | enough for near-term references; bounded prompt size for Phase 8's LLM |
| sync SQLAlchemy | turns are single-digit-ms; async buys nothing until pooled load |
| name = regex rules | a *demonstrable* memory feature without a slot-filling framework; Phase 8's LLM can generalize it |
| no summarization | windows truncate instead of summarizing — fine at 12 turns, wrong at 100 (future work) |
| in-memory fallback | availability over durability: chat keeps working when the DB is down |

**Not yet:** cross-session profiles, message edit/delete, pagination,
retention policies — all Phase 10+/production concerns.

## 8. How modern systems improve it

- **Summarized memory:** periodically condense history into a running
  summary (token-efficient for LLM contexts).
- **Episodic + semantic split:** facts ("name = Zephyr") extracted into a
  knowledge store separate from raw transcript.
- **Vector recall:** embed turns, retrieve relevant ones by similarity
  instead of recency (we already have the machinery from Phase 5).
- **Ephemeral session state (WebSockets/DOs):** live state co-located with
  the connection for sub-ms reads.

---

*Previous: [05_embeddings.md](nlp/05_embeddings.md) · Next: [07_rag.md](07_rag.md)*

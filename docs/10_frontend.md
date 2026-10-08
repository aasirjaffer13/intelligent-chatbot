# Frontend Production Pass (Phase 10)

**Phase 10** · Code: `frontend/src/`, `app/api/routes/{chat,conversations,status}.py` · Tests: `tests/test_phase10.py`

---

## 1. Streaming chat (SSE) — honestly

Phase 10 adds `POST /api/chat/stream`. The pipeline (NLP → agent → RAG)
still generates the reply **atomically** — there is no token-level
generation to tap into — so the endpoint chunks the finished reply
word-by-word with a small delay (`NOVA_STREAM_DELAY_MS`, default 15 ms)
and delivers it over **Server-Sent Events**. The UI paints each chunk as
it lands, so the answer visibly "types itself in". This is a streaming
*experience* on top of a non-streaming generator; true token streaming
arrives the day a provider exposes incremental generation, and the event
contract below will not need to change.

### Event contract

```
event: delta   data: {"text": "Hello"}     ← 0..n reply chunks
event: meta    data: { ...ChatResponse }    ← exact /api/chat contract
event: end     data: {}                     ← stream finished
event: error   data: {"code": ..., "detail": ...}   ← pipeline failure
```

Two properties matter:

1. **`meta` carries the full ChatResponse** — intent, confidence,
   entities, processing info, sources, session id. The frontend swaps it
   in after the last `delta`, so streaming never costs the contract.
   Concatenating every `delta` yields *exactly* `meta.response`
   (`\S+\s*` chunking is lossless) — the test suite asserts this.
2. **Errors are events, not broken sockets.** A `NovaError` mid-stream
   becomes an `error` event; the browser shows the same banner as the
   JSON API.

### Why fetch, not `EventSource`

`EventSource` only speaks GET, and chat is a POST. The client
(`frontend/src/services/api.js → streamChat`) uses `fetch` +
`ReadableStream`, buffers bytes, splits on blank lines, and dispatches
`event:`/`data:` pairs. Same SSE wire format, any HTTP method.

## 2. Conversation sidebar (`/api/conversations`)

`MemoryStore` grew one abstract method — `list_conversations()` —
implemented for both backends:

| | In-memory | SQLAlchemy |
| --- | --- | --- |
| source | walk `_messages` dict | `ConversationRow.updated_at DESC` |
| preview | last message content | last message content |
| updated_at | last message time | conversation row timestamp |

* `GET /api/conversations` → `{conversations: [{session_id,
  message_count, preview, updated_at}]}`
* `GET /api/conversations/{id}/messages` → full chronological history;
  unknown id → `404` with the standard `{"error":{code,detail}}`
  envelope.

Clicking a conversation replays it into the pane (`useChat.
loadConversation`); "New chat" clears the pane and lets the backend mint
the next `session_id`. The sidebar is overlay + backdrop on mobile,
static panel ≥ `md`.

## 3. Markdown + code blocks

Assistant messages render through **react-markdown + remark-gfm**
(paragraphs, lists, tables, quotes, links, fences). Every element is
hand-styled with the project's Tailwind tokens — no `prose` plugin — so
text colors flow through the same `--color-*` variables as the rest of
the app. Fenced blocks get a language label and a **Copy** button
(`CodeBlock.jsx`); inline code is distinguished from block code by the
`language-*` class / newline heuristic (react-markdown no longer passes
an `inline` flag).

RAG answers show **source chips** under the bubble: file name, chunk
part, cosine score — pulled from the `sources[]` that Phase 7 already
returned. During streaming the bubble shows a blinking caret; before
the first chunk it shows the typing dots.

## 4. Dark/light with ONE class

Tailwind v4 utilities compile to `var(--color-*)`. So flipping
`document.documentElement.classList.toggle('light')` and redefining the
palette under `:root.light` in `index.css` re-themes **every existing
utility class** — no component carries a `dark:`/`light:` variant. The
override mirrors the slate ramp (950 → lightest paper, 100 → darkest
ink) and pins accent steps (indigo-400, rose-300, emerald-400 …) to
values that keep contrast on white. Theme persists in
`localStorage['nova-theme']`.

## 5. Status indicator (`GET /api/status`)

What the process is *actually running with*, section by section, each
degrading to an explicit value instead of failing the endpoint:

```json
{
  "llm":     {"provider": "none", "model": null},
  "agent":   {"max_steps": 6},
  "rag":     {"mode": "numpy", "documents": 0},
  "memory":  {"backend": "postgresql"},
  "intent":  {"backend": "auto"}
}
```

The header pill renders it as `openai · gpt-4o-mini` or
`Templates · no LLM`; a `title` tooltip shows memory/RAG/agent details.
Tests assert the shape and allowed values, never a specific provider —
and assert no `sk-`/`hf_` secret ever leaks into the payload.

## 6. State coverage

| State | Where |
| --- | --- |
| First-load skeleton | conversation list (`conversationsLoading`) |
| History loading | centered spinner (`isLoadingHistory`) |
| Empty conversation | suggestion chips (`MessageList` empty state) |
| Streaming | typing dots → caret → markdown swap |
| API error | dismissible banner + placeholder dropped/partial kept |
| Backend offline | header pill → Offline (health polling) |
| Upload in progress / failed | sidebar button spinner + inline error |

## 7. Verification

* `pytest` → **257 tests** (8 new: delta reassembly = contract, RAG
  sources over the stream, 422 validation, conversations round trip +
  404 envelope, status shape + no-secret-leak).
* `npm run build` → clean Vite production build.
* Live smoke test against uvicorn + PostgreSQL: stream (11 deltas,
  reassembled exactly), conversations list/history, status, 404
  envelope — all green.

---

*Previous: [09_agents.md](09_agents.md) · Back to [ROADMAP.md](ROADMAP.md)*

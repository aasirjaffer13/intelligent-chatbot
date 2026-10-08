# LLM Integration (Phase 8)

**Phase 8** · Code: `app/llm/` · Tests: `tests/test_llm.py`

---

## 1. The rule: the pipeline decides, the LLM phrases

Phases 2–7 made NOVA work *without* an LLM: preprocessing, intent,
entities, memory and RAG all run deterministically. Phase 8 adds a model
on top — strictly for **response generation over the pipeline's outputs**:

```
user message ──▶ preprocess ──▶ intent ──▶ entities ──▶ memory ──▶ RAG?      (unchanged)
                                     │           │          │        │
                                     ▼           ▼          ▼        ▼
                              structured prompt  ─────────────────────▶  LLM  ──▶ reply
                                                                        │
                                                          any failure ──┴──▶ template fallback
```

Deterministic answers **never** go to a model: clock answers
(`_time_reply`), name recall (real stored history), and document quotes
(Phase 7's verbatim grounding). A model that "feels" like 3pm is exactly
the failure mode this architecture exists to prevent.

## 2. Provider abstraction

One interface, five implementations (`app/llm/base.py`,
`app/llm/providers.py`):

| Provider | Transport | Credentials | Selected when |
| --- | --- | --- | --- |
| `OpenAIProvider` | httpx → `/chat/completions` | `OPENAI_API_KEY` | `auto` + key present, or `NOVA_LLM_PROVIDER=openai` |
| `HuggingFaceProvider` | httpx → Inference API | `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | `auto` + token, no OpenAI key |
| `LocalModelProvider` | `transformers` pipeline (lazy load) | none | `NOVA_LLM_PROVIDER=local` |
| `MockProvider` | none — records requests | none | `NOVA_LLM_PROVIDER=mock` / tests |
| *(none)* | — | — | `auto` without keys (default here) |

Contract:

```python
class LLMProvider(ABC):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

- `LLMRequest(prompt, system, max_tokens, temperature)` — plain data.
- `LLMResponse(text, provider, model, latency_ms)` — plain data.
- **Failures raise `LLMError`** — auth, network, bad payload, empty text.
  `ChatService._llm_reply` catches it, logs one line, and returns `None`
  so the template path answers. A provider outage degrades fluency, never
  availability.

HTTP providers accept an injectable `httpx` transport, so every provider
test runs offline against `httpx.MockTransport` — no keys, no network.

## 3. Prompt construction (`app/llm/prompting.py`)

The prompt is a **structured dump of pipeline decisions** — the model is
told what the system already determined, not asked to figure it out:

```
System:  You are NOVA ... Use ONLY the facts supplied ... never invent
         names, times, dates or document contents ...

User:    Detected intent: reminder (confidence 0.87)
         Entities: TIME=3pm
         User's name (from memory): Aasir
         Recent conversation:
           user: ...
           NOVA: ...
         User message: remind me at 3pm
         Reply as NOVA.
```

Why this shape:

- **System prompt carries the rules** (1–3 sentences, no invented facts,
  ask to rephrase on `unknown` intent) — persona and guardrails in one
  place, shared by every provider.
- **Deterministic assembly** — same inputs produce the same prompt, so
  tests can assert on exact contents.
- **Bounded history** — last 8 messages only; prompt size stays constant
  even when `NOVA_MEMORY_WINDOW` grows.
- **Entities/intent as text** — works with any tokenizer; no provider
  -specific structured-input formats.

## 4. Configuration & secrets

| Env var | Default | Meaning |
| --- | --- | --- |
| `NOVA_LLM_PROVIDER` | `auto` | `auto \| openai \| huggingface \| local \| mock \| none` |
| `NOVA_LLM_MODEL` | *(provider default)* | model override (`gpt-4o-mini`, `distilgpt2`, …) |
| `NOVA_LLM_MAX_TOKENS` | `256` | completion budget |
| `NOVA_LLM_TEMPERATURE` | `0.2` | sampling (low = consistent replies) |
| `NOVA_LLM_TIMEOUT` | `30` | HTTP timeout, seconds |
| `OPENAI_API_KEY` | — | secret, read from environment only |
| `HF_TOKEN` | — | secret (`HUGGINGFACEHUB_API_TOKEN` also accepted) |

Secrets are `SecretStr` in settings: they never appear in `repr()`,
logs, or docs, and `.env.example` documents only their names. The
factory is `lru_cache`d — the key is read once per process.

**Auto-selection order:** OpenAI → HuggingFace → *no provider*
(templates). `local` is never auto-selected because it may download
hundreds of MBs on first use — you opt in explicitly.

## 5. Adding a provider (the point of the abstraction)

1. Subclass `LLMProvider` in `app/llm/providers.py`.
2. Implement `complete()` — raise `LLMError` on anything unexpected,
   return an `LLMResponse` with trimmed text.
3. Add its name to the validator in `config.py` and a branch in
   `get_llm_provider()`.
4. Add a `MockTransport`-style test.

`chat_service.py`, the API contract and every other provider stay
untouched — that is the abstraction earning its keep.

## 6. Verifying

```powershell
cd nova\backend
.\.venv\Scripts\python.exe -m pytest tests/test_llm.py -q   # 35 tests, offline
```

Covers: prompt contents/caps, auth headers and payload shapes for both
HTTP providers, error paths (401/503/malformed/empty/network), factory
auto-selection with scrubbed environment, secrets never in `repr`, and
chat integration proving LLM → fallback → routing priority (time, names
and RAG never reach the model).

## 7. Limitations — and where this goes

- **No streaming** — replies arrive whole; token streaming lands in the
  Phase 10 frontend pass (SSE), with the provider contract gaining an
  iterator method then.
- **No prompt-injection defence yet** — history is passed as text; a
  hardened system would delimit untrusted content and sanitize tool
  outputs (Phase 9's agent loop makes this mandatory).
- **Temperature 0.2, not 0** — providers differ in how they treat 0;
  low-but-nonzero is the portable "mostly deterministic" choice.
- **One completion per turn** — no retries or fallback chains; a second
  provider as standby is a config-level extension of `get_llm_provider()`.
- **Cost/latency are per-turn** — every non-deterministic reply bills a
  call; the pipeline ordering (cheap NLP first, model last) exists partly
  for this reason.

---

*Previous: [07_rag.md](07_rag.md) · Next: tools + agent (Phase 9)*

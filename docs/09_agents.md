# Tools + Agent Architecture (Phase 9)

**Phase 9** · Code: `app/tools/`, `app/agent/` · Tests: `tests/test_agent.py`

---

## 1. What an agent is here

Phases 2–8 produced a *pipeline*: fixed sequence, fixed decisions.
An **agent** adds a loop in which the model decides, at each step,
whether to (a) call an external **tool** or (b) answer:

```
user message ─▶ LLM ─┬─ plain text ────────────────────────────▶ final reply
                     │
                     └─ {"tool": ..., "args": {...}}
                              │
                              ▼
                        tool executes ──▶ observation (text)
                              │
                              └──────────▶ back to LLM (with observation)
```

This is the **ReAct** pattern (Reason + Act): reasoning is the model's
text, acting is a tool call, and the next reasoning step sees the result.
NOVA's loop (`app/agent/loop.py`) is bounded by
`NOVA_AGENT_MAX_STEPS` (default 6) — on exhaustion it returns an honest
step-limit message instead of spinning.

## 2. The tool contract (`app/tools/base.py`)

```python
class Tool(ABC):
    name: str                  # what the model calls
    description: str           # one line, rendered into the prompt
    parameters: dict[str, str] # param -> explanation for the model
    def run(self, args: dict) -> ToolResult: ...
```

Two design rules:

1. **Errors are data.** A tool returns `ToolResult(ok=False,
   output="reason")` for expected failures (bad input, unknown place,
   network down). The loop feeds that reason back to the model as an
   *observation* — the conversation continues. Uncaught exceptions are
   caught by the loop too, so no tool can crash a request.
2. **Modularity by registration.** The registry
   (`app/tools/registry.py`) is the single source of truth: `register()`
   a class and it appears in the next prompt automatically;
   `describe()` renders signatures + descriptions. Duplicate names
   raise — silent shadowing is how agent bugs start.

### Shipped tools

| Tool | Backs onto | Failure behaviour |
| --- | --- | --- |
| `calculator(expr)` | AST whitelist — **never `eval`** | parse/range/div-zero → readable error |
| `current_time()` | system clock | deterministic |
| `document_search(q)` | RAG store + embeddings | no docs / no match / store down → message |
| `weather(loc)` | Open-Meteo (keyless, 2 HTTP calls) | unknown place, network → `ok=False` |
| `web_search(q)` | DuckDuckGo instant answers | no answer → "no instant answer" |

Calculator is the safety exemplar: only numbers, operators and
parentheses survive the AST walk — no names, imports, or attribute
access, with size/power guards against runaway arithmetic. The two HTTP
tools take injectable transports so their tests never touch the network.

## 3. The loop (`app/agent/loop.py`)

```python
for step in range(max_steps):
    text = provider.complete(request)          # same LLMProvider as Phase 8
    call = parse_decision(text)                # JSON with a "tool" key?
    if call is None: return final_answer(text)
    observation = execute(call)                # unknown/crash -> error text
    prompt += f"Tool call: {call}\nObservation: {observation}\nContinue."
return STEPS_EXCEEDED
```

- **Forgiving parsing**: anything that isn't a JSON object with a
  `"tool"` string is treated as the final answer — a model that ignores
  the format still produces a reply (fenced ```json blocks handled).
- **Observations are append-only**: the model always sees the original
  prompt, the tool list, and every prior call/result pair — no hidden
  state, fully loggable (`AgentResult.steps`).
- **Prompt wiring**: the tools block is appended to Phase 8's
  `build_chat_request()` output — intent, entities and memory still come
  from the pipeline; the system prompt is unchanged.

### Where the pipeline still wins

Routing happens **before** the loop and never inside it:

| Case | Route | Why |
| --- | --- | --- |
| `document_question` | Phase 7 RAG (verbatim quotes) | document facts must be grounded |
| name questions / declarations | memory (Phase 6) | facts come from stored history |
| `time` intent | deterministic clock | a model must never guess the time |
| everything else | **agent loop** (or template fallback) | phrasing + tool decisions |

If there is no LLM provider (no keys — the default), `_llm_reply`
returns `None` and the deterministic templates answer, exactly as in
Phases 1–7. The agent is an *addition*, never a dependency.

## 4. When agents help — and when they hurt

**Help:**

- **Multi-step questions** — "will it rain in Nairobi after my meeting?"
  needs geocode → forecast → compose; one completion can't.
- **External data** — anything with a timestamp or database behind it
  (weather, document search, later: APIs).
- **Ambiguous requests** — the model can *choose* the right tool from
  descriptions instead of the pipeline pre-routing every phrasing.
- **Composition** — tool output becomes input to the next reasoning
  step (ReAct) instead of a hand-built template branch.

**Hurt:**

- **Simple deterministic facts** — time, arithmetic-only answers, stored
  names: a loop adds latency and cost for zero gain (this is why those
  routes bypass it).
- **Latency & cost** — each step is a full model round-trip; 6 steps ×
  800 ms feels slow for "hello".
- **Reliability surface** — more steps = more chances to misuse a tool;
  mitigated by bounded steps, forgiving parsing, and honest refusals.
- **High-stakes domains** — an agent that *acts* (sends mail, pays
  invoices) needs permissions, confirmations and audit logs; NOVA's
  tools are read-only by design.

## 5. Configuration & extending

| Env var | Default | Meaning |
| --- | --- | --- |
| `NOVA_AGENT_MAX_STEPS` | `6` | LLM/tool rounds per turn (1–20) |
| `NOVA_LLM_PROVIDER` | `auto` | without credentials the agent never engages |

Add a tool in three steps:

```python
class MyTool(Tool):
    name, description, parameters = "my_tool", "does a thing", {"x": "input"}
    def run(self, args: dict) -> ToolResult:
        ...  # return ok=False + reason on expected failures

registry.register(MyTool())       # next prompt includes it — nothing else changes
```

## 6. Verifying

```powershell
cd nova\backend
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py -q   # 49 tests, offline
```

Covers: calculator whitelist (`__import__` rejected, pow/size guards),
registry semantics, weather/web-search over `MockTransport`, decision
parsing (plain/fenced/malformed), the loop (tool→observation→final,
unknown tool, crashing tool, max-steps rail, provider failure), and chat
integration including deterministic bypass and template fallback.

## 7. Limitations — and where this goes

- **Sequential only** — one tool call per step; parallel calls and
  streaming steps are natural extensions of `AgentLoop.run`.
- **No memory of past steps across turns** — each turn rebuilds from the
  memory window; long-horizon agents would persist step traces.
- **Prompt injection** — tool observations are model-visible text; a
  hardened design delimits untrusted content (especially for
  `web_search` results) and validates arguments before execution.
- **No permissions/confirmation** — fine for read-only tools; any
  write-capable tool must gain an approval gate first.
- **Single agent** — no planner/worker split; multi-agent designs pay
  off only when tasks are genuinely decomposable.

---

*Previous: [08_llm_integration.md](08_llm_integration.md) · Next: frontend production pass (Phase 10)*

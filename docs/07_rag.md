# Retrieval-Augmented Generation (RAG)

**Phase 7** · Code: `app/rag/` · Tests: `tests/test_rag.py` ·
Notebook: [`07_rag.ipynb`](../notebooks/07_rag.ipynb)

---

## 1. What RAG is — and why

A chatbot's *knowledge* can come from three places:

| Source | Problem |
| --- | --- |
| Hardcoded rules | Doesn't scale past a FAQ |
| Fine-tuning | Expensive, stale the moment the document changes, no citations |
| **Retrieval (RAG)** | Answer is looked up at query time, grounded in a source you can cite |

**Retrieval-Augmented Generation**: before answering, retrieve the relevant
passages from an external knowledge store, then generate the answer **from
those passages only**. In NOVA, "generation" is deliberately extractive
(we quote the passage verbatim); Phase 8 may layer an LLM to *phrase*
answers, but the evidence stays a verbatim quote.

The pipeline:

```
upload ─▶ extract text ─▶ chunk ─▶ embed ─▶ store (vectors + text)
question ─▶ embed ─▶ top-k similarity search ─▶ score gate ─▶ quoted answer + sources
                                                          └─ or honest refusal
```

## 2. Chunking

Embedding models have a practical token limit, and retrieval granularity
must be *smaller than a document*: you want the paragraph that answers the
question, not the whole PDF. Chunks become the unit of embedding,
retrieval, and citation.

NOVA uses a **sliding word window with overlap**
(`app/rag/chunker.py`):

- `chunk_words = 80`, `overlap_words = 15` (≈ step of 65 words)
- **Why overlap?** An answer can straddle a boundary. Without overlap the
  sentence that answers the question gets cut in half, and neither half
  retrieves well. ~20% overlap is a common default.
- Offsets are exact: `text[chunk.start:chunk.end] == chunk.text`, so
  citations can point back into the original document.

Other strategies (trade-offs worth knowing):

| Strategy | Idea | Good when |
| --- | --- | --- |
| Sentence window | each sentence ± neighbours | Q&A over well-formed prose |
| Recursive character split | split on `\n\n`, then `\n`, then ` ` | generic text, LangChain-style |
| Structure-aware | markdown headers, PDF sections | docs with headings |
| Semantic chunking | split where embedding similarity drops | heterogeneous documents |
| Parent/child chunks | retrieve small, return the larger parent | precision + context balance |

## 3. Embeddings → similarity (the math)

Each chunk is embedded with `all-MiniLM-L6-v2` (384-d, unit-normalized —
see [Phase 5](nlp/05_embeddings.md)). Cosine similarity between query *q*
and chunk *c*:

```
cos(q, c) = (q · c) / (‖q‖ ‖c‖)
```

Because both vectors are unit-length (`‖q‖ = ‖c‖ = 1`), this simplifies to
the **dot product** — which is exactly what the numpy search path computes:

```python
scores = embeddings @ query          # (n_chunks, 384) @ (384,) -> (n_chunks,)
```

## 4. Vector storage: pgvector, with a portable fallback

`app/rag/store.py` picks one of two modes at startup — **same schema,
same API, zero code change to switch**:

| Mode | When | Search |
| --- | --- | --- |
| **pgvector** | `CREATE EXTENSION vector` succeeds | inside Postgres: `ORDER BY embedding_vec <=> $q`, HNSW/IVF indexes available |
| **numpy fallback** | extension missing (or SQLite) | load `bytea` embeddings, dot product in-process |

- **pgvector** keeps relational data and vectors in one engine, supports
  `<=>` (cosine distance), and indexes (`hnsw`, `ivfflat`) for ANN search
  at scale. This is the production shape.
- The **fallback** is honest engineering, not a hack: 50k chunks × 384 ×
  4 B ≈ 77 MB in RAM, brute-force dot products in <10 ms — fine for a
  portfolio-scale corpus, and it lets the whole test suite run on SQLite.

> **This machine:** pgvector is not installed (no VS Build Tools, no
> Docker), so PostgreSQL runs in numpy mode — verified end-to-end
> (ingest → search → grounded answer → delete, zero orphan rows).
> To enable pgvector on Windows: install VS Build Tools, then
> `make install` from [pgvector/PG18](https://github.com/pgvector/pgvector)
> against your PostgreSQL 18, restart the service, and NOVA switches modes
> automatically on next start.

Other options in the ecosystem: **FAISS** (in-process ANN from Meta),
**Chroma / LanceDB** (embedded vector DBs), **Pinecone / Weaviate /
Qdrant / Milvus** (managed/scale). NOVA stays on SQL because Phase 7's
corpus is small and the design leaves the door open.

## 5. Grounding and refusal (no hallucinated document facts)

The strict contract in `app/rag/answerer.py`:

1. **No documents uploaded** → explain that and ask for an upload
   (`NO_DOCUMENTS_REPLY`), never a fabricated answer.
2. **Documents exist but the best score < `NOVA_RAG_MIN_SCORE`
   (default 0.35)** → explicit refusal: *"I couldn't find that in the
   uploaded documents, so I won't guess."* — same philosophy as intent's
   `unknown` fallback.
3. **Score clears the gate** → answer = `According to {file} (part n):`
   plus the **verbatim sentence** most overlapping the query
   (extractive — it must appear in the document to be quotable), with up
   to 3 `sources[]` (`document_id`, `filename`, `chunk_index`, `score`,
   `quote`).

Routing: the `document_question` intent (Phase 3) sends a message down the
RAG path; everything else stays in the conversational pipeline. Document
knowledge and general conversation remain strictly separated.

## 6. API + configuration

| Method | Path | Behaviour |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | multipart file → 201 with metadata · 415 unsupported type · 400 empty · 413 too large · 503 embedding unavailable |
| `GET` | `/api/documents` | newest-first list |
| `DELETE` | `/api/documents/{id}` | removes document, chunks (FK cascade) and the stored file → 204 · 404 unknown id |

| Env var | Default | Meaning |
| --- | --- | --- |
| `NOVA_RAG_TOP_K` | `4` | chunks retrieved per question |
| `NOVA_RAG_MIN_SCORE` | `0.35` | cosine gate below which we refuse |
| `NOVA_MAX_UPLOAD_MB` | `10` | upload size cap (1–100) |
| `NOVA_DOCUMENT_DIR` | `data/documents/` | where uploaded files are stored |

Accepted types: `.pdf` (pypdf text extraction), `.txt`, `.md` (UTF-8
decode with latin-1 fallback; a byte-heuristic rejects obvious binaries).

## 7. Verifying

```powershell
cd nova\backend
.\.venv\Scripts\python.exe -m pytest tests/test_rag.py -q
```

Covers: chunk offsets/overlap, extraction (incl. a programmatically-built
PDF fixture), store ranking, refusal paths, API status codes, and an
end-to-end chat question answered with citations.

## 8. Limitations — and how modern systems improve

- **Lexical sentence picker** picks by word overlap; a real system uses a
  cross-encoder **reranker** over the top-k (better precision).
- **Dense-only retrieval** misses exact keywords (error codes, SKUs);
  **hybrid search** (BM25 + dense, fused with RRF) fixes that.
- **Single-shot retrieval** struggles with multi-hop questions
  ("compare X in doc A with Y in doc B"); agentic retrieval
  (loop: retrieve → reason → retrieve) is Phase 9 territory.
- **Fixed window size** is one-size-fits-all; structure-aware or semantic
  chunking adapts to document shape.
- Scores are cosine over one embedding model — evaluation on a labelled
  query→chunk set (recall@k) is what production teams actually track.

---

*Previous: [memory.md](memory.md) · Next: LLM integration (Phase 8)*

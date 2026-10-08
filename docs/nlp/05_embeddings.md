# NLP Concept: Sentence Embeddings & Semantic Search

> **Project phase:** 5 · **Code:** `backend/app/services/embedding_service.py`,
> `similarity_service.py`, `app/nlp/intent.py` (`EmbeddingIntentClassifier`) ·
> **Notebook:** `05_embeddings.ipynb`

---

## 1. What is it?

An **embedding** maps text into a fixed-length vector in a space where
*distance means semantic difference*: paraphrases land near each other,
unrelated topics stay far apart — **even with zero shared words**.

```
"I forgot my password"          ─┐
                                  ├── cosine ≈ 0.79   (same meaning)
"I can't remember my login
 password"                      ─┘

"I forgot my password"          ─┐
                                  ├── cosine ≈ 0.01   (different meaning)
"what is the weather like"      ─┘
```

TF-IDF (Phase 3) could never do this: those paraphrases share almost no
*strings*. Embeddings compare *meaning*.

## 2. Why it exists?

- **Vocabulary mismatch is the #1 failure of keyword search** — "fluffy
  kittens" vs "cute cats", "I'm locked out" vs "password reset".
- **RAG needs it (Phase 7):** find document chunks *about* the question,
  not chunks containing the same words.
- **Cost:** one ~80 MB model, milliseconds on CPU, no API key — versus an
  LLM call for every comparison.

## 3. How it works

### The model

`all-MiniLM-L6-v2` (sentence-transformers): a compact transformer trained on
~1B sentence pairs with contrastive learning — pairs that mean the same thing
are pulled together, unrelated pairs pushed apart. Output: **384-dimension
vector per sentence**, L2-normalized (unit length).

### The math (same cosine as Phase 3, different vectors)

```
cosine(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

Because we normalize at encode time, **cosine = plain dot product** — the
fastest similarity there is. Scores: `1.0` identical direction, `0.0`
orthogonal, negative = opposing (we clip confidence to [0, 1]).

### Pipelines

```
EmbeddingService.encode(texts)      -> unit vectors (batched, cached model)
SimilarityService.rank(query, cands) -> candidates best-first with scores
EmbeddingIntentClassifier.predict    -> nearest pattern's label (or unknown)
```

The intent classifier embeds all 489 training patterns **once** (lazily,
cached), then each query is one 384-dim dot-product sweep — microseconds.

## 4. Thresholds

Embedding cosines cluster differently from TF-IDF scores:

| pair type | typical cosine |
|---|---|
| exact/near-duplicate | 0.9 – 1.0 |
| genuine paraphrase | 0.5 – 0.8 |
| same broad topic | 0.2 – 0.5 |
| unrelated | 0.0 – 0.1 |

So `EMBEDDING_UNKNOWN_THRESHOLD = 0.45` (vs 0.35 for TF-IDF) — below it,
answer `unknown` rather than force a label.

## 5. How we implemented it

- **`embedding_service.py`** — lazy singleton, batch encode, normalized
  vectors, empty-input handling; raises if the model can't load.
- **`similarity_service.py`** — `cosine`, `cosine_similarity_matrix`,
  `SimilarityService.rank(query, candidates, top_k)` with injected embedder
  (tests pass fakes; production passes the singleton).
- **`intent.py`** — `EmbeddingIntentClassifier` + backend selection:
  `NOVA_INTENT_BACKEND=auto|sklearn|embedding|keyword` (default `auto` =
  sklearn-first; embedding is opt-in because a 32s model warm-up and
  per-request inference aren't free).

**Measured results (dataset v1.3):**

```
rank("I forgot my password"):
  0.788  i cannot remember my login password
  0.618  reset my login credentials
  0.063  the stock market rose today
  0.001  what is the weather like

embeddings backend: all 12 spec examples classified correctly;
gibberish -> unknown (0.59 or lower), paraphrase -> password_help (0.90)
```

## 6. TF-IDF vs embeddings — when to use which

| | TF-IDF | Embeddings |
|---|---|---|
| signal | shared *strings* | shared *meaning* |
| cost | µs, no model | ms + ~80 MB model |
| unseen paraphrase | fails | works |
| rare keywords (names, codes) | excellent | weaker |
| needs | vectorizer artifact | model file/network |

Our architecture deliberately keeps **both**: TF-IDF as the cheap default
router, embeddings for semantic tasks (paraphrase retrieval, RAG search).

## 7. Limitations

1. **No word-order sensitivity beyond training** — MiniLM sees local context
   but isn't a reasoning engine; negation sometimes slips ("I like it" vs
   "I don't like it" ≈ 0.8+).
2. **Domain mismatch** — general-purpose training; medical/legal text needs
   domain-tuned models for tight thresholds.
3. **Fixed granularity** — sentence-level by default; long documents need
   chunking (Phase 7 does exactly this).
4. **First-load cost** — model download (~80 MB, first run only) + ~2–4 s
   load + 30 s to embed the full pattern set; all cached per process.
5. **Cosine ≠ truth** — a score of 0.6 means "similar to", not "entails";
   RAG systems need re-ranking/verification for faithfulness (Phase 7).

## 8. How modern systems improve it

- **Larger/better models:** `bge-large`, `e5-mistral`, GTE — +5–10 pts on
  retrieval benchmarks at 5–50× the size.
- **Matryoshka embeddings:** train 1024-dim, truncate to 128 with minimal
  loss — smaller indexes.
- **Late interaction (ColBERT):** token-level matching, near-transformer
  quality at search-engine speed.
- **LLM re-rankers:** embed for recall → cross-encoder/LLM for precision
  (the classic two-stage retrieval pattern; Phase 7 can adopt it).
- **Hybrid search:** dense + sparse (TF-IDF/BM25) fused — each covers the
  other's blind spots. *We already have both halves of this.*

---

*Previous: [04_entity_extraction.md](04_entity_extraction.md) · Next: memory (Phase 6)*

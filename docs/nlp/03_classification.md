# NLP Concept: Intent Classification (TF-IDF → Classical ML)

> **Project phase:** 3 · **Code:** `backend/app/nlp/intent.py`, `train_intent_model.py` ·
> **Dataset:** `data/intents/intents.json` (489 utterances, 12 intents) ·
> **Report:** `docs/reports/intent_model.md` · **Notebooks:** `02_tf_idf.ipynb`, `03_intent_classification.ipynb`

---

## 1. What is it?

Intent classification maps a user message to a **discrete category of
communicative purpose** — greeting, request for help, password problem,
document question — which the system then uses to choose a response strategy.
It is text classification applied to the first decision a chatbot makes.

## 2. Why it exists?

A chatbot must decide *what the user wants* before deciding *what to say*.
Intents turn open-ended language into a finite routing problem: 12 classes
instead of ∞ possible replies. Deterministic, testable, and cheap — no GPU, no
API key, answer in <1 ms.

## 3. How it works

### Step 1 — Vectorize with TF-IDF

A document becomes a vector over the vocabulary `V`:

**Term frequency** (how often term `t` appears in document `d`):

```
TF(t, d) = count(t in d) / |d|          (we use sublinear: 1 + log(count))
```

**Inverse document frequency** (how rare `t` is across all documents `D`):

```
IDF(t) = log( N / df(t) )        N = |D|,  df(t) = #docs containing t
```

**TF-IDF** = product of the two:

```
TF-IDF(t, d, D) = TF(t, d) · IDF(t)
```

*Intuition:* a word scores high when it is frequent **in this message** but
rare **across the corpus** — exactly "informative about this message's topic".
`the` is frequent everywhere → near-zero IDF; `password_reset` is rare → high IDF.

We use `sublinear_tf` (`1 + log(tf)`) because raw counts overvalue repetition,
and `ngram_range=(1,2)` so discriminative phrases like `what_s up` or
`log in` become single features.

### Step 2 — Similarity or model

**Baseline (TF-IDF + cosine):** vectorize every training pattern; for a query,
compute cosine similarity against all patterns; the winner's label wins:

```
cosine(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

Cosine = **angle** between vectors, ignoring magnitude (we L2-normalize, so
`cos = dot product`). `1.0` = same direction (semantically aligned bag-of-words),
`0.0` = orthogonal (no shared terms). Confidence = best cosine score; below
`unknown_threshold` (0.35) → `unknown`.

**Classical models** learn *decision boundaries* instead of nearest neighbors:

| Model | Idea | Behavior |
|---|---|---|
| Logistic Regression | `P(y\|x) = softmax(Wx)` trained by cross-entropy | strong, calibrated linear baseline |
| Multinomial Naive Bayes | `argmax_y Π P(xᵢ\|y)·P(y)` (word independence) | fast, works with tiny data |
| Linear SVM | max-margin hyperplane between classes | often best on sparse text |

### Step 3 — Low-confidence fallback

Whatever the model, `confidence < threshold → unknown` with a template that
*admits* low confidence — never a fabricated answer.

## 4. Evaluation (why macro F1, not accuracy)

- **Accuracy** = correct / total. Misleading when classes differ in size
  (goodbye has 45 samples, capabilities 28).
- **Precision** = TP / (TP+FP) — *"when we say X, how often are we right?"*
- **Recall** = TP / (TP+FN) — *"of all real X, how many did we catch?"*
- **F1** = harmonic mean `2·P·R / (P+R)` — punishes one-sided classifiers.
- **Macro F1** = unweighted mean of per-class F1 — treats all 12 intents as
  equally important. **This is our selection metric.**
- **Confusion matrix** — rows = actual, columns = predicted; the diagonal is
  success, everything off-diagonal is a named failure mode.

**Evaluation methodology:** with 489 samples, a single train/test split swings
±0.08 macro F1 between runs. We therefore use **stratified 5-fold
cross-validation** with the vectorizer refit inside each fold (via `Pipeline`,
no leakage): every utterance is scored exactly once out-of-fold.

## 5. How we implemented it

- `data/intents/intents.json` — 12 intents × 28–51 natural patterns + response
  templates (dataset v1.3, hand-written, includes typos and colloquialisms).
- `train_intent_model.py` — CV over 4 models, writes `docs/reports/intent_model.md`,
  refits the winner on all data, saves `vectorizer.joblib` + `model.joblib` +
  `meta.json` to `backend/artifacts/intent/`.
- `intent.py` — `classify_intent()` lazily loads artifacts; **fallbacks**:
  TF-IDF baseline (no artifacts needed) → keyword rules (no dataset needed) →
  `unknown` on error. The API never crashes, never retrains at startup.

**Results (5-fold CV, dataset v1.3):**

| Model | Accuracy | Macro F1 (mean ± std) |
|---|---|---|
| **Linear SVM (calibrated)** ← selected | 0.748 | **0.757 ± 0.027** |
| Logistic Regression | 0.728 | 0.737 ± 0.034 |
| Multinomial NB | 0.730 | 0.732 ± 0.026 |
| TF-IDF + cosine baseline | 0.712 | 0.721 ± 0.031 |

**Two real bugs this evaluation caught** (worth remembering):
1. `class_weight="balanced"` on LR *hurt* precision — capabilities swallowed
   goodbye/identity messages (precision 0.24).
2. `stop_words="english"` stripped `who/what/where/how/you` — "who are you"
   vectorized to `[]`. **For intent classification, interrogatives are signal.**
   Removing that stopword list raised macro F1 from 0.65 → 0.76.

## 6. Example

```
"hey, what's up?"            → greeting       (0.83)
"I can't remember my login
 password"                   → password_help  (0.90)
"purple monkey dishwasher"    → unknown        (low-confidence fallback)
```

## 7. Limitations

1. **Bag-of-words blindness:** word order lost — "can you log me out" vs
   "log me in please" share features.
2. **Closed world:** only 12 intents; anything else → `unknown`.
3. **Inherent ambiguity:** "how are you" is legitimately greeting *or*
   small_talk (0.38 vs 0.40) — humans disagree too.
4. **Dataset scale:** 489 utterances ≈ toy scale; production systems want
   thousands per class with real user logs.
5. **No context:** "what about tomorrow?" alone has no intent signal.

## 8. How modern systems improve it

- **Embedding + similarity (our Phase 5):** semantic vectors handle unseen
  paraphrases TF-IDF cannot ("I can't get into my account" ≈ password_help
  with zero shared keywords).
- **Fine-tuned transformers:** BERT-style classifiers reach ~95%+ on intent
  benchmarks by understanding word order/context.
- **LLM zero-shot:** no dataset needed — but slower, costlier, less predictable
  (we deliberately keep classical NLP for routing, per the project's design).
- **Hybrid (our architecture):** TF-IDF/ML intent → route → only invoke
  expensive machinery (RAG, LLM, tools) when the router says it's needed.

---

*Previous: [02_tokenization.md](02_tokenization.md) · Next: NER (Phase 4)*

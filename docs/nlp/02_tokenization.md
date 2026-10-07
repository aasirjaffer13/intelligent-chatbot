# NLP Concept: Tokenization

> **Project phase:** 2 · **Code:** `backend/app/nlp/tokenizer.py` ·
> **Tests:** `backend/tests/test_tokenizer.py` · **Notebook:** `notebooks/01_text_preprocessing.ipynb`

---

## 1. What is it?

Tokenization splits a raw string into **tokens** — the atomic units every
downstream component operates on. Two granularities:

- **Sentence tokenization:** text → sentences (context/unit boundaries).
- **Word tokenization:** text → words & punctuation marks (the vocabulary units).

There is no canonical answer to "what is a word?" — tokenization is a
*decision* about the unit of analysis, and that decision shapes everything
after it (vocabulary size, model sequence length, what "same word" means).

## 2. Why it exists

A model cannot run regex over a paragraph. Every NLP method — counting,
TF-IDF, embeddings, transformers — first needs a sequence of discrete units:

```
"Hello, world!"   →   ["Hello", ",", "world", "!"]
```

Get tokenization wrong and errors cascade: `"don't"` counted as one token in
one place and two in another; `"can't"` vs `"cant"` never matching;
`"New-York"` and `"New York"` as different things.

## 3. How it works

### 3.1 The naive approach (we ship it as `naive_word_tokenize`)

```python
re.findall(r"[A-Za-z0-9']+", text.lower())
```

Three lines, fully understandable — and instructively broken:

| Input | naive | what went wrong |
| --- | --- | --- |
| `Hello, world!` | `['hello', 'world']` | punctuation silently vanished — you lose the *question vs statement* signal |
| `I don't know` | `['i', "don't", 'know']` | contraction never split → `don't` won't match `do` + `not` |
| `see www.x.com` | `['see', 'www', 'x', 'com']` | URL shredded into junk tokens |
| `猫 is cute` | `['is', 'cute']` | any non-Latin script deleted entirely |

### 3.2 NLTK's rule-based tokenizers (what we actually use)

**Word level — `TreebankWordTokenizer`** (Penn Treebank rules): a hand-written
state machine encoding editorial conventions:

- `don't` → `do` + `n't`, `can't` → `ca` + `n't` (split *irregular* contractions)
- `3.14` stays whole, `U.S.A.` stays whole, quotes/brackets handled
- `word_tokenize(text, preserve_line=True)` skips sentence splitting → **zero
  model data needed** (it's pure rules, which is why our word tokenizer works
  offline even before corpora are downloaded)

**Sentence level — PunktSentenceTokenizer:** an *unsupervised* model trained on
co-occurrence statistics of punctuation — it learned which `.`/`?`/`!`
abbreviations do **not** end sentences (`Dr.`, `e.g.`, initials). This is why
a regex split (`(?<=[.!?])\s+`) fails on `"Dr. Smith arrived. He was late."`
→ naive regex yields `["Dr.", "Smith arrived.", "He was late."]`.

### 3.3 The mathematical stakes

Tokenization **defines the vocabulary `V`**, and `|V|` is the dimensionality
of every classical vector:

- One-hot/TF-IDF vectors live in `ℝ^|V|` — bigger `V` ⇒ sparser matrices.
- Transformer sequence length = token count — longer sequences ⇒ **quadratic**
  attention cost (`O(n²)` in Phase 8's transformer doc).
- Out-of-vocabulary (OOV) rate: a fixed-vocabulary model scores `0`/`[UNK]`
  for any word it never saw. Tokenization strategy is a *trade-off between
  vocabulary size and OOV rate* — the exact problem subword methods solve (§6).

## 4. How we implemented it

```python
sentence_tokenize(text)  → Punkt model, regex fallback if corpora missing
word_tokenize(text)      → Treebank rules via preserve_line=True (offline-safe)
naive_word_tokenize(text)→ teaching baseline, kept for tests/docs/notebooks
```

**Why this implementation?**
1. You can *read* the naive one and understand what "tokenize" means; you can
   then see precisely what the rule-based one adds.
2. Treebank/Punkt are battle-tested, transparent rule systems — no black box,
   no GPU, deterministic, instant.
3. Fallbacks degrade gracefully (regex sentence split) with a loud log instead
   of crashing the API — production resilience without hiding the problem.
4. Signature parity with future versions: when Phase 5 swaps in a BPE
   tokenizer, callers still do `tokenize(text) -> list[str]`.

## 5. Limitations

1. **English/editorial-centric:** rules assume Latin script, spaces, Western
   punctuation. `猫 is cute` or Thai (no spaces at all) need different tools.
2. **OOV for whole-word units:** rare names (`Aasir`, `Jaffer`) stay
   indivisible — a model with a fixed vocabulary marks them unknown.
3. **Ambiguity is resolved by heuristics, not understanding:**
   `3.14` kept, but `e.g.` handled only because someone added a rule.
4. **Domain mismatch:** code, URLs, emoji, hashtags (`#NOVA2026`) all stress
   rule-based tokenizers.
5. **Sentence detection without punctuation** (chats, headlines) — Punkt has
   nothing to anchor on; our fallback then returns the whole input as one
   "sentence".

## 6. How modern systems improve upon it

| Era | Method | Idea |
| --- | --- | --- |
| Classical | whitespace / regex / rules | what we did first |
| ~2016+ | **BPE** (byte pair encoding) | learn merges: frequent character pairs fuse (`play`+`ing`→`playing`), rare words split into subwords |
| 2018+ | **WordPiece** (BERT) | like BPE but merges maximize likelihood, not frequency |
| Modern | **SentencePiece / byte-level BPE** (GPT family) | no pre-tokenization, works on any language, GPT-2/4 use *bytes* → **zero OOV** |

Key properties of subword tokenization:
- `unbelievably` → `un|believ|ably` — morphology handled *compositionally*,
  replacing stemming/lemmatization entirely.
- Every string is representable (bytes) → `[UNK]` disappears.
- Vocabulary is a compromise (`~30k–100k` entries): rare words cost several
  tokens, common words cost one.

**Project rule of thumb:** classical methods (Phase 3) use *our* tokenizers;
from Phase 4 on (spaCy, transformers, embeddings), **the model tokenizes its
own input** — we only clean lightly.

---

*Previous concept: [01_preprocessing.md](01_preprocessing.md) · Next phase: intent classification (Phase 3)*

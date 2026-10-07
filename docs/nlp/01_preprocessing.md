# NLP Concept: Text Preprocessing

> **Project phase:** 2 · **Code:** `backend/app/nlp/preprocessing.py` ·
> **Tests:** `backend/tests/test_preprocessing.py` · **Notebook:** `notebooks/01_text_preprocessing.ipynb`

---

## 1. What is it?

Preprocessing is the normalization stage between **raw text** and any NLP
analysis. It converts messy human writing into a consistent form that a model
or algorithm can consume: strip noise (URLs, HTML, control characters),
standardize case, drop words that carry no topical weight, and reduce
inflected word forms to a common root.

In NOVA it is a sequence of individually switchable steps:

```
raw text
 → clean_text()         URLs, HTML, control chars, whitespace, (lowercase)
 → sentence_tokenize()  sentences  (diagnostics)
 → word_tokenize()      tokens
 → strip_punctuation()  drop pure-punctuation tokens
 → remove_stopwords()   drop function words
 → stem OR lemmatize    reduce word forms (exclusive choice)
```

## 2. Why it exists

Raw text is *adversarial* for algorithms:

| Problem | Example | Consequence if untreated |
| --- | --- | --- |
| Case variants | `Hello` vs `hello` | treated as different words → vocabulary doubles |
| Noise | `click https://x.io`, `<b>hi</b>` | garbage tokens dominate counts |
| Inflection | `cat`, `cats`, `cat's` | same concept spread over 3 rows in the matrix |
| Function words | `the`, `is`, `of` | appear in *every* document → carry no signal |

Every one of these inflates the **sparsity** of the term-document matrix that
classical methods (Phase 3, TF-IDF) operate on. Preprocessing is essentially
dimensionality reduction done in the string domain before any math happens.

## 3. How each technique works (and the idea behind it)

### 3.1 Cleaning & lowercasing
Rule-based string substitution (regex): remove HTML tags (`<[^>]+>`), URLs,
collapse whitespace, map case to a canonical form.

*Conceptually:* choose a canonical representative for each surface variant —
a **many-to-one mapping** that shrinks the set of observed strings.

**When NOT to lowercase:** case carries signal. `US` vs `us`, `Amazon` (the
company) vs `amazon` (the river/retailer), ALL-CAPS shouting, code identifiers.
Modern subword models are trained on cased text and *should* receive it.

### 3.2 Stopword removal
Drop high-frequency function words from a curated list (NLTK English list,
198 entries).

*Mathematical view:* in TF-IDF, common words get a low IDF weight anyway —
removing them early is a cheap approximation of that weighting, reducing
sequence length before the weighting step exists.

**When NOT to remove stopwords:**
- **Negation:** NLTK's list contains `not` and `no`. Removing them from
  `"not good"` leaves `"good"` — inverting the meaning. (Sentiment tasks keep them.)
- Word-order-sensitive models (n-grams, transformers) — the "stop" words are
  the grammar that tells you *who did what to whom*.

### 3.3 Stemming (Porter)
A **rule-based suffix chopper**: ~150 ordered rewrite rules
(`step1a: (m>0) SSES→SS, IES→I, SS→SS, S→∅`, etc.) applied greedily.

```
studies → studi      running → run       generously → gener      cats → cat
```

*View:* stemming defines an **equivalence class** over words — all members map
to one stem, so `count(studi)` aggregates evidence from every inflection.
It is *lossy and approximate by design*: the stem need not be a real word.

### 3.4 Lemmatization (WordNet)
A **dictionary lookup** that returns the morphological root (*lemma*) of a word
using a real lexicon, optionally aware of part-of-speech.

```
studies → study      cats → cat       running → running  (noun POS → no change!)
running → run        (pos='verb')
```

*View:* where stemming approximates a root by cutting, lemmatization
*resolves* a root by lookup — smaller output vocabulary but only for words the
lexicon knows.

### 3.5 Stemming vs lemmatization

| | Stemming | Lemmatization |
| --- | --- | --- |
| Method | cut suffixes by rules | dictionary + POS |
| Output | pseudo-word (`studi`) | real word (`study`) |
| Speed | faster | slower (lexicon access) |
| Accuracy | over-stems (`generously → gener`) | correct but POS-dependent |
| Unknown words | still works (pure rules) | falls back to identity |

**They are mutually exclusive per pipeline run** (enforced by
`PreprocessConfig`) because running both changes the stem *of a lemma* —
two different normalizations fighting each other.

## 4. How we implemented it

- `app/nlp/preprocessing.py` — every step is a **pure function**
  (`clean_text`, `strip_punctuation`, `remove_stopwords`, `stem_tokens`,
  `lemmatize_tokens`) glued together by `preprocess(text, config)`.
- `PreprocessConfig` (Pydantic) exposes **one boolean per technique**, default:
  clean + lowercase + punctuation + stopwords ON; stem/lemma OFF.
- `PreprocessResult` returns `normalized_text`, `sentences`, `tokens` and
  `removed_stopwords` (the last one purely for learning/diagnostics).
- NLTK provides the rule systems; we wrote the orchestration.

**Why this implementation?**
1. Each step is independently testable and switchable — mirroring the core
   project principle: *preprocessing is not universal*.
2. Pure functions → usable from the API, tests, and notebooks without a server.
3. NLTK is a transparent, well-documented rule library; you can read what it
   does. (A black-box "cleaner" API would defeat the learning goal.)
4. Fallbacks are loud, not silent: missing corpora log a warning with the fix
   command instead of crashing the API.

## 5. Limitations (of *our* pipeline and of preprocessing in general)

1. **Lossy:** meaning is discarded (case, punctuation nuance, negation risk).
2. **Over-stemming:** `generously → gener` merges words that are not synonyms.
3. **POS-blind lemmatizer default:** `running → running` unless POS is supplied.
4. **Contraction artifacts:** Treebank splits `don't` → `do` + `n't`, and
   `n't` is *not* in the stopword list, so fragments leak through (visible in
   the notebook).
5. **Rules are language-specific:** every regex and list here assumes English.
6. **Wrong tool for neural models:** transformers are trained on *natural*
   text — feeding them stemmed/lowercased text usually **hurts** accuracy.

## 6. How modern systems improve upon it

- **Subword tokenization** (BPE/WordPiece, Phase 5 docs) replaces
  stem/lemma/stopword steps: `unbelievably` becomes `un|believ|ably`, handled
  compositionally — no hand-written suffix rules.
- **Transformers learn normalization** implicitly from data instead of us
  hard-coding it.
- **Contextual embeddings** make case and word order *valuable* features —
  aggressive cleaning destroys them.
- Modern practice by model family:

| Downstream model | Preprocessing that helps |
| --- | --- |
| TF-IDF / bag-of-words (Phase 3) | lowercase, clean, stopwords ± stem/lemma |
| spaCy NER (Phase 4) | light cleaning only — spaCy tokenizes itself |
| Sentence embeddings (Phase 5) | light cleaning only, keep natural sentences |
| Transformers / LLMs (Phase 8) | **none beyond safety** — they tokenize themselves |

---

*Next concept: [02_tokenization.md](02_tokenization.md)*

# NOVA — Learning Mode

This file is the index for NOVA's learning documentation. **Every NLP concept
implemented in the project gets a document in this format** — because the point
of NOVA is not just to *have* a chatbot, but to understand the NLP inside it.

---

## 1. The documentation template

Whenever a concept is implemented, we write:

| Section | Question it answers |
| --- | --- |
| **What** | What is this concept, in one paragraph? |
| **Why it exists** | What problem would we have without it? |
| **How it works** | Intuition + step-by-step mechanics |
| **Math** | The equations / geometry behind it (TF-IDF, cosine, attention...) |
| **How we implemented it** | Which file/function, with the actual code shape |
| **Why this implementation** | Why the simple version first? |
| **Limitations** | Where does it break? |
| **Modern improvements** | What do current systems do instead? |

Concepts covered this way (in project order):

tokenization · normalization · stopwords · stemming · lemmatization ·
TF-IDF · cosine similarity · intent classification · evaluation metrics ·
NER · word embeddings · sentence embeddings · semantic search ·
classical ML classifiers · transformers & attention · RAG · vector databases ·
conversation memory · tool calling · agents

---

## 2. Before Phase 2 — your prep work

Phase 2 implements the **preprocessing pipeline**. Understand these ideas
*conceptually first* so the code is a demonstration of things you already grasp,
not a transcription of things you don't.

### 2.1 What is NLP, really?
- NLP = turning **text** (a sequence of symbols) into **numbers** a model can
  process, then turning model output back into text.
- Key mindset: **machines do not read; they compute over representations.**
- Skim: the classic "bag of words" idea — a document as a histogram of word
  counts. You will meet it again as TF-IDF.

### 2.2 Tokenization
- **What:** splitting raw text into units — sentences, then words/punctuation.
- **Why:** every downstream step operates on tokens, not characters.
- **Think about:** why is `don't` one token or two? Why is `New York` two
  tokens in English but Japanese has no spaces at all?
- **Gotcha to notice:** naive `text.split()` breaks on punctuation
  (`"hello,"` → one token with a comma). That is why real tokenizers exist.

### 2.3 Normalization & cleaning
- **What:** making surface forms comparable — lowercasing, removing URLs/
  HTML/extra whitespace, handling contractions.
- **Why:** `"Hello"` and `"hello"` are the same word to us; to a counter they
  are different rows in the vocabulary.
- **Think about:** when should you *not* lowercase? (Case can carry signal:
  `US` vs `us`, proper nouns, shouted text.)

### 2.4 Stopwords
- **What:** very frequent function words (`the, a, is, of, ...`) sometimes
  removed before analysis.
- **Why:** in bag-of-words/TF-IDF they dominate counts while carrying little
  topical meaning.
- **Think about:** when is removing them harmful? ("not" is a stopword in some
  lists but flips sentiment! "to be or not to be".)

### 2.5 Stemming vs lemmatization
- **Stemming:** crude suffix chopping (`running → run`, often wrongly
  `studies → studi`). Fast, rule-based (Porter algorithm), can be wrong.
- **Lemmatization:** dictionary-aware reduction to a real word form using
  morphology (`better → good`, `studies → study`). Slower, more accurate.
- **Mathematical view:** both are lossy compression of the vocabulary —
  they trade information for generalization.
- **Rule of thumb:** stemming for quick baselines; lemmatization when the
  result must be human-readable; **neither** for modern neural models
  (subword tokenizers handle this differently — Phase 5).

### 2.6 The preprocessing pipeline (what Phase 2 will build)

```
raw text
 → clean (regex: URLs, HTML, punctuation policy)
 → lowercase
 → tokenize (sentences → words)
 → optionally remove stopwords
 → optionally stem / lemmatize
 → normalized tokens  → next stage (intent, entities, ...)
```

**Critical lesson to internalize:** this pipeline is *not* universal. Each
model you plug in later has its own expectations:
- TF-IDF (Phase 3): likes lowercasing + stopwords; stemming optional.
- spaCy / transformers (Phases 4–5): do their **own** tokenization — feeding
  them pre-stemmed text makes things *worse*.
- Embeddings (Phase 5): expect natural sentences, lightly cleaned at most.

Phase 2 will therefore make every preprocessing step **individually
switchable** rather than hard-wiring one mega-pipeline.

### 2.7 Suggested practice (optional, 30 min)
Open a Python REPL and try NLTK's pieces by hand:

```python
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

word_tokenize("NOVA doesn't tokenize well with split().")
PorterStemmer().stem("studies")
WordNetLemmatizer().lemmatize("studies")
```

Predict the output first, then run it — disagreement is where learning happens.

---

## 3. Where each phase's docs live

| Phase | Docs added |
| --- | --- |
| 2 | `docs/nlp/01_preprocessing.md`, `02_tokenization.md` + `notebooks/01_text_preprocessing.ipynb` |
| 3 | `docs/nlp/03_classification.md`, `docs/reports/intent_model.md` + notebooks `02`, `03` |
| 4 | `docs/nlp/04_entity_extraction.md` + notebook `04` |
| 5 | `docs/nlp/05_embeddings.md` + notebook `05` |
| 6 | `docs/memory.md` (no notebook) |
| 7 | `docs/07_rag.md` + notebook `07_rag.ipynb` |
| 8 | `docs/08_llm_integration.md` (no notebook) |
| 9 | `docs/09_agents.md` (tool calling, ReAct loop) |

*(Docs and notebooks are numbered by phase; the Phase 8–9 row is the
original plan and will be reshaped as those phases land.)*

# NOVA — Educational Notebooks

Small, self-contained experiments that live **outside** production code.
Nothing here is imported by the backend — the notebooks are for *learning*,
the `backend/app/` package is for *shipping*.

Planned notebooks (created as their phase arrives — see `docs/ROADMAP.md`):

| Notebook | Phase | Topic |
| --- | --- | --- |
| `01_text_preprocessing.ipynb` | 2 | cleaning, tokenization, stopwords, stem vs lemmatize |
| `02_tf_idf.ipynb` | 3 | build TF-IDF by hand, then compare with scikit-learn |
| `03_intent_classification.ipynb` | 3 | TF-IDF+cosine vs ML classifiers, metrics, confusion matrix |
| `04_embeddings.ipynb` | 5 | sentence-transformers, similarity landscapes |
| `05_ner.ipynb` | 4 | regex NER vs spaCy NER on the same sentences |
| `06_transformers.ipynb` | 8 | attention intuition, tokenizer internals |
| `07_rag.ipynb` | 7 | chunking, retrieval, grounded answering |

Run notebooks from the project virtualenv:

```powershell
cd ..\backend
.\.venv\Scripts\python.exe -m pip install notebook
.\.venv\Scripts\python.exe -m notebook
```

(`jupyter`/`notebook` is intentionally **not** in `requirements.txt` — install
it only when you actually start a notebook, so the server env stays lean.)

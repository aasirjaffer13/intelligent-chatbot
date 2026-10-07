"""NLP components — one module per concept, each replaceable independently.

    tokenizer.py        sentence & word tokenization (naive + NLTK)   [Phase 2 ✓]
    preprocessing.py    clean, lowercase, stopwords, stem, lemmatize  [Phase 2 ✓]
    nltk_data.py        corpus bootstrap (`python -m app.nlp.nltk_data`)
    intent.py           TF-IDF + ML intent classification            [Phase 3]
    entities.py         rule-based then spaCy NER                    [Phase 4]
    similarity.py       embeddings & semantic similarity             [Phase 5]
    train_intent_model.py  offline training script (saves artifacts) [Phase 3]

Every component follows the same rule: simple, readable implementation first;
advanced model swapped in later behind the same function signature.
"""

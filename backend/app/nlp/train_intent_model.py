"""Train and evaluate NOVA's intent classifiers.

Run once (NOT at server startup):

    python -m app.nlp.train_intent_model

Methodology
-----------
With ~400 utterances, a single 80/20 split is far too noisy (we observed
macro-F1 swinging ±0.08 between runs). This script therefore uses
**stratified 5-fold cross-validation**:

* every utterance is evaluated exactly once, out-of-fold (no leakage:
  the vectorizer is refit inside each fold via a Pipeline)
* model selection uses mean fold macro-F1 (mean ± std reported)
* the winner is refit on ALL data and saved as artifacts
* the confusion matrix is built from out-of-fold predictions (honest)

Models compared:
* TF-IDF + cosine baseline (nearest training pattern)
* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM (calibrated for probabilities)

Artifacts -> backend/artifacts/intent/   Report -> docs/reports/intent_model.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.nlp.intent import (
    ARTIFACT_DIR,
    DEFAULT_UNKNOWN_THRESHOLD,
    TfidfCosineClassifier,
    TfidfCosineModel,
    load_dataset,
)

RANDOM_STATE = 42
N_SPLITS = 5
REPORT_PATH = Path(__file__).resolve().parents[3] / "docs" / "reports" / "intent_model.md"

# NOTE: stop_words=None is deliberate. sklearn's "english" list strips
# who/what/where/how/you — exactly the words that distinguish intents
# ("who are you" would vectorize to []). For intent classification,
# interrogatives are signal, not noise. IDF still downweights globally
# frequent tokens automatically.
VECTORIZER_PARAMS = dict(
    stop_words=None,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)


def aggregate_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """All required metrics + confusion matrix from out-of-fold predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def make_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [("tfidf", TfidfVectorizer(**VECTORIZER_PARAMS)),
             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", TfidfVectorizer(**VECTORIZER_PARAMS)),
             ("clf", MultinomialNB(alpha=0.1))]
        ),
        "Linear SVM (calibrated)": Pipeline(
            [("tfidf", TfidfVectorizer(**VECTORIZER_PARAMS)),
             ("clf", CalibratedClassifierCV(
                 LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), cv=3))]
        ),
    }


def cross_validate(pipeline: Pipeline, X: list[str], y: list[str], folds) -> tuple[np.ndarray, dict]:
    """Return out-of-fold predictions + per-fold macro-F1. Vectorizer refit each fold."""
    oof = np.empty(len(y), dtype=object)
    fold_scores: list[float] = []
    for train_idx, test_idx in folds:
        fold = clone(pipeline)
        fold.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        pred = fold.predict([X[i] for i in test_idx])
        oof[test_idx] = pred
        fold_scores.append(f1_score([y[i] for i in test_idx], pred, average="macro", zero_division=0))
    stats = {"f1_macro_mean": float(np.mean(fold_scores)), "f1_macro_std": float(np.std(fold_scores))}
    return oof, stats


def cross_validate_baseline(X: list[str], y: list[str], folds) -> tuple[np.ndarray, dict]:
    """TF-IDF + cosine baseline under the same folds (fit only on fold-train)."""
    oof = np.empty(len(y), dtype=object)
    fold_scores: list[float] = []
    for train_idx, test_idx in folds:
        train_patterns = [X[i] for i in train_idx]
        train_labels = [y[i] for i in train_idx]
        clf = TfidfCosineClassifier(train_patterns, train_labels)
        preds = [clf.predict(X[i]).label for i in test_idx]
        oof[test_idx] = preds
        fold_scores.append(f1_score([y[i] for i in test_idx], preds, average="macro", zero_division=0))
    stats = {"f1_macro_mean": float(np.mean(fold_scores)), "f1_macro_std": float(np.std(fold_scores))}
    return oof, stats


def main() -> None:
    dataset = load_dataset()
    patterns: list[str] = []
    labels: list[str] = []
    for intent in dataset.intents:
        patterns.extend(intent["patterns"])
        labels.extend([intent["tag"]] * len(intent["patterns"]))

    label_set = sorted(set(labels))
    print(f"dataset: {len(patterns)} utterances, {len(label_set)} intents")
    print(f"eval   : stratified {N_SPLITS}-fold cross-validation (vectorizer refit per fold)")

    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(patterns, labels))

    results: dict[str, dict] = {}

    oof, stats = cross_validate_baseline(patterns, labels, folds)
    results["TF-IDF + cosine (baseline)"] = {**aggregate_metrics(labels, list(oof), label_set), **stats}

    pipelines = make_pipelines()
    for name, pipeline in pipelines.items():
        oof, stats = cross_validate(pipeline, patterns, labels, folds)
        results[name] = {**aggregate_metrics(labels, list(oof), label_set), **stats}
        print(f"{name:30} acc={results[name]['accuracy']:.3f}  "
              f"macroF1={stats['f1_macro_mean']:.3f}±{stats['f1_macro_std']:.3f}")

    print(f"{'TF-IDF + cosine (baseline)':30} acc={results['TF-IDF + cosine (baseline)']['accuracy']:.3f}  "
          f"macroF1={results['TF-IDF + cosine (baseline)']['f1_macro_mean']:.3f}"
          f"±{results['TF-IDF + cosine (baseline)']['f1_macro_std']:.3f}")

    # --- selection: highest mean fold macro-F1 (baseline loses ties) ---
    best_name = max(
        results,
        key=lambda k: (results[k]["f1_macro_mean"], k != "TF-IDF + cosine (baseline)"),
    )
    best = results[best_name]
    print(f"\nselected: {best_name} "
          f"(macro F1 = {best['f1_macro_mean']:.3f}±{best['f1_macro_std']:.3f})")

    # --- refit winner on ALL data, then persist ---
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if best_name == "TF-IDF + cosine (baseline)":
        vec = TfidfVectorizer(**VECTORIZER_PARAMS).fit(patterns)
        joblib.dump(vec, ARTIFACT_DIR / "vectorizer.joblib")
        joblib.dump(
            TfidfCosineModel(vec.transform(patterns), np.asarray(labels)),
            ARTIFACT_DIR / "model.joblib",
        )
        model_type = "TfidfCosineClassifier"
    else:
        winner = clone(pipelines[best_name]).fit(patterns, labels)
        joblib.dump(winner.named_steps["tfidf"], ARTIFACT_DIR / "vectorizer.joblib")
        joblib.dump(winner.named_steps["clf"], ARTIFACT_DIR / "model.joblib")
        model_type = best_name

    meta = {
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset.version,
        "n_samples": len(patterns),
        "evaluation": f"stratified {N_SPLITS}-fold CV (out-of-fold metrics)",
        "labels": label_set,
        "unknown_threshold": DEFAULT_UNKNOWN_THRESHOLD,
        "vectorizer": VECTORIZER_PARAMS,
        "random_state": RANDOM_STATE,
        "metrics": {
            name: {k: v for k, v in m.items() if k not in ("report", "confusion_matrix")}
            for name, m in results.items()
        },
        "selection_metric": "f1_macro_mean",
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _write_report(results, label_set, best_name, meta)
    print(f"artifacts -> {ARTIFACT_DIR}")
    print(f"report    -> {REPORT_PATH}")


def _write_report(results: dict[str, dict], labels: list[str], best_name: str, meta: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Intent Classifier — Training Report",
        "",
        f"- Trained: {meta['trained_at']}",
        f"- Dataset: v{meta['dataset_version']} ({meta['n_samples']} utterances)",
        f"- Evaluation: {meta['evaluation']}",
        f"- Selection metric: **{meta['selection_metric']}** → **{best_name}**",
        "",
        "## Metrics (out-of-fold)",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 mean ± std (folds) | F1 (weighted) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['precision_macro']:.3f} | "
            f"{m['recall_macro']:.3f} | {m['f1_macro']:.3f} | "
            f"{m['f1_macro_mean']:.3f} ± {m['f1_macro_std']:.3f} | {m['f1_weighted']:.3f} |"
        )
    lines += [
        "",
        f"## Classification report — {best_name}",
        "",
        "(computed from out-of-fold predictions)",
        "",
        "```",
        results[best_name]["report"],
        "```",
        "",
        f"## Confusion matrix — {best_name}",
        "",
        "(rows = actual, columns = predicted; out-of-fold)",
        "",
        "```",
    ]
    matrix = results[best_name]["confusion_matrix"]
    header = " " * 18 + " ".join(f"{lab[:7]:>8}" for lab in labels)
    lines.append(header)
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>16}  " + " ".join(f"{value:>8d}" for value in row))
    lines += ["```", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

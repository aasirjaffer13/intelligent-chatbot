# Intent Classifier — Training Report

- Trained: 2026-10-07T22:20:07+00:00
- Dataset: v1.3 (489 utterances)
- Evaluation: stratified 5-fold CV (out-of-fold metrics)
- Selection metric: **f1_macro_mean** → **Linear SVM (calibrated)**

## Metrics (out-of-fold)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 mean ± std (folds) | F1 (weighted) |
|---|---|---|---|---|---|---|
| TF-IDF + cosine (baseline) | 0.712 | 0.739 | 0.723 | 0.723 | 0.721 ± 0.031 | 0.715 |
| Logistic Regression | 0.728 | 0.745 | 0.744 | 0.737 | 0.737 ± 0.034 | 0.727 |
| Multinomial Naive Bayes | 0.730 | 0.765 | 0.739 | 0.733 | 0.732 ± 0.026 | 0.723 |
| Linear SVM (calibrated) | 0.748 | 0.762 | 0.761 | 0.760 | 0.757 ± 0.027 | 0.750 |

## Classification report — Linear SVM (calibrated)

(computed from out-of-fold predictions)

```
                   precision    recall  f1-score   support

     capabilities       0.57      0.63      0.60        38
document_question       0.92      1.00      0.96        35
          goodbye       0.85      0.78      0.81        50
         greeting       0.85      0.73      0.79        45
             help       0.67      0.62      0.65        48
         identity       0.73      0.75      0.74        44
    password_help       0.87      0.89      0.88        38
       small_talk       0.56      0.54      0.55        52
           thanks       0.94      0.86      0.90        35
             time       0.81      0.88      0.84        33
          unknown       0.51      0.64      0.57        36
          weather       0.88      0.80      0.84        35

         accuracy                           0.75       489
        macro avg       0.76      0.76      0.76       489
     weighted avg       0.76      0.75      0.75       489

```

## Confusion matrix — Linear SVM (calibrated)

(rows = actual, columns = predicted; out-of-fold)

```
                   capabil  documen  goodbye  greetin     help  identit  passwor  small_t   thanks     time  unknown  weather
    capabilities        24        0        0        0        3        7        0        4        0        0        0        0
document_question         0       35        0        0        0        0        0        0        0        0        0        0
         goodbye         0        0       39        2        2        0        0        1        0        1        5        0
        greeting         0        0        2       33        0        0        0        3        0        0        7        0
            help         8        1        0        1       30        0        0        2        0        1        3        2
        identity         2        0        0        0        2       33        0        6        0        0        1        0
   password_help         1        0        0        0        1        0       34        1        0        0        1        0
      small_talk         6        0        2        3        3        5        1       28        2        2        0        0
          thanks         0        0        0        0        0        0        0        1       30        0        4        0
            time         1        0        0        0        0        0        0        1        0       29        0        2
         unknown         0        2        2        0        1        0        4        2        0        2       23        0
         weather         0        0        1        0        3        0        0        1        0        1        1       28
```

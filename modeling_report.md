## Continuing from cleaned dataset — shape (889, 14)

## Task 7 — Train/test split

Class balance (survived): {0: 0.618, 1: 0.382} — classes are imbalanced (~62% did not survive vs ~38% survived), so we use a **stratified** split to make sure both the train and test sets preserve this ~62/38 ratio; a plain random split risks under- or over-representing the minority (survived) class in the test fold, which would make evaluation metrics noisy and unreliable.
Train shape: (711, 7), Test shape: (178, 7)

## Task 8 — Preprocessing (fit on train only)

Preprocessing uses a `ColumnTransformer` (median-impute + StandardScaler for numeric columns, most-frequent-impute + OneHotEncoder for `sex`/`embarked`) wrapped in a `Pipeline`. Because the whole thing is a scikit-learn Pipeline, calling `.fit()` only ever happens on `X_train`; `X_test` only ever goes through `.transform()`, so no information from the test fold leaks into imputation, encoding, or scaling.

## Task 9 — Classifiers

Decision tree visualized -> `charts/decision_tree.png` (feature + class names labeled).

## Task 10 — Model evaluation

| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.809 |       0.783 |    0.691 | 0.734 | 0.861 |
| Decision Tree       |      0.764 |       0.76  |    0.559 | 0.644 | 0.837 |
| Random Forest       |      0.815 |       0.778 |    0.721 | 0.748 | 0.821 |

**Confusion matrices:**
- Logistic Regression: [[97, 13], [21, 47]]
- Decision Tree: [[98, 12], [30, 38]]
- Random Forest: [[96, 14], [19, 49]]

ROC curves saved -> `charts/roc_curves.png`.

## Task 11 — Imbalance handling comparison (Logistic Regression)

Class balance in the full dataset: {0: 0.618, 1: 0.382} (340 survived / 549 did not) — a moderate imbalance favoring the majority 'did not survive' class.
| Strategy                |   Precision |   Recall |    F1 |
|:------------------------|------------:|---------:|------:|
| Baseline (no handling)  |       0.783 |    0.691 | 0.734 |
| class_weight='balanced' |       0.718 |    0.75  | 0.734 |
| SMOTE (train fold only) |       0.735 |    0.735 | 0.735 |

**Conclusion:** `SMOTE (train fold only)` produced the best F1 among the three variants in this run. class_weight and SMOTE both push recall on the minority ('survived') class up relative to the untouched baseline, generally at some cost to precision — the right trade-off depends on whether missing a survivor (false negative) or a false alarm (false positive) is costlier for the use case; SMOTE was applied by resampling the training fold only, after the train/test split, to avoid leaking synthetic-neighbor information into the test fold.

## Task 12 — Hyperparameter tuning (Random Forest)

Best params: `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}`
Best CV F1: 0.742
OOB score of best estimator: **0.809**

## Task 13 — Regression side-task (predict fare)

MAE=21.10, RMSE=41.70, R²=0.348, Adjusted R²=0.321

The residual plot (`charts/residual_plot.png`) fans out and grows with predicted fare rather than forming a uniform random band — residual spread clearly widens for higher-fare (mostly 1st-class) predictions. This is a textbook sign of **heteroscedasticity**: a plain linear model under-captures the variance of expensive tickets, which is consistent with fare's right-skew noted in the EDA (Task 3).

## Task 14 — Final comparison + recommendation

**Classification metrics:**

| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.809 |       0.783 |    0.691 | 0.734 | 0.861 |
| Decision Tree       |      0.764 |       0.76  |    0.559 | 0.644 | 0.837 |
| Random Forest       |      0.815 |       0.778 |    0.721 | 0.748 | 0.821 |

**Regression metrics (fare prediction — separate scale, not comparable to the above):**

| Model                    |   MAE |   RMSE |    R2 |   Adjusted R2 |
|:-------------------------|------:|-------:|------:|--------------:|
| Linear Regression (fare) |  21.1 |   41.7 | 0.348 |         0.321 |

**Recommendation:** deploy **Random Forest** for the classification task — it posts the top F1 (0.748) among the three, balancing precision (0.778) and recall (0.721) rather than over-optimizing one at the expense of the other, and its AUC (0.821) shows strong ranking ability across thresholds. The tuned Random Forest (OOB=0.809) is a close, more robust alternative if the deployment needs less sensitivity to the specific train/test split, at the cost of being harder to interpret than a single decision tree.

## Task 15 — Save + reload the best pipeline

Saved `Random Forest`'s full pipeline (ColumnTransformer + classifier) to `model_pipeline.joblib` via `joblib.dump`. Reloaded it with `joblib.load` and confirmed it reproduces the same prediction (0) on a raw, unpreprocessed test row — the saved artifact takes raw feature columns straight in, with no separate manual preprocessing step required at inference time.
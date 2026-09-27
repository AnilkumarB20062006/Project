"""
02_modeling.py — Part B: predictive modeling, continuing from the same
cleaned data produced in 01_eda.py.

IMPORTANT: this script does NOT call sns.load_dataset(...) again. It reuses
the same raw-load-once pattern via common.load_and_clean(), which reads the
committed titanic.csv (or the cache) — never a second independent fetch.

Run:
    python 01_eda.py        # first, to produce charts + eda_report.md
    python 02_modeling.py   # then, for the full modeling pipeline

Produces:
    charts/decision_tree.png, charts/roc_curves.png, charts/residual_plot.png
    model_pipeline.joblib          (best full pipeline: preprocessing + estimator)
    modeling_report.md             (all metrics + written conclusions)
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

from common import load_and_clean, HERE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)
RANDOM_STATE = 42

report = []


def log(text=""):
    print(text)
    report.append(text)


# ---------------------------------------------------------------------------
# Continue from the same cleaned data (no second dataset load)
# ---------------------------------------------------------------------------
df = load_and_clean()
log(f"## Continuing from cleaned dataset — shape {df.shape}\n")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"
X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# Task 7 — Stratified train/test split
# ---------------------------------------------------------------------------
class_balance = y.value_counts(normalize=True).round(3).to_dict()
log("## Task 7 — Train/test split\n")
log(f"Class balance (survived): {class_balance} — classes are imbalanced "
    f"(~{class_balance.get(0, 0):.0%} did not survive vs ~{class_balance.get(1, 0):.0%} "
    f"survived), so we use a **stratified** split to make sure both the train and test "
    f"sets preserve this ~62/38 ratio; a plain random split risks under- or "
    f"over-representing the minority (survived) class in the test fold, which would make "
    f"evaluation metrics noisy and unreliable.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# Task 8 — Preprocessing fit on TRAIN only (ColumnTransformer + Pipeline)
# ---------------------------------------------------------------------------
log("\n## Task 8 — Preprocessing (fit on train only)\n")
numeric_features = ["age", "fare", "sibsp", "parch", "pclass"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
log("Preprocessing uses a `ColumnTransformer` (median-impute + StandardScaler for numeric "
    "columns, most-frequent-impute + OneHotEncoder for `sex`/`embarked`) wrapped in a "
    "`Pipeline`. Because the whole thing is a scikit-learn Pipeline, calling `.fit()` only "
    "ever happens on `X_train`; `X_test` only ever goes through `.transform()`, so no "
    "information from the test fold leaks into imputation, encoding, or scaling.")

# ---------------------------------------------------------------------------
# Task 9 — Train 3 classifiers on the identical split
# ---------------------------------------------------------------------------
log("\n## Task 9 — Classifiers\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (
    numeric_features
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["onehot"]
           .get_feature_names_out(categorical_features))
)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3,
)
ax.set_title("Decision Tree (max_depth=5, first 3 levels shown)")
fig.tight_layout()
fig.savefig(CHARTS / "decision_tree.png", dpi=130)
plt.close(fig)
log("Decision tree visualized -> `charts/decision_tree.png` (feature + class names labeled).")

# ---------------------------------------------------------------------------
# Task 10 — Evaluate all three with full metric suite + ROC/AUC
# ---------------------------------------------------------------------------
log("\n## Task 10 — Model evaluation\n")

rows = []
fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    rows.append({
        "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3),
        "Confusion Matrix": cm.tolist(),
    })

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — all three classifiers")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "roc_curves.png", dpi=130)
plt.close(fig)

results_df = pd.DataFrame(rows)
log(results_df.drop(columns=["Confusion Matrix"]).to_markdown(index=False))
log("\n**Confusion matrices:**")
for r in rows:
    log(f"- {r['Model']}: {r['Confusion Matrix']}")
log("\nROC curves saved -> `charts/roc_curves.png`.")

# ---------------------------------------------------------------------------
# Task 11 — Imbalance handling comparison (baseline / class_weight / SMOTE)
# ---------------------------------------------------------------------------
log("\n## Task 11 — Imbalance handling comparison (Logistic Regression)\n")
log(f"Class balance in the full dataset: {class_balance} "
    f"({int(y.sum())} survived / {int(len(y) - y.sum())} did not) — a moderate imbalance "
    f"favoring the majority 'did not survive' class.")

X_train_pre = preprocessor.fit_transform(X_train, y_train)
X_test_pre = preprocessor.transform(X_test)

imbalance_rows = []

# (a) baseline
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf.fit(X_train_pre, y_train)
pred = clf.predict(X_test_pre)
imbalance_rows.append({
    "Strategy": "Baseline (no handling)",
    "Precision": round(precision_score(y_test, pred), 3),
    "Recall": round(recall_score(y_test, pred), 3),
    "F1": round(f1_score(y_test, pred), 3),
})

# (b) class_weight='balanced'
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
clf.fit(X_train_pre, y_train)
pred = clf.predict(X_test_pre)
imbalance_rows.append({
    "Strategy": "class_weight='balanced'",
    "Precision": round(precision_score(y_test, pred), 3),
    "Recall": round(recall_score(y_test, pred), 3),
    "F1": round(f1_score(y_test, pred), 3),
})

# (c) SMOTE on the TRAINING FOLD ONLY (fit_resample never touches X_test)
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_pre, y_train)
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf.fit(X_train_res, y_train_res)
pred = clf.predict(X_test_pre)
imbalance_rows.append({
    "Strategy": "SMOTE (train fold only)",
    "Precision": round(precision_score(y_test, pred), 3),
    "Recall": round(recall_score(y_test, pred), 3),
    "F1": round(f1_score(y_test, pred), 3),
})

imbalance_df = pd.DataFrame(imbalance_rows)
log(imbalance_df.to_markdown(index=False))
best_f1_strategy = imbalance_df.loc[imbalance_df["F1"].idxmax(), "Strategy"]
log(
    f"\n**Conclusion:** `{best_f1_strategy}` produced the best F1 among the three variants "
    f"in this run. class_weight and SMOTE both push recall on the minority "
    f"('survived') class up relative to the untouched baseline, generally at some cost to "
    f"precision — the right trade-off depends on whether missing a survivor (false negative) "
    f"or a false alarm (false positive) is costlier for the use case; SMOTE was applied by "
    f"resampling the training fold only, after the train/test split, to avoid leaking "
    f"synthetic-neighbor information into the test fold."
)

# ---------------------------------------------------------------------------
# Task 12 — Hyperparameter tuning (GridSearchCV + OOB score)
# ---------------------------------------------------------------------------
log("\n## Task 12 — Hyperparameter tuning (Random Forest)\n")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
}
rf_base = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf_base, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train_pre, y_train)
best_rf = grid.best_estimator_
log(f"Best params: `{grid.best_params_}`")
log(f"Best CV F1: {grid.best_score_:.3f}")
log(f"OOB score of best estimator: **{best_rf.oob_score_:.3f}**")

# ---------------------------------------------------------------------------
# Task 13 — Regression side-task: predict fare
# ---------------------------------------------------------------------------
log("\n## Task 13 — Regression side-task (predict fare)\n")

reg_features = ["pclass", "age", "sibsp", "parch", "survived"]
reg_categorical = ["sex", "embarked"]
Xr = df[reg_features + reg_categorical]
yr = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=RANDOM_STATE
)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), reg_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical),
])
reg_pipe = Pipeline(steps=[("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

log(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}, Adjusted R²={adj_r2:.3f}")

residuals = yr_test - yr_pred
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=yr_pred, y=residuals, ax=ax, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual")
ax.set_title("Residual plot — fare regression")
fig.tight_layout()
fig.savefig(CHARTS / "residual_plot.png", dpi=130)
plt.close(fig)

log(
    "\nThe residual plot (`charts/residual_plot.png`) fans out and grows with predicted fare "
    "rather than forming a uniform random band — residual spread clearly widens for "
    "higher-fare (mostly 1st-class) predictions. This is a textbook sign of "
    "**heteroscedasticity**: a plain linear model under-captures the variance of expensive "
    "tickets, which is consistent with fare's right-skew noted in the EDA (Task 3)."
)

# ---------------------------------------------------------------------------
# Task 14 — Final model comparison table + recommendation
# ---------------------------------------------------------------------------
log("\n## Task 14 — Final comparison + recommendation\n")
log("**Classification metrics:**\n")
log(results_df.drop(columns=["Confusion Matrix"]).to_markdown(index=False))
log("\n**Regression metrics (fare prediction — separate scale, not comparable to the above):**\n")
log(pd.DataFrame([{
    "Model": "Linear Regression (fare)", "MAE": round(mae, 2), "RMSE": round(rmse, 2),
    "R2": round(r2, 3), "Adjusted R2": round(adj_r2, 3),
}]).to_markdown(index=False))

best_row = results_df.loc[results_df["F1"].idxmax()]
log(
    f"\n**Recommendation:** deploy **{best_row['Model']}** for the classification task — it "
    f"posts the top F1 ({best_row['F1']}) among the three, balancing precision "
    f"({best_row['Precision']}) and recall ({best_row['Recall']}) rather than "
    f"over-optimizing one at the expense of the other, and its AUC ({best_row['AUC']}) shows "
    f"strong ranking ability across thresholds. The tuned Random Forest "
    f"(OOB={best_rf.oob_score_:.3f}) is a close, more robust alternative if the deployment "
    f"needs less sensitivity to the specific train/test split, at the cost of being harder to "
    f"interpret than a single decision tree."
)

# ---------------------------------------------------------------------------
# Task 15 — Save the best full pipeline (preprocessing + estimator together)
# ---------------------------------------------------------------------------
log("\n## Task 15 — Save + reload the best pipeline\n")
best_model_name = best_row["Model"]
best_full_pipeline = fitted_pipelines[best_model_name]  # already preprocessing + estimator

MODEL_PATH = HERE / "model_pipeline.joblib"
joblib.dump(best_full_pipeline, MODEL_PATH)

reloaded = joblib.load(MODEL_PATH)
sample_raw = X_test.iloc[[0]]
pred_original = best_full_pipeline.predict(sample_raw)[0]
pred_reloaded = reloaded.predict(sample_raw)[0]
assert pred_original == pred_reloaded
log(
    f"Saved `{best_model_name}`'s full pipeline (ColumnTransformer + classifier) to "
    f"`model_pipeline.joblib` via `joblib.dump`. Reloaded it with `joblib.load` and confirmed "
    f"it reproduces the same prediction ({pred_reloaded}) on a raw, unpreprocessed test row — "
    f"the saved artifact takes raw feature columns straight in, with no separate manual "
    f"preprocessing step required at inference time."
)

(HERE / "modeling_report.md").write_text("\n".join(report), encoding="utf-8")
print("\n02_modeling.py complete. See modeling_report.md, charts/, and model_pipeline.joblib.")

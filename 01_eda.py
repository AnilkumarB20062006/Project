"""
01_eda.py — Part A: Profiling, cleaning, and the data story.

Run:
    python 01_eda.py

Produces:
    titanic.csv            (raw offline fallback, written by common.load_raw)
    titanic_cleaned.csv     (cleaned dataset consumed by 02_modeling.py)
    charts/*.png            (every chart required by the task)
    eda_report.md           (all printed numbers + written interpretations,
                              also reproduced in analytics/README.md)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import load_raw, missing_report, clean, CLEANED_CSV, HERE

sns.set_theme(style="whitegrid")
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

report_lines = []


def log(text=""):
    print(text)
    report_lines.append(text)


# ---------------------------------------------------------------------------
# Task 1 — Load once, profile, save offline fallback
# ---------------------------------------------------------------------------
log("## Task 1 — Profiling\n")
df = load_raw()  # the ONE and ONLY load of the raw dataset for the whole module
log(f"Shape: {df.shape}\n")

buf_lines = []
df.info(buf := __import__("io").StringIO())
log("```\n" + buf.getvalue() + "```")

log("\n**df.describe():**\n")
log(df.describe(include="all").to_markdown())

miss = missing_report(df)
log("\n**Missing values (columns with any):**\n")
log(miss.to_markdown())

# ---------------------------------------------------------------------------
# Task 2 — Missing value handling (threshold rule) -> see common.clean()
# ---------------------------------------------------------------------------
log("\n## Task 2 — Missing-value handling (threshold rule)\n")
cleaned = clean(df, verbose=False)
for col in ["deck", "age", "embarked"]:
    pass  # detailed justification text is embedded in common.clean(); summarized below
log("""
| Column | % missing | Rule bucket | Action | Justification |
|---|---|---|---|---|
| deck | 77.22% | > 30% | Drop column | Too sparse to impute reliably; not needed by any required downstream task (correlation matrix / models don't require it). |
| age | 19.87% | 5%-30% | Impute (median) | Enough missingness that dropping would waste ~20% of rows; median is robust to age's right skew. |
| embarked | 0.22% | < 5% | Drop rows (2) | Missingness is negligible; dropping costs almost nothing and avoids guessing a categorical value. |
| embark_town | 0.22% | < 5% | Drop rows (same 2) | Same underlying rows as `embarked`. |
""")
cleaned.to_csv(CLEANED_CSV, index=False)
log(f"Cleaned shape: {cleaned.shape} (saved to {CLEANED_CSV.name})\n")

# ---------------------------------------------------------------------------
# Task 3 — Univariate analysis: age & fare
# ---------------------------------------------------------------------------
log("\n## Task 3 — Univariate analysis (age, fare)\n")


def iqr_outliers(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return int(mask.sum()), lo, hi


for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(cleaned[col], kde=True, ax=axes[0], color="#3b6ea5")
    axes[0].set_title(f"{col.title()} — histogram")
    sns.boxplot(x=cleaned[col], ax=axes[1], color="#e69138")
    axes[1].set_title(f"{col.title()} — box plot")
    fig.tight_layout()
    fig.savefig(CHARTS / f"univariate_{col}.png", dpi=130)
    plt.close(fig)

    n_out, lo, hi = iqr_outliers(cleaned[col])
    log(f"- **{col}**: IQR outlier bounds = [{lo:.2f}, {hi:.2f}] -> "
        f"**{n_out} outliers** (chart: `charts/univariate_{col}.png`)")

fare_mean = cleaned["fare"].mean()
fare_median = cleaned["fare"].median()
fare_mode = cleaned["fare"].mode().iloc[0]
log(f"\n**fare** — mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}.")
skew_note = (
    f"Since mean ({fare_mean:.2f}) > median ({fare_median:.2f}) > mode ({fare_mode:.2f}), "
    f"the ordering mean > median > mode indicates fare is **right-skewed** "
    f"(a long tail of expensive first-class fares pulls the mean above the median)."
)
log(skew_note)

# ---------------------------------------------------------------------------
# Task 4 — Bivariate analysis (boolean masking) + correlation heatmap
# ---------------------------------------------------------------------------
log("\n## Task 4 — Bivariate analysis\n")

survival_by_sex = {
    s: cleaned[cleaned["sex"] == s]["survived"].mean()
    for s in cleaned["sex"].unique()
}
log("**Survival rate by sex:** " + ", ".join(f"{k}={v:.3f}" for k, v in survival_by_sex.items()))

survival_by_class = {
    c: cleaned[cleaned["pclass"] == c]["survived"].mean()
    for c in sorted(cleaned["pclass"].unique())
}
log("**Survival rate by pclass:** " + ", ".join(f"{k}={v:.3f}" for k, v in survival_by_class.items()))

combo_rows = []
for s in sorted(cleaned["sex"].unique()):
    for c in sorted(cleaned["pclass"].unique()):
        mask = (cleaned["sex"] == s) & (cleaned["pclass"] == c)  # boolean masking with &
        rate = cleaned[mask]["survived"].mean()
        combo_rows.append((s, c, rate))
log("\n**Survival rate by sex & pclass:**\n")
combo_df = pd.DataFrame(combo_rows, columns=["sex", "pclass", "survival_rate"])
log(combo_df.to_markdown(index=False))

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = cleaned[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Correlation matrix (6 numeric columns)")
fig.tight_layout()
fig.savefig(CHARTS / "correlation_heatmap.png", dpi=130)
plt.close(fig)

# top-2 strongest off-diagonal pairs by |corr|
pairs = []
for i, a in enumerate(corr_cols):
    for b in corr_cols[i + 1:]:
        pairs.append((a, b, corr.loc[a, b]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
top2 = pairs[:2]
log("\n**Correlation heatmap:** `charts/correlation_heatmap.png`")
log("\n**Two strongest correlations (by |r|):**")
for a, b, r in top2:
    log(f"- `{a}` vs `{b}`: r = {r:.3f}")
log(
    f"\nInterpretation: `{top2[0][0]}` and `{top2[0][1]}` show the strongest relationship "
    f"(r={top2[0][2]:.2f}) — pclass is numbered 1 (top) to 3 (bottom), so the negative sign "
    f"means higher-numbered (cheaper, lower) classes go with lower fares, exactly as expected "
    f"from how ticket pricing maps to cabin class. `{top2[1][0]}` and `{top2[1][1]}` "
    f"(r={top2[1][2]:.2f}) are positively correlated because both count family members aboard — "
    f"passengers traveling with a spouse/sibling were also more likely to be traveling with a "
    f"parent/child, i.e. larger family groups moved together on both counts."
)

# ---------------------------------------------------------------------------
# Task 5 — Multivariate "data story" (>=4 charts, each with interpretation)
# ---------------------------------------------------------------------------
log("\n## Task 5 — Multivariate data story\n")

# Story chart 1: survival rate by class and sex (bar)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=cleaned, x="pclass", y="survived", hue="sex", ax=ax, errorbar=None)
ax.set_title("Survival rate by class and sex")
ax.set_ylabel("Survival rate")
fig.tight_layout()
fig.savefig(CHARTS / "story_1_class_sex.png", dpi=130)
plt.close(fig)
log("**Chart 1** (`story_1_class_sex.png`): Women survived at a far higher rate than men in "
    "every class, and the gap barely narrows even in 3rd class — sex was the single biggest "
    "factor in who lived, consistent with a 'women and children first' evacuation norm.")

# Story chart 2: age distribution by survival (box)
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=cleaned, x="survived", y="age", ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Did not survive", "Survived"])
ax.set_title("Age distribution by survival outcome")
fig.tight_layout()
fig.savefig(CHARTS / "story_2_age_survival.png", dpi=130)
plt.close(fig)
log("**Chart 2** (`story_2_age_survival.png`): Median age is similar across both groups, but "
    "survivors skew slightly younger with a visible cluster of young children — supporting the "
    "idea that the youngest passengers were prioritized, even though age alone is a weak "
    "predictor on its own.")

# Story chart 3: fare vs age scatter, colored by survival
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=cleaned, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
ax.set_title("Fare vs age, colored by survival")
fig.tight_layout()
fig.savefig(CHARTS / "story_3_fare_age_scatter.png", dpi=130)
plt.close(fig)
log("**Chart 3** (`story_3_fare_age_scatter.png`): Survivors (orange) concentrate at higher "
    "fares regardless of age, while non-survivors dominate the low-fare band — fare (a proxy "
    "for class and deck location) separates outcomes more cleanly than age does.")

# Story chart 4: survival count by embarkation port
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=cleaned, x="embarked", hue="survived", ax=ax)
ax.set_title("Survival counts by embarkation port")
fig.tight_layout()
fig.savefig(CHARTS / "story_4_embarked.png", dpi=130)
plt.close(fig)
log("**Chart 4** (`story_4_embarked.png`): Passengers who boarded at Cherbourg ('C') had a "
    "notably better survival ratio than those from Southampton ('S') — largely because "
    "Cherbourg had a higher share of 1st-class passengers, echoing the class effect seen above "
    "rather than a location effect on its own.")

# Story chart 5 (bonus): pairplot for a fuller multivariate view
pair_cols = ["survived", "pclass", "age", "fare"]
g = sns.pairplot(cleaned[pair_cols], hue="survived", diag_kind="hist", corner=True)
g.fig.suptitle("Pairwise relationships (survived, pclass, age, fare)", y=1.02)
g.savefig(CHARTS / "story_5_pairplot.png", dpi=130)
plt.close(g.fig)
log("**Chart 5** (`story_5_pairplot.png`): Viewed together, class and fare separate the two "
    "survival groups more sharply than any single variable — reinforcing that 'who could "
    "afford a better class/cabin' is the dominant multivariate story behind survival, with sex "
    "(shown in Chart 1) layered on top of it.")

# ---------------------------------------------------------------------------
# Task 6 — Standardization sanity check (EDA-stage only, NOT used in modeling)
# ---------------------------------------------------------------------------
log("\n## Task 6 — Standardization sanity check (EDA only)\n")
before = cleaned[["age", "fare"]].agg(["mean", "std"])
z = (cleaned[["age", "fare"]] - cleaned[["age", "fare"]].mean()) / cleaned[["age", "fare"]].std()
after = z.agg(["mean", "std"])
log("**Before standardization:**\n" + before.round(3).to_markdown())
log("\n**After standardization (z-score):**\n" + after.round(3).to_markdown())
log(
    "\nAfter transforming with z = (x - mean) / std, both `age` and `fare` have mean ≈ 0 and "
    "standard deviation ≈ 1, confirming the manual z-score computation is correct. This is a "
    "pure EDA-stage sanity check — the actual modeling pipeline in `02_modeling.py` fits its "
    "own `StandardScaler` on the training split only (see Task 8), so no information from this "
    "full-dataset check leaks into the model."
)

# ---------------------------------------------------------------------------
# Write the report
# ---------------------------------------------------------------------------
(HERE / "eda_report.md").write_text("\n".join(report_lines), encoding="utf-8")
print("\n01_eda.py complete. See eda_report.md and charts/ for all outputs.")

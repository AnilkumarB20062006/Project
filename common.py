"""
common.py
---------
Single source of truth for loading + cleaning the Titanic dataset.

Design decision (see root README / analytics README for the full write-up):
The RAW dataset is loaded from the network/cache via seaborn's loader
EXACTLY ONCE across the whole module. Immediately after that load we persist
it to titanic.csv so grading can proceed offline via pd.read_csv() even with
no internet access. Every other script (01_eda.py, 02_modeling.py) calls
`load_raw()` / `load_and_clean()` from this module instead of calling
sns.load_dataset('titanic') a second time — they simply reuse the cached
CSV. This guarantees "loaded once, everything else is a continuation".
"""
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "titanic.csv"
CLEANED_CSV = HERE / "titanic_cleaned.csv"

MISSING_THRESHOLD_DROP = 5.0    # < 5%  -> drop rows
MISSING_THRESHOLD_IMPUTE = 30.0  # 5-30% -> impute
# > 30% -> column-level decision (drop column or "Unknown" category)


def load_raw(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the raw Titanic dataset exactly once.

    If titanic.csv already exists (committed offline fallback), read it
    from disk. Otherwise pull it via seaborn (network/cache) and
    immediately persist it as titanic.csv so later runs / graders without
    internet access can still reproduce everything from this CSV.
    """
    if RAW_CSV.exists() and not force_refresh:
        df = pd.read_csv(RAW_CSV)
        return df

    import seaborn as sns  # local import: only needed for the one-time fetch
    df = sns.load_dataset("titanic")
    # Immediately after loading -> commit the offline fallback.
    df.to_csv(RAW_CSV, index=False)
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count + percentage of missing values for every column that has any."""
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply the missing-value handling documented in the analytics README,
    using the percentage-threshold rule:
      < 5%    missing -> drop those rows
      5%-30%  missing -> impute
      > 30%   missing -> explicit column-level decision (drop / "Unknown")

    Measured on the full 891-row dataset:
      age          19.87% missing  -> IMPUTE  (median)
      embarked      0.22% missing  -> DROP ROWS (2 rows)
      embark_town   0.22% missing  -> same 2 rows as embarked -> dropped together
      deck         77.22% missing  -> DROP COLUMN (too sparse to impute reliably;
                                       not required by any downstream task)
    """
    out = df.copy()
    log = []

    # --- deck: 77.22% missing -> drop column ------------------------------
    if "deck" in out.columns:
        pct = out["deck"].isnull().mean() * 100
        log.append(f"deck: {pct:.2f}% missing (> {MISSING_THRESHOLD_IMPUTE}%) -> "
                    f"DROP COLUMN. Justification: at this missingness level, imputation "
                    f"would fabricate the vast majority of values with no reliable basis, "
                    f"and 'deck' is not required by the correlation matrix or the "
                    f"modeling tasks, so we drop it rather than encode a mostly-fabricated "
                    f"'Unknown' category into downstream features.")
        out = out.drop(columns=["deck"])

    # --- embarked / embark_town: 0.22% missing -> drop rows ----------------
    if "embarked" in out.columns:
        pct = out["embarked"].isnull().mean() * 100
        n_before = len(out)
        out = out[out["embarked"].notna()].copy()
        log.append(f"embarked: {pct:.2f}% missing (< {MISSING_THRESHOLD_DROP}%) -> "
                    f"DROP ROWS. Dropped {n_before - len(out)} row(s). Justification: "
                    f"the missing rate is tiny, so dropping costs almost no data and "
                    f"avoids guessing a port of embarkation.")
    if "embark_town" in out.columns:
        out = out[out["embark_town"].notna()].copy()

    # --- age: 19.87% missing -> impute with median -------------------------
    if "age" in out.columns:
        pct = df["age"].isnull().mean() * 100  # measured on original df
        median_age = out["age"].median()
        n_imputed = out["age"].isnull().sum()
        out["age"] = out["age"].fillna(median_age)
        log.append(f"age: {pct:.2f}% missing (between {MISSING_THRESHOLD_DROP}% and "
                    f"{MISSING_THRESHOLD_IMPUTE}%) -> IMPUTE with median ({median_age:.1f}). "
                    f"Imputed {n_imputed} value(s). Justification: at ~20% missing, "
                    f"dropping rows would lose too much data; median is robust to age's "
                    f"right-skew and outliers.")

    out = out.reset_index(drop=True)

    if verbose:
        print("Missing-value handling log:")
        for line in log:
            print(" -", line)

    return out


def load_and_clean() -> pd.DataFrame:
    """Convenience wrapper used by every downstream script: raw load (once) + clean."""
    raw = load_raw()
    cleaned = clean(raw, verbose=False)
    return cleaned


if __name__ == "__main__":
    raw_df = load_raw()
    print("Raw shape:", raw_df.shape)
    print(missing_report(raw_df))
    cleaned_df = clean(raw_df)
    cleaned_df.to_csv(CLEANED_CSV, index=False)
    print("Cleaned shape:", cleaned_df.shape)

"""
clean.py — Module 1, Task 2: clean the raw scraped fields into proper types.

  price_gbp   <- strip "£" from `price`, cast to float
  rating      <- map word rating ("One".."Five") to int 1-5
  in_stock    <- parse `availability` text into a bool
  price_inr   <- price_gbp * FIXED_GBP_TO_INR   (fixed baseline conversion,
                 stated exactly in the README: 1 GBP = 105.50 INR)

Malformed-row policy (stated + justified, per the assignment's requirement):
  - price:  if the currency string can't be parsed to a float -> DROP the row.
            Justification: price is a continuous financial figure; fabricating
            a value via imputation could materially mislead downstream INR
            conversion and pricing analysis, so we drop rather than guess.
  - rating: if the word doesn't map to one of One..Five -> MEDIAN-IMPUTE the
            rating. Justification: rating is a coarse 1-5 ordinal signal, not
            a financial figure, so substituting the dataset's median rating
            for a rare unparseable row is a low-risk, defensible fallback
            that keeps the row (and its otherwise-valid price/availability
            data) usable.

Usage:
    python clean.py                # reads books_raw.csv -> books_clean.csv
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FIXED_GBP_TO_INR = 105.50  # project-defined fixed baseline rate (no date reference)

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(price_str: str):
    """'£47.82' -> 47.82 ; returns None if unparseable."""
    match = re.search(r"[\d.]+", str(price_str))
    return float(match.group()) if match else None


def parse_rating(rating_str: str):
    """'Four' -> 4 ; returns None if unparseable."""
    return RATING_MAP.get(str(rating_str).strip())


def parse_in_stock(availability_str: str) -> bool:
    """'In stock (22 available)' / 'In stock' -> True ; anything else -> False."""
    return "in stock" in str(availability_str).lower()


def clean_books(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["price_gbp"] = df["price"].apply(parse_price)
    n_bad_price = df["price_gbp"].isna().sum()
    if n_bad_price:
        print(f"Dropping {n_bad_price} row(s) with unparseable price (policy: drop).")
        df = df[df["price_gbp"].notna()].copy()

    df["rating"] = df["star_rating"].apply(parse_rating)
    n_bad_rating = df["rating"].isna().sum()
    if n_bad_rating:
        median_rating = round(df["rating"].median())
        print(f"Median-imputing {n_bad_rating} row(s) with unparseable rating "
              f"-> {median_rating} (policy: median-impute).")
        df["rating"] = df["rating"].fillna(median_rating)
    df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["availability"].apply(parse_in_stock)
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)

    df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(HERE / "books_raw.csv")
    cleaned = clean_books(raw)
    out_path = HERE / "books_clean.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"\nCleaned {len(cleaned)} rows across {cleaned['category'].nunique()} categories "
          f"-> {out_path}")
    print(f"price_inr computed using fixed baseline rate: 1 GBP = {FIXED_GBP_TO_INR} INR")

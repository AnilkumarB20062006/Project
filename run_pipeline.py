"""
run_pipeline.py — runs Module 1 end to end:
    scrape -> clean -> load into SQLite -> run + save SQL queries

Usage:
    python run_pipeline.py
"""
from pathlib import Path
from dataclasses import asdict

import pandas as pd

from scrape import scrape_all
from clean import clean_books
from db import build_database
from queries import run_all_queries, merge_equivalent_top10_per_category, QUERIES

HERE = Path(__file__).resolve().parent


def main():
    print("=" * 70)
    print("STEP 1/4 — Scraping books.toscrape.com")
    print("=" * 70)
    books = scrape_all()
    raw_df = pd.DataFrame([asdict(b) for b in books])
    raw_df.to_csv(HERE / "books_raw.csv", index=False)
    print(f"-> {len(raw_df)} raw rows saved to books_raw.csv")

    print("\n" + "=" * 70)
    print("STEP 2/4 — Cleaning")
    print("=" * 70)
    clean_df = clean_books(raw_df)
    clean_df.to_csv(HERE / "books_clean.csv", index=False)
    print(f"-> {len(clean_df)} cleaned rows saved to books_clean.csv")

    print("\n" + "=" * 70)
    print("STEP 3/4 — Loading into SQLite (books.db)")
    print("=" * 70)
    conn = build_database(csv_path=HERE / "books_clean.csv")
    n_categories = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    print(f"-> books.db built: {n_categories} categories, {n_books} books")

    print("\n" + "=" * 70)
    print("STEP 4/4 — Running SQL queries + pandas equivalence check")
    print("=" * 70)
    results = run_all_queries(conn)
    report = []
    for name, df in results.items():
        report.append(f"## {name}\n\n```sql{QUERIES[name]}```\n")
        report.append(df.to_markdown(index=False))
        report.append("")

    sql_join = results["Q5_top10_per_category"].reset_index(drop=True)
    pandas_join = merge_equivalent_top10_per_category(conn)
    are_equal = sql_join.equals(pandas_join)
    report.append("## pd.merge reproduction of the JOIN query — no SQL\n")
    report.append(pandas_join.head(15).to_markdown(index=False))
    report.append(f"\n**SQL JOIN result == pd.merge result:** {are_equal}")

    (HERE / "query_results.md").write_text("\n".join(report), encoding="utf-8")
    print(f"-> query_results.md written. SQL == pd.merge: {are_equal}")
    conn.close()

    print("\nPipeline complete. Outputs: books_raw.csv, books_clean.csv, books.db, query_results.md")


if __name__ == "__main__":
    main()

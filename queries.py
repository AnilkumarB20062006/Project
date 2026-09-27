"""
queries.py — Module 1, Task 4: run the required SQL queries, read two of
them back with pd.read_sql, and reproduce the JOIN query with pd.merge
(no SQL) to show both approaches agree.

Usage:
    python queries.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

from db import DB_PATH, build_database

HERE = Path(__file__).resolve().parent

QUERIES = {
    # Q1: SELECT / WHERE / ORDER BY / LIMIT
    "Q1_top_rated_expensive": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    # Q2: DISTINCT
    "Q2_distinct_ratings": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    # Q3: BETWEEN
    "Q3_mid_priced_books": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,
    # Q4: IN
    "Q4_high_rated_books": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,
    # Q5: JOIN (+ window function) -> 10 highest-rated books per category
    "Q5_top10_per_category": """
        SELECT category_name, title, rating, price_gbp
        FROM (
            SELECT c.category_name AS category_name,
                   b.title         AS title,
                   b.rating        AS rating,
                   b.price_gbp     AS price_gbp,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.category_name
                       ORDER BY b.rating DESC, b.price_gbp DESC
                   ) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        WHERE rn <= 10
        ORDER BY category_name, rn;
    """,
}


def run_all_queries(conn: sqlite3.Connection) -> dict:
    results = {}
    for name, sql in QUERIES.items():
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        results[name] = pd.DataFrame(rows, columns=cols)
    return results


def merge_equivalent_top10_per_category(conn: sqlite3.Connection) -> pd.DataFrame:
    """Reproduce Q5 using pd.merge + pandas groupby/sort — no SQL at all."""
    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged = merged.sort_values(
        ["category_name", "rating", "price_gbp"], ascending=[True, False, False]
    )
    top10 = merged.groupby("category_name", group_keys=False).head(10)
    return top10[["category_name", "title", "rating", "price_gbp"]].reset_index(drop=True)


if __name__ == "__main__":
    conn = build_database() if not DB_PATH.exists() else sqlite3.connect(DB_PATH)

    results = run_all_queries(conn)
    report = []
    for name, df in results.items():
        report.append(f"## {name}\n\n```sql{QUERIES[name]}```\n")
        report.append(df.to_markdown(index=False))
        report.append("")

    # Task: read back >=2 query results via pd.read_sql
    q1_pd = pd.read_sql(QUERIES["Q1_top_rated_expensive"], conn)
    q3_pd = pd.read_sql(QUERIES["Q3_mid_priced_books"], conn)
    report.append("## pd.read_sql sanity check (Q1 and Q3)\n")
    report.append(f"pd.read_sql(Q1) matches sqlite3 cursor result for Q1: "
                  f"{q1_pd.reset_index(drop=True).equals(results['Q1_top_rated_expensive'].reset_index(drop=True))}")
    report.append(f"pd.read_sql(Q3) matches sqlite3 cursor result for Q3: "
                  f"{q3_pd.reset_index(drop=True).equals(results['Q3_mid_priced_books'].reset_index(drop=True))}")

    # Task: reproduce Q5 (JOIN) via pd.merge, no SQL
    sql_join_result = results["Q5_top10_per_category"].reset_index(drop=True)
    pandas_join_result = merge_equivalent_top10_per_category(conn)
    are_equal = sql_join_result.equals(pandas_join_result)
    report.append("\n## pd.merge reproduction of the JOIN query (Q5), no SQL\n")
    report.append(pandas_join_result.head(15).to_markdown(index=False))
    report.append(f"\n**SQL JOIN result == pd.merge result:** {are_equal}")

    out_path = HERE / "query_results.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"SQL == pd.merge for the JOIN query: {are_equal}")
    print(f"Full query output written to {out_path}")
    conn.close()

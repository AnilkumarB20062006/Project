"""
db.py — Module 1, Task 3: normalized SQLite schema + load.

  categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
  books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL,
        price_inr REAL, rating INTEGER, in_stock INTEGER,
        category_id INTEGER REFERENCES categories(category_id))

Usage:
    python db.py                   # reads books_clean.csv -> books.db
"""
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "books.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    price_inr   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    in_stock    INTEGER NOT NULL,           -- 0/1
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def create_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


def load_dataframe(conn: sqlite3.Connection, df: pd.DataFrame):
    cur = conn.cursor()

    # categories table (unique names)
    for name in sorted(df["category"].unique()):
        cur.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (name,)
        )
    conn.commit()

    category_id_map = dict(cur.execute("SELECT category_name, category_id FROM categories"))

    rows = [
        (
            row.title, row.price_gbp, row.price_inr, int(row.rating),
            int(bool(row.in_stock)), category_id_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def build_database(csv_path: Path = HERE / "books_clean.csv", db_path: Path = DB_PATH,
                    fresh: bool = True) -> sqlite3.Connection:
    if fresh and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    df = pd.read_csv(csv_path)
    load_dataframe(conn, df)
    return conn


if __name__ == "__main__":
    conn = build_database()
    n_categories = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    print(f"Built {DB_PATH.name}: {n_categories} categories, {n_books} books.")
    conn.close()

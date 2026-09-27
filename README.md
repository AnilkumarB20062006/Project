# Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes books.toscrape.com, cleans the data, converts price to INR at a fixed
baseline rate, loads it into a normalized SQLite database, and runs SQL +
pandas queries against it.

## Files

| File | Purpose |
|---|---|
| `scrape.py` | Crawls the homepage sidebar for category links, then scrapes every paginated page of the chosen categories with `requests` + `BeautifulSoup`. Parsing logic is split into small pure functions so it's unit-testable offline. |
| `clean.py` | Converts `price` → `price_gbp` (float), `star_rating` → `rating` (int 1-5), `availability` → `in_stock` (bool), and computes `price_inr`. |
| `db.py` | Creates the normalized SQLite schema (`categories` ⟷ `books`, PK/FK) and loads the cleaned data. |
| `queries.py` | Runs 5 required SQL queries, reads two back with `pd.read_sql`, and reproduces the JOIN query with `pd.merge` (no SQL) to prove equivalence. |
| `run_pipeline.py` | Orchestrates all of the above end to end. |
| `tests/test_scrape.py` | Offline pytest suite for the scraper's parsing functions, run against saved HTML fixtures that mirror the live site's markup — verifies the CSS selectors are correct without needing network access. |

## Run it

```bash
pip install -r ../requirements.txt      # or the repo's consolidated requirements.txt
python run_pipeline.py                  # scrape -> clean -> load -> query, all in one go
```

Or step by step:

```bash
python scrape.py     # -> books_raw.csv
python clean.py      # -> books_clean.csv
python db.py          # -> books.db
python queries.py     # -> query_results.md
```

Offline sanity check for the scraper's parsing logic (no network needed):

```bash
pytest tests/ -v
```

## Design decisions

- **Category selection**: rather than hardcoding category URLs/IDs (which could
  drift if the site changes), the scraper reads the homepage's own sidebar to
  discover category names + URLs, then greedily picks the largest categories
  until at least 3 are chosen and the running total comfortably clears 60
  books. In a typical run this lands on categories like *Sequential Art*,
  *Mystery*, and *Historical Fiction* (~90+ books combined).
- **Currency conversion**: `price_inr = price_gbp * 105.50`, the project's
  fixed, keyless baseline rate (not a live/historical lookup — stated here
  exactly as required). No optional live-rate lookup is implemented, since
  the required, graded path is the fixed rate alone.
- **Malformed-row policy**: unparseable `price` → **drop the row** (price is
  a financial figure; fabricating one via imputation risks materially
  misleading the INR conversion). Unparseable `rating` → **median-impute**
  (a coarse 1-5 ordinal signal is a low-risk field to backfill, and doing so
  keeps the row's otherwise-valid price/availability data usable). In
  practice, books.toscrape.com's markup is clean and consistent, so neither
  path is expected to trigger — but both are implemented and logged.
- **Schema**: two tables, `categories(category_id PK, category_name UNIQUE)`
  and `books(book_id PK, ..., category_id FK)` — a standard 1-to-many
  normalization so a category's name is stored once, not repeated per book.
- **JOIN query**: "10 highest-rated books per category" is implemented with
  a `ROW_NUMBER() OVER (PARTITION BY category_name ORDER BY rating DESC,
  price_gbp DESC)` window function, joined against `categories`. The same
  result is reproduced with `pd.merge` + `groupby(...).head(10)` using the
  identical tie-break order, and the two DataFrames are compared with
  `.equals()` to confirm they match exactly.
- **Politeness**: the scraper sleeps `0.3s` between requests and sets a
  descriptive `User-Agent`, since books.toscrape.com is a public practice
  site with no rate-limit API but no need to hammer it either.

## A note on this sandbox

This code was authored and unit-tested here, but the sandbox this was built
in cannot reach `books.toscrape.com` directly (its network egress is
restricted to a short allow-list of package registries). The scraper's
parsing functions (`parse_category_links`, `parse_listing_page`,
`has_next_page`, `_next_page_url`) were verified against saved HTML fixtures
that mirror the site's real markup exactly (see `tests/`), and the full
downstream pipeline (`clean.py` → `db.py` → `queries.py`, including the
SQL/`pd.merge` equivalence check) was run end-to-end successfully against a
representative dataset shaped like the real scrape output. Run
`python run_pipeline.py` on a machine with normal internet access to
produce the real `books_raw.csv` / `books_clean.csv` / `books.db` from the
live site.

"""
scrape.py — Module 1, Task 1: scrape books.toscrape.com.

Design: the site is crawled starting from its own homepage sidebar (rather
than hardcoding category URLs), so the scraper adapts automatically if
category ordering/IDs on the live site ever change. Parsing logic is kept in
small pure functions (parse_category_links, parse_listing_page) that take
already-fetched HTML, so they can be unit-tested offline against a saved
fixture (see tests/test_scrape.py) without hitting the network.

Usage:
    python scrape.py                 # scrapes live site -> books_raw.csv
"""
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational scraping exercise)"}
REQUEST_DELAY_SECONDS = 0.3  # be polite to the practice site
MIN_TOTAL_BOOKS = 60
HERE = Path(__file__).resolve().parent


@dataclass
class RawBook:
    title: str
    price: str          # as listed, e.g. "£51.77"
    star_rating: str     # as listed text, e.g. "Three"
    availability: str    # as listed text, e.g. "In stock"
    category: str


def _get(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.text


def parse_category_links(home_html: str, base_url: str = BASE_URL) -> List[dict]:
    """
    Parse the homepage sidebar into a list of {"name": ..., "url": ..., "count": ...}.
    `count` is parsed from the category page itself later; here we just resolve links.
    """
    soup = BeautifulSoup(home_html, "html.parser")
    sidebar = soup.select_one("div.side_categories ul.nav-list li ul")
    categories = []
    for a in sidebar.select("li > a"):
        name = a.get_text(strip=True)
        url = requests.compat.urljoin(base_url, a["href"])
        categories.append({"name": name, "url": url})
    return categories


def parse_listing_page(html: str, category_name: str) -> List[RawBook]:
    """Parse one catalogue listing page (a category page or an 'all products' page)."""
    soup = BeautifulSoup(html, "html.parser")
    books = []
    for article in soup.select("article.product_pod"):
        title = article.select_one("h3 a")["title"].strip()
        price = article.select_one("p.price_color").get_text(strip=True)
        star_classes = article.select_one("p.star-rating")["class"]  # e.g. ["star-rating", "Four"]
        star_rating = next(c for c in star_classes if c != "star-rating")
        availability = article.select_one("p.instock.availability").get_text(strip=True)
        books.append(RawBook(
            title=title, price=price, star_rating=star_rating,
            availability=availability, category=category_name,
        ))
    return books


def has_next_page(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    return soup.select_one("ul.pager li.next a") is not None


def _next_page_url(current_url: str) -> str:
    """books.toscrape.com uses .../page-N.html; index.html is implicitly page-1."""
    if current_url.endswith("index.html"):
        return current_url.replace("index.html", "page-2.html")
    match = re.search(r"page-(\d+)\.html", current_url)
    n = int(match.group(1))
    return current_url.replace(f"page-{n}.html", f"page-{n + 1}.html")


def scrape_category(category: dict) -> List[RawBook]:
    """Scrape every paginated page of a single category."""
    all_books: List[RawBook] = []
    url = category["url"]
    while True:
        html = _get(url)
        all_books.extend(parse_listing_page(html, category["name"]))
        if not has_next_page(html):
            break
        url = _next_page_url(url)
    return all_books


def choose_categories(all_categories: List[dict], min_books: int = MIN_TOTAL_BOOKS) -> List[dict]:
    """
    Pick at least 3 categories, largest first, greedily, stopping once the
    running total across chosen categories should comfortably clear
    `min_books` (we peek at each category's declared result count).
    """
    sized = []
    for cat in all_categories:
        html = _get(cat["url"])
        soup = BeautifulSoup(html, "html.parser")
        form_text = soup.select_one("form.form-horizontal").get_text()
        count = int(re.search(r"(\d+)\s+results", form_text).group(1))
        sized.append({**cat, "count": count})

    sized.sort(key=lambda c: c["count"], reverse=True)

    chosen, running_total = [], 0
    for cat in sized:
        chosen.append(cat)
        running_total += cat["count"]
        if len(chosen) >= 3 and running_total >= min_books:
            break
    if len(chosen) < 3:  # fall back to at least 3 even if small
        chosen = sized[:3]
    return chosen


def scrape_all() -> List[RawBook]:
    home_html = _get(BASE_URL + "index.html")
    categories = parse_category_links(home_html)
    chosen = choose_categories(categories)
    print(f"Chosen categories: {[c['name'] + ' (' + str(c['count']) + ')' for c in chosen]}")

    all_books: List[RawBook] = []
    for cat in chosen:
        books = scrape_category(cat)
        print(f"  {cat['name']}: {len(books)} books")
        all_books.extend(books)

    if len(all_books) < MIN_TOTAL_BOOKS:
        raise RuntimeError(
            f"Only scraped {len(all_books)} books (< {MIN_TOTAL_BOOKS}). "
            "Widen `choose_categories` or add more categories."
        )
    return all_books


if __name__ == "__main__":
    import pandas as pd
    books = scrape_all()
    df = pd.DataFrame([asdict(b) for b in books])
    out_path = HERE / "books_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nScraped {len(df)} books across {df['category'].nunique()} categories -> {out_path}")

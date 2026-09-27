"""
Offline tests for scrape.py's pure parsing functions.

These run against saved HTML fixtures (tests/fixtures/*.html) that mirror
books.toscrape.com's real markup, so the parsing logic can be verified
without any network access — useful in CI or any offline environment.

Run:
    pytest data_pipeline/tests/test_scrape.py -v
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scrape import parse_category_links, parse_listing_page, has_next_page, _next_page_url

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_parse_category_links():
    html = (FIXTURES / "home.html").read_text()
    cats = parse_category_links(html, base_url="http://books.toscrape.com/")
    assert len(cats) == 4
    names = [c["name"] for c in cats]
    assert "Mystery" in names
    mystery = next(c for c in cats if c["name"] == "Mystery")
    assert mystery["url"] == "http://books.toscrape.com/catalogue/category/books/mystery_3/index.html"


def test_parse_listing_page():
    html = (FIXTURES / "listing.html").read_text()
    books = parse_listing_page(html, "Mystery")
    assert len(books) == 2

    sharp_objects = books[0]
    assert sharp_objects.title == "Sharp Objects"
    assert sharp_objects.price == "£47.82"
    assert sharp_objects.star_rating == "Four"
    assert sharp_objects.availability == "In stock"
    assert sharp_objects.category == "Mystery"

    assert books[1].star_rating == "One"


def test_has_next_page():
    html = (FIXTURES / "listing.html").read_text()
    assert has_next_page(html) is True


def test_next_page_url():
    assert _next_page_url("http://x/mystery_3/index.html") == "http://x/mystery_3/page-2.html"
    assert _next_page_url("http://x/mystery_3/page-2.html") == "http://x/mystery_3/page-3.html"

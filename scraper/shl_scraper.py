"""
SHL Assessment Catalog Scraper
================================
Crawls https://www.shl.com/solutions/products/product-catalog/
Extracts Individual Test Solutions ONLY (ignores Pre-packaged Job Solutions).
Captures: name, url, description, duration, remote_support, adaptive_support, test_type.

Strategy:
- Page-based pagination through the catalog
- Filter by category = Individual Test Solutions
- Follow each product link and scrape detail page
- Deduplicate by URL
- Validate field presence and store as JSON
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import re
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlencode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "assessments_raw.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_DELAY = 1.2  # seconds between requests — polite crawling
MAX_RETRIES = 3


@dataclass
class Assessment:
    name: str
    url: str
    description: str = ""
    duration: Optional[int] = None          # minutes
    remote_support: str = "No"              # "Yes" | "No"
    adaptive_support: str = "No"            # "Yes" | "No"
    test_type: list[str] = field(default_factory=list)
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    category: str = "Individual Test Solutions"

    def is_valid(self) -> bool:
        return bool(self.name and self.url)


def safe_get(url: str, session: requests.Session, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    """GET with retry logic and polite delay."""
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    logger.error(f"All retries exhausted for: {url}")
    return None


def parse_test_type_badges(soup: BeautifulSoup) -> list[str]:
    """
    Extract test type letters (A, B, C, D, E, K, P, S) from detail page badges.
    SHL uses colored badge elements with single-letter codes that map to:
      A = Ability & Aptitude
      B = Biodata & Situational Judgement
      C = Competencies
      D = Development & 360
      E = Assessment Exercises
      K = Knowledge & Skills
      P = Personality & Behavior
      S = Simulations
    """
    type_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }
    badges = []
    # SHL renders test types as small colored span/div elements
    for el in soup.select(".product-catalogue__key, [class*='test-type'], [class*='type-badge']"):
        text = el.get_text(strip=True).upper()
        if text in type_map:
            badges.append(type_map[text])

    # Fallback: look for the letters in table rows or definition lists
    if not badges:
        for el in soup.find_all(string=re.compile(r'^[ABCDEKPS]$')):
            letter = el.strip()
            if letter in type_map:
                full = type_map[letter]
                if full not in badges:
                    badges.append(full)
    return badges


def parse_yes_no(soup: BeautifulSoup, label: str) -> str:
    """Extract Yes/No value for a given label in the product detail table."""
    # Try structured table rows
    for row in soup.select("tr, .product-detail__row, .catalogue__row"):
        cells = row.find_all(["td", "th", "dd", "dt", "span", "div"])
        texts = [c.get_text(strip=True) for c in cells]
        for i, text in enumerate(texts):
            if label.lower() in text.lower() and i + 1 < len(texts):
                val = texts[i + 1].strip()
                if val.lower() in ("yes", "no"):
                    return val.capitalize()
    # Fallback: regex anywhere in the page
    pattern = rf'{re.escape(label)}.*?(Yes|No)'
    match = re.search(pattern, soup.get_text(), re.IGNORECASE)
    return match.group(1).capitalize() if match else "No"


def parse_duration(soup: BeautifulSoup) -> Optional[int]:
    """Extract integer duration in minutes from detail page."""
    text = soup.get_text()
    # Common patterns: "30 minutes", "Approximately 25 mins", "Time: 40 min"
    patterns = [
        r'(\d+)\s*(?:to\s*\d+\s*)?(?:minutes|mins|min)\b',
        r'duration[:\s]+(\d+)',
        r'time[:\s]+(\d+)',
        r'approximately\s+(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def parse_description(soup: BeautifulSoup) -> str:
    """Extract clean description text from the product detail page."""
    # Priority selectors
    for sel in [
        ".product-catalogue__description",
        ".product-detail__description",
        "[class*='description']",
        "article p",
        ".entry-content p",
    ]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > 40:
                return text[:1500]  # cap length

    # Fallback: first substantial paragraph
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 80:
            return text[:1500]
    return ""


def scrape_detail_page(url: str, session: requests.Session) -> dict:
    """Scrape a single assessment detail page and return extracted fields."""
    soup = safe_get(url, session)
    if not soup:
        return {}

    return {
        "description": parse_description(soup),
        "duration": parse_duration(soup),
        "remote_support": parse_yes_no(soup, "Remote Testing"),
        "adaptive_support": parse_yes_no(soup, "Adaptive/IRT"),
        "test_type": parse_test_type_badges(soup),
    }


def scrape_catalog_page(page_url: str, session: requests.Session) -> list[dict]:
    """
    Scrape one page of the SHL catalog listing.
    Returns list of {name, url} dicts for Individual Test Solutions.
    """
    soup = safe_get(page_url, session)
    if not soup:
        return []

    items = []

    # SHL catalog uses a table or card grid for products.
    # Individual test solutions and pre-packaged solutions are in separate tabs/sections.
    # We target rows with data-type="individual" or within the correct section.

    # Strategy 1: Look for table rows in the Individual Solutions section
    # Strategy 2: Look for product cards with the right category

    # Find all assessment links — filter to product-catalog/view/* paths
    for a_tag in soup.find_all("a", href=re.compile(r"/products/product-catalog/view/")):
        name = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        if not name or not href:
            continue
        full_url = urljoin(BASE_URL, href)

        # Heuristic guard: skip obvious pre-packaged solution links
        # Pre-packaged solutions typically have "job-solution" or "sap-" in URL
        if any(skip in href.lower() for skip in ["job-solution", "sap-", "volume-"]):
            continue

        items.append({"name": name, "url": full_url})

    return items


def get_catalog_pages() -> list[str]:
    """
    Generate all paginated URLs for the Individual Test Solutions section.
    SHL catalog paginates with ?start=0&type=1 style params.
    type=1 = Individual Test Solutions
    type=2 = Pre-packaged Job Solutions (we SKIP this)
    """
    # SHL catalog filter params — type=1 for individual tests
    # Pagination: start=0, 12, 24, 36, ... (12 items per page typically)
    pages = []
    for start in range(0, 500, 12):  # up to ~42 pages to cover 377+ items
        params = urlencode({"start": start, "type": 1})
        pages.append(f"{CATALOG_URL}?{params}")
    return pages


def scrape_all(limit: Optional[int] = None) -> list[Assessment]:
    """Main orchestration: crawl all pages → deduplicate → scrape details."""
    session = requests.Session()
    session.headers.update(HEADERS)

    seen_urls: set[str] = set()
    stub_list: list[dict] = []

    logger.info("Phase 1: Crawling catalog listing pages...")
    pages = get_catalog_pages()

    consecutive_empty = 0
    for page_url in pages:
        items = scrape_catalog_page(page_url, session)
        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info("3 consecutive empty pages — stopping pagination.")
                break
            continue
        consecutive_empty = 0

        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                stub_list.append(item)

        logger.info(f"  Collected {len(stub_list)} unique assessments so far...")

        if limit and len(stub_list) >= limit:
            break

    logger.info(f"Phase 1 complete. Found {len(stub_list)} unique listing entries.")

    if len(stub_list) < 377:
        logger.warning(
            f"Only {len(stub_list)} entries found. May need to adjust pagination params. "
            "Continuing with detail scraping."
        )

    logger.info("Phase 2: Scraping detail pages...")
    assessments: list[Assessment] = []

    for i, stub in enumerate(stub_list):
        logger.info(f"  [{i+1}/{len(stub_list)}] {stub['name'][:60]}")
        details = scrape_detail_page(stub["url"], session)

        a = Assessment(
            name=stub["name"],
            url=stub["url"],
            description=details.get("description", ""),
            duration=details.get("duration"),
            remote_support=details.get("remote_support", "No"),
            adaptive_support=details.get("adaptive_support", "No"),
            test_type=details.get("test_type", []),
        )

        if a.is_valid():
            assessments.append(a)
        else:
            logger.warning(f"  Skipping invalid entry: {stub}")

    logger.info(f"Phase 2 complete. {len(assessments)} valid assessments.")
    return assessments


def save(assessments: list[Assessment], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(a) for a in assessments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} assessments → {path}")


def load(path: Path = OUTPUT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    assessments = scrape_all()
    save(assessments)
    print(f"\n✅ Scraped {len(assessments)} Individual Test Solutions")
    print(f"📁 Saved to {OUTPUT_PATH}")

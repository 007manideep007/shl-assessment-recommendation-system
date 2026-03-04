"""
Data Cleaning & Preprocessing Pipeline
========================================
Input:  data/raw/assessments_raw.json  (output of shl_scraper.py)
Output: data/processed/assessments.json

Steps:
  1. Validate mandatory fields (name, url)
  2. Normalize test_type to known vocabulary
  3. Clean description text (strip HTML artifacts, normalize whitespace)
  4. Infer missing durations via regex fallback on description
  5. Deduplicate by canonical URL
  6. Build rich text field for embedding: name + test_type + description
  7. Output clean JSON + summary stats
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "assessments_raw.json"
CLEAN_PATH = Path(__file__).parent.parent / "data" / "processed" / "assessments.json"

# Canonical test type vocabulary (from assignment doc)
VALID_TEST_TYPES = {
    "Ability & Aptitude",
    "Biodata & Situational Judgement",
    "Competencies",
    "Development & 360",
    "Assessment Exercises",
    "Knowledge & Skills",
    "Personality & Behavior",
    "Simulations",
}

# Synonym normalisation for messy scraped values
TYPE_SYNONYMS = {
    "personality": "Personality & Behavior",
    "personality & behaviour": "Personality & Behavior",
    "behaviour": "Personality & Behavior",
    "behavioral": "Personality & Behavior",
    "knowledge": "Knowledge & Skills",
    "skills": "Knowledge & Skills",
    "knowledge & skills": "Knowledge & Skills",
    "ability": "Ability & Aptitude",
    "aptitude": "Ability & Aptitude",
    "ability & aptitude": "Ability & Aptitude",
    "competency": "Competencies",
    "competencies": "Competencies",
    "simulation": "Simulations",
    "simulations": "Simulations",
    "exercise": "Assessment Exercises",
    "assessment exercises": "Assessment Exercises",
    "biodata": "Biodata & Situational Judgement",
    "situational judgement": "Biodata & Situational Judgement",
    "sjt": "Biodata & Situational Judgement",
    "development": "Development & 360",
    "360": "Development & 360",
    "development & 360": "Development & 360",
}


def normalize_test_types(raw_types: list) -> list[str]:
    """Map raw scraped test_type strings to canonical vocabulary."""
    result = []
    for t in raw_types:
        if not isinstance(t, str):
            continue
        lower = t.lower().strip()
        if lower in TYPE_SYNONYMS:
            canonical = TYPE_SYNONYMS[lower]
        elif t in VALID_TEST_TYPES:
            canonical = t
        else:
            # Try partial match
            matched = None
            for syn, canonical_val in TYPE_SYNONYMS.items():
                if syn in lower:
                    matched = canonical_val
                    break
            canonical = matched or t  # keep unknown as-is

        if canonical and canonical not in result:
            result.append(canonical)
    return result


def clean_text(text: str) -> str:
    """Strip HTML artifacts, normalize whitespace, remove boilerplate."""
    if not text:
        return ""
    # Remove HTML tags if any slipped through
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common boilerplate fragments
    boilerplate = [
        "Cookie Policy", "Privacy Policy", "Terms of Use",
        "©", "All rights reserved", "SHL Group",
    ]
    for bp in boilerplate:
        text = text.replace(bp, "")
    return text.strip()


def infer_duration(description: str) -> Optional[int]:
    """Infer duration from description text if not present in structured field."""
    patterns = [
        r'(\d+)\s*(?:to\s*\d+\s*)?(?:minutes|mins|min)\b',
        r'approximately\s+(\d+)\s*(?:minutes|mins)',
        r'takes\s+(\d+)\s*(?:minutes|mins)',
        r'(\d+)\s*-\s*minute',
        r'timed\s+(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 480:  # sanity: 1 min to 8 hours
                return val
    return None


def build_embedding_text(assessment: dict) -> str:
    """
    Construct a rich text representation for embedding.
    Combines: name, test_type labels, description.
    This text is what gets embedded into the vector store.
    """
    parts = []

    name = assessment.get("name", "")
    if name:
        parts.append(f"Assessment: {name}")

    types = assessment.get("test_type", [])
    if types:
        parts.append(f"Type: {', '.join(types)}")

    desc = assessment.get("description", "")
    if desc:
        parts.append(f"Description: {desc[:800]}")  # cap to avoid token bloat

    remote = assessment.get("remote_support", "No")
    adaptive = assessment.get("adaptive_support", "No")
    parts.append(f"Remote testing: {remote}. Adaptive testing: {adaptive}.")

    return " | ".join(parts)


def canonicalize_url(url: str) -> str:
    """Strip trailing slashes and query strings for dedup comparison."""
    url = url.strip().rstrip("/")
    url = re.sub(r'\?.*$', '', url)
    return url.lower()


def clean_pipeline(raw: list[dict]) -> list[dict]:
    """Run full cleaning pipeline on raw scraped data."""
    seen_urls: set[str] = set()
    cleaned = []

    for i, item in enumerate(raw):
        # --- Validate mandatory fields ---
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()

        if not name or not url:
            logger.warning(f"Skipping item {i}: missing name or url")
            continue

        # --- Deduplication ---
        canon_url = canonicalize_url(url)
        if canon_url in seen_urls:
            logger.debug(f"Duplicate skipped: {url}")
            continue
        seen_urls.add(canon_url)

        # --- Normalize fields ---
        description = clean_text(item.get("description", ""))
        raw_types = item.get("test_type", [])
        if isinstance(raw_types, str):
            raw_types = [raw_types]
        test_type = normalize_test_types(raw_types)

        # --- Duration ---
        duration = item.get("duration")
        if duration is not None:
            try:
                duration = int(duration)
                if not (1 <= duration <= 480):
                    duration = None
            except (ValueError, TypeError):
                duration = None
        if duration is None:
            duration = infer_duration(description)

        # --- Remote / Adaptive ---
        remote_support = item.get("remote_support", "No")
        if remote_support not in ("Yes", "No"):
            remote_support = "No"
        adaptive_support = item.get("adaptive_support", "No")
        if adaptive_support not in ("Yes", "No"):
            adaptive_support = "No"

        record = {
            "name": name,
            "url": url,
            "description": description,
            "duration": duration,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "test_type": test_type,
            "embedding_text": "",  # filled below
        }

        record["embedding_text"] = build_embedding_text(record)
        cleaned.append(record)

    return cleaned


def print_stats(data: list[dict]) -> None:
    total = len(data)
    with_desc = sum(1 for d in data if d["description"])
    with_dur = sum(1 for d in data if d["duration"] is not None)
    with_types = sum(1 for d in data if d["test_type"])
    remote_yes = sum(1 for d in data if d["remote_support"] == "Yes")
    adaptive_yes = sum(1 for d in data if d["adaptive_support"] == "Yes")

    type_dist: dict[str, int] = {}
    for d in data:
        for t in d["test_type"]:
            type_dist[t] = type_dist.get(t, 0) + 1

    logger.info("=" * 60)
    logger.info(f"Total assessments:         {total}")
    logger.info(f"With description:          {with_desc} ({with_desc/total*100:.1f}%)")
    logger.info(f"With duration:             {with_dur} ({with_dur/total*100:.1f}%)")
    logger.info(f"With test_type:            {with_types} ({with_types/total*100:.1f}%)")
    logger.info(f"Remote support = Yes:      {remote_yes}")
    logger.info(f"Adaptive support = Yes:    {adaptive_yes}")
    logger.info("Test type distribution:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {t:<40} {c}")
    logger.info("=" * 60)

    if total < 377:
        logger.error(f"❌ CRITICAL: Only {total} assessments. Need ≥377. Re-run scraper.")
    else:
        logger.info(f"✅ {total} assessments — meets ≥377 requirement.")


def run(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH) -> list[dict]:
    logger.info(f"Loading raw data from {raw_path}")
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    logger.info(f"Loaded {len(raw)} raw entries")

    cleaned = clean_pipeline(raw)
    print_stats(cleaned)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(cleaned)} clean records → {clean_path}")
    return cleaned


if __name__ == "__main__":
    run()

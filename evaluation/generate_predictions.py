"""
Test Set Prediction Generator
================================
Generates predictions for the unlabeled test set and outputs the
CSV in the exact required format:

    Query,Assessment_url
    Query 1,https://www.shl.com/...
    Query 1,https://www.shl.com/...
    ...
    Query 2,https://www.shl.com/...

Usage:
    python evaluation/generate_predictions.py \
        --test data/test.csv \
        --output submissions/predictions.csv \
        --k 10

The test CSV has a single "query" column (unlabeled, no relevant_url).
"""

import csv
import logging
import argparse
from pathlib import Path
from time import sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_CSV = Path(__file__).parent.parent / "data" / "test.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "submissions" / "predictions.csv"


def load_test_queries(csv_path: Path) -> list[str]:
    """Load unlabeled test queries from CSV."""
    queries = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (
                row.get("query") or row.get("Query") or
                row.get("question") or ""
            ).strip()
            if query and query not in queries:
                queries.append(query)

    logger.info(f"Loaded {len(queries)} test queries")
    return queries


def generate_predictions(
    queries: list[str],
    k: int = 10,
    delay: float = 0.5,
) -> list[tuple[str, str]]:
    """
    Run full recommendation pipeline on each test query.

    Args:
        queries: List of query strings
        k: Max recommendations per query
        delay: Seconds to wait between queries (rate limit LLM)

    Returns:
        List of (query, assessment_url) tuples
    """
    from retrieval.pipeline import recommend

    rows: list[tuple[str, str]] = []

    for i, query in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] Processing: '{query[:70]}'")
        try:
            result = recommend(query=query, max_results=k, min_results=5)
            assessments = result.get("recommended_assessments", [])

            for a in assessments:
                url = a.get("url", "").strip()
                if url:
                    rows.append((query, url))

            logger.info(f"  → {len(assessments)} recommendations")

        except Exception as e:
            logger.error(f"  Failed for query {i+1}: {e}")

        if i < len(queries) - 1:
            sleep(delay)  # polite pacing for LLM API

    return rows


def save_predictions(rows: list[tuple[str, str]], output_path: Path) -> None:
    """
    Save predictions in the exact required CSV format:
        Query,Assessment_url
        query_text,url1
        query_text,url2
        ...
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Assessment_url"])   # exact header from assignment
        for query, url in rows:
            writer.writerow([query, url])

    logger.info(f"Saved {len(rows)} prediction rows → {output_path}")

    # Validation
    queries_seen = set()
    for q, _ in rows:
        queries_seen.add(q)
    logger.info(f"Covered {len(queries_seen)} unique queries")
    for q in queries_seen:
        count = sum(1 for row_q, _ in rows if row_q == q)
        logger.info(f"  '{q[:60]}': {count} predictions")


def run(
    test_path: Path = TEST_CSV,
    output_path: Path = OUTPUT_CSV,
    k: int = 10,
) -> None:
    queries = load_test_queries(test_path)

    if not queries:
        logger.error("No test queries found. Check CSV format.")
        return

    logger.info(f"Generating predictions for {len(queries)} queries, k={k}")
    rows = generate_predictions(queries, k=k)

    save_predictions(rows, output_path)
    print(f"\n✅ Predictions saved to: {output_path}")
    print(f"   Total rows: {len(rows)}")
    print(f"   Queries covered: {len(set(q for q, _ in rows))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test set predictions")
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    run(test_path=args.test, output_path=args.output, k=args.k)

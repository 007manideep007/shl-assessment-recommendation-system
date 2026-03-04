"""
Evaluation Module
==================
Implements Mean Recall@K evaluation on the labeled train set.

Metric (from assignment):
    Recall@K_i = |relevant ∩ top-K| / |relevant|
    MeanRecall@K = (1/N) * sum(Recall@K_i)

Usage:
    python evaluation/evaluator.py \
        --train data/train.csv \
        --k 10 \
        --output evaluation/results.json

Evaluates BOTH:
  1. Retrieval stage (hybrid search only, no LLM)
  2. Final stage (full pipeline with LLM reranking + balancing)

This separation is required by the assignment's evaluation criteria.
"""

import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CSV = Path(__file__).parent.parent / "data" / "train.csv"
RESULTS_PATH = Path(__file__).parent / "results.json"


@dataclass
class QueryResult:
    query: str
    relevant_urls: list[str]
    retrieved_urls_retrieval_stage: list[str] = field(default_factory=list)
    retrieved_urls_final_stage: list[str] = field(default_factory=list)
    recall_at_k_retrieval: float = 0.0
    recall_at_k_final: float = 0.0


def load_train_data(csv_path: Path) -> dict[str, list[str]]:
    """
    Load labeled train data.

    Expected format:
        query,relevant_url
        "I am hiring for Java...",https://www.shl.com/...
        ...

    Returns: {query_str: [relevant_url1, relevant_url2, ...]}
    """
    query_to_urls: dict[str, list[str]] = {}

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try common column name variants
            query = (
                row.get("query") or row.get("Query") or
                row.get("question") or ""
            ).strip()
            url = (
                row.get("relevant_url") or row.get("url") or
                row.get("Assessment_url") or row.get("assessment_url") or ""
            ).strip()

            if query and url:
                if query not in query_to_urls:
                    query_to_urls[query] = []
                if url not in query_to_urls[query]:
                    query_to_urls[query].append(url)

    logger.info(f"Loaded {len(query_to_urls)} queries from train set")
    return query_to_urls


def recall_at_k(predicted_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ top-K predicted| / |relevant|
    """
    if not relevant_urls:
        return 0.0
    top_k = set(predicted_urls[:k])
    # Normalize URLs for comparison (strip trailing slash)
    top_k = {u.rstrip("/") for u in top_k}
    relevant = {u.rstrip("/") for u in relevant_urls}
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mean_recall_at_k(results: list[QueryResult], k: int, stage: str = "final") -> float:
    """Compute MeanRecall@K across all queries for a given stage."""
    if not results:
        return 0.0
    attr = f"recall_at_k_{stage}"
    scores = [getattr(r, attr, 0.0) for r in results]
    return sum(scores) / len(scores)


def evaluate_retrieval_stage(
    query_to_urls: dict[str, list[str]],
    k: int = 10,
) -> list[QueryResult]:
    """
    Evaluate retrieval stage ONLY (no LLM reranking, no balancing).
    This isolates the quality of the hybrid search engine.
    """
    from retrieval.pipeline import get_search_engine

    engine = get_search_engine()
    results = []

    for query, relevant_urls in query_to_urls.items():
        candidates = engine.search(query, top_k=k * 3)
        predicted_urls = [c["url"] for c in candidates[:k]]

        r = QueryResult(
            query=query,
            relevant_urls=relevant_urls,
            retrieved_urls_retrieval_stage=predicted_urls,
        )
        r.recall_at_k_retrieval = recall_at_k(predicted_urls, relevant_urls, k)
        results.append(r)
        logger.info(
            f"[Retrieval] Query: '{query[:50]}...' | "
            f"Recall@{k}: {r.recall_at_k_retrieval:.3f}"
        )

    return results


def evaluate_final_stage(
    query_to_urls: dict[str, list[str]],
    results: list[QueryResult],
    k: int = 10,
) -> list[QueryResult]:
    """
    Evaluate full pipeline (retrieval + LLM reranking + balancing).
    Adds final stage metrics to existing QueryResult objects.
    """
    from retrieval.pipeline import recommend

    query_to_result = {r.query: r for r in results}

    for query, relevant_urls in query_to_urls.items():
        output = recommend(query=query, max_results=k, min_results=1)
        assessments = output.get("recommended_assessments", [])
        predicted_urls = [a["url"] for a in assessments]

        if query in query_to_result:
            r = query_to_result[query]
        else:
            r = QueryResult(query=query, relevant_urls=relevant_urls)
            results.append(r)

        r.retrieved_urls_final_stage = predicted_urls
        r.recall_at_k_final = recall_at_k(predicted_urls, relevant_urls, k)

        logger.info(
            f"[Final]     Query: '{query[:50]}...' | "
            f"Recall@{k}: {r.recall_at_k_final:.3f}"
        )

    return results


def print_evaluation_report(results: list[QueryResult], k: int = 10) -> None:
    """Print a formatted evaluation report."""
    mean_retrieval = mean_recall_at_k(results, k, stage="retrieval")
    mean_final = mean_recall_at_k(results, k, stage="final")

    print("\n" + "=" * 70)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM — EVALUATION REPORT")
    print("=" * 70)
    print(f"Total queries evaluated: {len(results)}")
    print(f"K = {k}")
    print()
    print(f"{'Stage':<30} {'Mean Recall@K':>15}")
    print("-" * 50)
    print(f"{'Hybrid Retrieval Only':<30} {mean_retrieval:>14.4f}")
    print(f"{'Full Pipeline (LLM)':<30} {mean_final:>14.4f}")
    print(f"{'Improvement':<30} {(mean_final - mean_retrieval):>+14.4f}")
    print()
    print("Per-Query Results:")
    print(f"  {'Query':<50} {'Retrieval':>10} {'Final':>10}")
    print("  " + "-" * 72)
    for r in results:
        q_short = r.query[:48] + ".." if len(r.query) > 50 else r.query
        print(
            f"  {q_short:<50} "
            f"{r.recall_at_k_retrieval:>9.3f} "
            f"{r.recall_at_k_final:>9.3f}"
        )
    print("=" * 70)
    print(f"📊 Mean Recall@{k} (Retrieval): {mean_retrieval:.4f}")
    print(f"📊 Mean Recall@{k} (Full Pipeline): {mean_final:.4f}")
    print("=" * 70)


def save_results(results: list[QueryResult], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


def run_evaluation(
    train_path: Path = TRAIN_CSV,
    k: int = 10,
    output_path: Path = RESULTS_PATH,
    stages: list[str] = None,
) -> dict[str, float]:
    """
    Full evaluation run.

    Args:
        train_path: Path to labeled CSV
        k: Recall@K cutoff
        output_path: Where to save JSON results
        stages: ["retrieval", "final"] or subset

    Returns:
        {"mean_recall_retrieval": float, "mean_recall_final": float}
    """
    if stages is None:
        stages = ["retrieval", "final"]

    query_to_urls = load_train_data(train_path)
    results: list[QueryResult] = []

    if "retrieval" in stages:
        results = evaluate_retrieval_stage(query_to_urls, k=k)
    else:
        results = [
            QueryResult(query=q, relevant_urls=urls)
            for q, urls in query_to_urls.items()
        ]

    if "final" in stages:
        results = evaluate_final_stage(query_to_urls, results, k=k)

    print_evaluation_report(results, k=k)
    save_results(results, output_path)

    return {
        "mean_recall_retrieval": mean_recall_at_k(results, k, "retrieval"),
        "mean_recall_final": mean_recall_at_k(results, k, "final"),
        "num_queries": len(results),
        "k": k,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SHL Recommendation System")
    parser.add_argument("--train", type=Path, default=TRAIN_CSV, help="Labeled train CSV")
    parser.add_argument("--k", type=int, default=10, help="Recall@K cutoff")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["retrieval", "final"],
        choices=["retrieval", "final"],
        help="Which stages to evaluate",
    )
    args = parser.parse_args()

    metrics = run_evaluation(
        train_path=args.train,
        k=args.k,
        output_path=args.output,
        stages=args.stages,
    )
    print(f"\nFinal Metrics: {metrics}")

"""
Balanced Recommendation Engine
================================
Ensures multi-domain queries receive a balanced mix of test types.

Key insight from assignment:
  "if a query pertains to both behavioral and technical skills, the results
   should contain a balanced mix of assessments"

Example: "Java developer + collaboration"
  → Should return both K (Knowledge & Skills) AND P (Personality & Behavior)
  → NOT 10 technical assessments, NOT 10 personality assessments

Algorithm:
  1. Classify query into required test types (from LLM layer)
  2. Retrieve top-N candidates via hybrid search
  3. LLM re-ranks candidates
  4. Apply constraint-based balancing:
     - For multi-domain: allocate slots proportionally across primary types
     - Fill remaining slots with best-scoring items of any type
  5. Final cap: 5-10 assessments
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 10
DEFAULT_RECOMMENDATIONS = 10


def _slot_allocation(
    primary_types: list[str],
    secondary_types: list[str],
    total_slots: int,
) -> dict[str, int]:
    """
    Compute how many slots to allocate to each test type.

    Primary types share 70% of slots equally.
    Secondary types share remaining 30%.
    """
    if not primary_types:
        return {}

    primary_slots = round(total_slots * 0.65)
    secondary_slots = total_slots - primary_slots

    allocation: dict[str, int] = {}

    per_primary = max(1, primary_slots // len(primary_types))
    for t in primary_types:
        allocation[t] = per_primary

    if secondary_types:
        per_secondary = max(1, secondary_slots // len(secondary_types))
        for t in secondary_types:
            allocation[t] = per_secondary

    return allocation


def balance_recommendations(
    candidates: list[dict],
    type_mapping: dict,
    is_multi_domain: bool,
    max_results: int = DEFAULT_RECOMMENDATIONS,
    min_results: int = MIN_RECOMMENDATIONS,
) -> list[dict]:
    """
    Apply balanced selection from ranked candidates.

    Args:
        candidates: Ranked list from retrieval + LLM reranking (best first)
        type_mapping: Output of llm_layer.map_to_test_types()
        is_multi_domain: Whether query spans multiple domains
        max_results: Maximum number of recommendations (≤10)
        min_results: Minimum number of recommendations (≥1)

    Returns:
        Final balanced list of assessments (min_results ≤ len ≤ max_results)
    """
    if not candidates:
        return []

    max_results = min(max_results, MAX_RECOMMENDATIONS)
    max_results = max(max_results, min_results)

    if not is_multi_domain:
        # Single domain: just take top-N from ranked list
        result = candidates[:max_results]
        logger.info(f"Single-domain: returning top {len(result)} candidates")
        return result

    primary_types = type_mapping.get("primary_types", [])
    secondary_types = type_mapping.get("secondary_types", [])
    required_types = type_mapping.get("required_test_types", [])

    if not required_types:
        return candidates[:max_results]

    # Allocate slots
    allocation = _slot_allocation(primary_types, secondary_types, max_results)
    logger.info(f"Multi-domain slot allocation: {allocation}")

    # Group candidates by their first test type
    type_buckets: dict[str, list[dict]] = {t: [] for t in required_types}
    untyped: list[dict] = []

    for c in candidates:
        test_types = c.get("test_type", [])
        placed = False
        for t in required_types:
            if t in test_types:
                type_buckets[t].append(c)
                placed = True
                break  # place in first matching bucket only
        if not placed:
            untyped.append(c)

    # Fill slots from each bucket
    selected: list[dict] = []
    seen_urls: set[str] = set()

    def add_if_new(item: dict) -> bool:
        url = item.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            selected.append(item)
            return True
        return False

    # Primary types first
    for t in primary_types:
        quota = allocation.get(t, 1)
        added = 0
        for item in type_buckets.get(t, []):
            if added >= quota:
                break
            if add_if_new(item):
                added += 1

    # Secondary types
    for t in secondary_types:
        quota = allocation.get(t, 1)
        added = 0
        for item in type_buckets.get(t, []):
            if added >= quota:
                break
            if add_if_new(item):
                added += 1

    # Fill remaining slots with best unplaced candidates (any type)
    remaining_slots = max_results - len(selected)
    if remaining_slots > 0:
        # Merge all remaining candidates preserving score order
        all_remaining = []
        for t, bucket in type_buckets.items():
            for item in bucket:
                if item.get("url") not in seen_urls:
                    all_remaining.append(item)
        all_remaining.extend(u for u in untyped if u.get("url") not in seen_urls)

        # Sort remaining by score
        all_remaining.sort(key=lambda x: x.get("_score", 0), reverse=True)

        for item in all_remaining:
            if len(selected) >= max_results:
                break
            add_if_new(item)

    # Enforce min_results
    if len(selected) < min_results:
        for c in candidates:
            if len(selected) >= min_results:
                break
            add_if_new(c)

    logger.info(
        f"Balanced selection: {len(selected)} assessments | "
        f"types: {[t for t in required_types if any(t in a.get('test_type', []) for a in selected)]}"
    )
    return selected[:max_results]


def validate_result_balance(
    results: list[dict],
    required_types: list[str],
) -> dict:
    """
    Diagnostic: check how well the result set covers required test types.
    Returns coverage stats for evaluation/logging.
    """
    type_counts: dict[str, int] = {}
    for a in results:
        for t in a.get("test_type", []):
            type_counts[t] = type_counts.get(t, 0) + 1

    covered = [t for t in required_types if type_counts.get(t, 0) > 0]
    coverage_ratio = len(covered) / len(required_types) if required_types else 1.0

    return {
        "type_distribution": type_counts,
        "covered_types": covered,
        "missing_types": [t for t in required_types if t not in covered],
        "coverage_ratio": coverage_ratio,
        "total_results": len(results),
    }

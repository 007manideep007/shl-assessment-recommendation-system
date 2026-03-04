"""
Recommendation Pipeline Orchestrator
======================================
End-to-end pipeline:
  Input (query/JD text/URL)
    → URL fetch (if URL)
    → LLM query understanding
    → LLM test type mapping
    → Hybrid retrieval (dense + sparse)
    → LLM re-ranking
    → Balanced post-processing
    → Final JSON response

This is the single entry point used by the FastAPI server.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_engine = None
_engine_loaded = False


def get_search_engine():
    """Lazy-load the search engine (expensive operation, done once at startup)."""
    global _engine, _engine_loaded
    if not _engine_loaded:
        from embeddings.build_index import load_engine
        _engine = load_engine()
        _engine_loaded = True
    return _engine


def format_assessment(assessment: dict) -> dict:
    """
    Format a raw assessment dict into the exact API response schema.

    API schema (from assignment):
      url, name, adaptive_support, description, duration, remote_support, test_type
    """
    return {
        "url": assessment.get("url", ""),
        "name": assessment.get("name", ""),
        "adaptive_support": assessment.get("adaptive_support", "No"),
        "description": assessment.get("description", ""),
        "duration": assessment.get("duration"),      # int or None
        "remote_support": assessment.get("remote_support", "No"),
        "test_type": assessment.get("test_type", []),
    }


def recommend(
    query: str,
    max_results: int = 10,
    min_results: int = 1,
    retrieval_top_k: int = 30,
    use_llm: bool = True,
) -> dict:
    """
    Main recommendation function.

    Args:
        query: Natural language query, JD text, or URL
        max_results: Maximum assessments to return (capped at 10)
        min_results: Minimum assessments to return (at least 1)
        retrieval_top_k: How many candidates to retrieve before reranking
        use_llm: Whether to use LLM layers (disable for speed/testing)

    Returns:
        {"recommended_assessments": [...], "metadata": {...}}
    """
    from retrieval.llm_layer import (
        understand_query, map_to_test_types, llm_rerank,
        fetch_url_content, is_url
    )
    from retrieval.balancer import balance_recommendations, validate_result_balance

    # ── Step 1: Resolve input ──────────────────────────────────
    if is_url(query):
        logger.info(f"Input detected as URL — fetching content from: {query}")
        query = fetch_url_content(query)
        if not query:
            return {
                "recommended_assessments": [],
                "metadata": {"error": "Could not fetch URL content"}
            }

    logger.info(f"Processing query: {query[:100]}...")

    # ── Step 2: LLM Query Understanding ───────────────────────
    if use_llm:
        query_understanding = understand_query(query)
        type_mapping = map_to_test_types(query_understanding)
    else:
        # Minimal fallback
        query_understanding = {
            "normalized_query": query,
            "domains": [],
            "is_multi_domain": False,
            "hard_skills": [],
            "soft_skills": [],
        }
        type_mapping = {
            "required_test_types": [],
            "primary_types": [],
            "secondary_types": [],
        }

    is_multi_domain = query_understanding.get("is_multi_domain", False)
    normalized_query = query_understanding.get("normalized_query", query)

    logger.info(f"Multi-domain: {is_multi_domain} | "
               f"Types: {type_mapping.get('required_test_types', [])}")

    # ── Step 3: Hybrid Retrieval ───────────────────────────────
    engine = get_search_engine()
    candidates = engine.search(normalized_query, top_k=retrieval_top_k)
    logger.info(f"Retrieved {len(candidates)} candidates")

    if not candidates:
        return {"recommended_assessments": [], "metadata": {"warning": "No candidates found"}}

    # ── Step 4: LLM Re-ranking ─────────────────────────────────
    if use_llm and len(candidates) > 1:
        reranked = llm_rerank(
            query=query,
            query_understanding=query_understanding,
            candidates=candidates,
            top_k=max(max_results * 2, 20),
        )
    else:
        reranked = candidates

    # ── Step 5: Balanced Post-Processing ──────────────────────
    final = balance_recommendations(
        candidates=reranked,
        type_mapping=type_mapping,
        is_multi_domain=is_multi_domain,
        max_results=max_results,
        min_results=min_results,
    )

    # ── Step 6: Format Response ────────────────────────────────
    balance_stats = validate_result_balance(
        final, type_mapping.get("required_test_types", [])
    )
    logger.info(f"Balance stats: {balance_stats}")

    formatted = [format_assessment(a) for a in final]

    return {
        "recommended_assessments": formatted,
        "metadata": {
            "query_preview": query[:100],
            "is_multi_domain": is_multi_domain,
            "required_types": type_mapping.get("required_test_types", []),
            "retrieved_candidates": len(candidates),
            "final_count": len(formatted),
            "balance_stats": balance_stats,
        },
    }

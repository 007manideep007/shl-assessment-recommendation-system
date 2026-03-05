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
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None
_engine_loaded = False


# --------------------------------------------------
# Search Engine Loader
# --------------------------------------------------

def get_search_engine():
    """
    Lazy-load search engine.

    If index does not exist, automatically rebuild it.
    This prevents deployment failures on cloud platforms.
    """

    global _engine, _engine_loaded

    if not _engine_loaded:

        try:
            from embeddings.build_index import load_engine

            logger.info("Loading existing search index...")
            _engine = load_engine()

        except Exception as e:

            logger.warning(f"Search index missing or failed to load: {e}")
            logger.info("Rebuilding search index...")

            from embeddings.build_index import build_index, load_engine

            build_index()
            _engine = load_engine()

        _engine_loaded = True

    return _engine


# --------------------------------------------------
# Assessment Formatter
# --------------------------------------------------

def format_assessment(assessment: dict) -> dict:
    """
    Convert raw assessment object to API response schema.
    """

    return {
        "url": assessment.get("url", ""),
        "name": assessment.get("name", ""),
        "adaptive_support": assessment.get("adaptive_support", "No"),
        "description": assessment.get("description", ""),
        "duration": assessment.get("duration"),
        "remote_support": assessment.get("remote_support", "No"),
        "test_type": assessment.get("test_type", []),
    }


# --------------------------------------------------
# Main Recommendation Pipeline
# --------------------------------------------------

def recommend(
    query: str,
    max_results: int = 10,
    min_results: int = 1,
    retrieval_top_k: int = 20,
    use_llm: bool = True,
) -> dict:
    """
    Main recommendation pipeline.
    """

    from retrieval.llm_layer import (
        understand_query,
        map_to_test_types,
        llm_rerank,
        fetch_url_content,
        is_url,
    )

    from retrieval.balancer import (
        balance_recommendations,
        validate_result_balance,
    )

    # --------------------------------------------------
    # Step 1 — Resolve URL input
    # --------------------------------------------------

    if is_url(query):

        logger.info(f"Input detected as URL: {query}")

        query = fetch_url_content(query)

        if not query:
            return {
                "recommended_assessments": [],
                "metadata": {"error": "Failed to fetch URL content"},
            }

    logger.info(f"Processing query: {query[:100]}...")

    # --------------------------------------------------
    # Step 2 — Query understanding
    # --------------------------------------------------

    if use_llm:

        query_understanding = understand_query(query)
        type_mapping = map_to_test_types(query_understanding)

    else:

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

    normalized_query = query_understanding.get("normalized_query", query)
    is_multi_domain = query_understanding.get("is_multi_domain", False)

    logger.info(
        f"Query processed | multi_domain={is_multi_domain} | "
        f"required_types={type_mapping.get('required_test_types', [])}"
    )

    # --------------------------------------------------
    # Step 3 — Hybrid retrieval
    # --------------------------------------------------

    engine = get_search_engine()

    candidates = engine.search(normalized_query, top_k=retrieval_top_k)

    logger.info(f"Retrieved {len(candidates)} candidates")

    # fallback search if no results
    if not candidates:

        logger.warning("No candidates found — retrying with larger search window")

        candidates = engine.search(normalized_query, top_k=50)

    if not candidates:

        return {
            "recommended_assessments": [],
            "metadata": {"warning": "No candidates found"},
        }

    # --------------------------------------------------
    # Step 4 — LLM reranking
    # --------------------------------------------------

    if use_llm and len(candidates) > 1:

        reranked = llm_rerank(
            query=query,
            query_understanding=query_understanding,
            candidates=candidates,
            top_k=max(max_results * 2, 20),
        )

    else:

        reranked = candidates

    # --------------------------------------------------
    # Step 5 — Balanced recommendation selection
    # --------------------------------------------------

    final = balance_recommendations(
        candidates=reranked,
        type_mapping=type_mapping,
        is_multi_domain=is_multi_domain,
        max_results=max_results,
        min_results=min_results,
)

    # Fallback if balancing removed everything
    if not final:
        logger.warning("Balancer removed all results — falling back to raw retrieval")
        final = reranked[:max_results]

    # --------------------------------------------------
    # Step 6 — Validate result balance
    # --------------------------------------------------

    balance_stats = validate_result_balance(
        final,
        type_mapping.get("required_test_types", []),
    )

    logger.info(f"Balance stats: {balance_stats}")

    # --------------------------------------------------
    # Step 7 — Format response
    # --------------------------------------------------

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
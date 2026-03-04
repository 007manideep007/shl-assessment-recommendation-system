"""
FastAPI Backend
================
Implements exact API schema from the SHL assignment:

  GET  /health         → {"status": "healthy"}
  POST /recommend      → {"recommended_assessments": [...]}

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment:
  GEMINI_API_KEY=your_key   (for LLM features)
  MAX_RESULTS=10            (optional override)
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search engine at startup (heavy operation, done once)."""
    global _pipeline
    logger.info("🚀 Loading SHL Recommendation Engine...")
    start = time.time()

    # Import here to avoid circular imports at module level
    from retrieval.pipeline import recommend as _recommend
    _pipeline = _recommend

    logger.info(f"✅ Engine loaded in {time.time() - start:.1f}s")
    yield
    logger.info("🛑 Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description=(
        "Intelligent assessment recommendation system using RAG + LLM. "
        "Returns relevant SHL Individual Test Solutions for a given query or JD."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str

    @validator("query")
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str        # "Yes" | "No"
    description: str
    duration: Optional[int]      # minutes, can be null
    remote_support: str          # "Yes" | "No"
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]


# ── Middleware: request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration:.3f}s)"
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns {"status": "healthy"} when the service is running.
    """
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    """
    Assessment recommendation endpoint.

    Accepts a natural language query, job description text, or URL.
    Returns 1–10 relevant SHL Individual Test Solutions.

    Request body:
        {"query": "your query or JD text or URL"}

    Response:
        {"recommended_assessments": [{url, name, adaptive_support,
          description, duration, remote_support, test_type}, ...]}
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not yet initialized. Please retry."
        )

    try:
        max_results = int(os.getenv("MAX_RESULTS", "10"))
        max_results = min(max(max_results, 1), 10)

        logger.info(f"Recommendation request: query='{request.query[:80]}...'")

        result = _pipeline(
            query=request.query,
            max_results=max_results,
            min_results=1,
            retrieval_top_k=30,
            use_llm=True,
        )

        assessments = result.get("recommended_assessments", [])

        # Enforce API contract: 1 ≤ count ≤ 10
        assessments = assessments[:10]

        if not assessments:
            logger.warning("No recommendations generated")
            # Return empty list (not an error — valid response)

        return RecommendResponse(recommended_assessments=assessments)

    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ── Error handlers ────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found"})


@app.exception_handler(422)
async def validation_error(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,       # single worker: FAISS index not fork-safe
        log_level="info",
    )

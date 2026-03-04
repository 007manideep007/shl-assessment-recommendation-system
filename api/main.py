"""
FastAPI Backend
================
Implements exact API schema from the SHL assignment:

  GET  /health         → {"status": "healthy"}
  POST /recommend      → {"recommended_assessments": [...]}

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment:
  GEMINI_API_KEY=your_key
  MAX_RESULTS=10
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search engine at startup."""
    global _pipeline
    logger.info("🚀 Loading SHL Recommendation Engine...")
    start = time.time()

    from retrieval.pipeline import recommend as _recommend
    _pipeline = _recommend

    logger.info(f"✅ Engine loaded in {time.time() - start:.1f}s")
    yield
    logger.info("🛑 Shutting down")


app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG + LLM powered SHL assessment recommendation system.",
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


# ─────────────────────────────────────────────
# Serve Frontend (THIS FIXES YOUR PROBLEM)
# ─────────────────────────────────────────────

@app.get("/")
async def serve_home():
    """Serve the web UI."""
    return FileResponse("api/index.html")


@app.get("/index.html")
async def serve_index():
    return FileResponse("api/index.html")


# ─────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────

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
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]


# ─────────────────────────────────────────────
# Logging Middleware
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):

    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not yet initialized."
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
        assessments = assessments[:10]

        return RecommendResponse(recommended_assessments=assessments)

    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )


@app.exception_handler(422)
async def validation_error(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )


# ─────────────────────────────────────────────
# Dev Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,
        log_level="info",
    )
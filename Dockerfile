# ─────────────────────────────────────────────────────────────
# SHL Assessment Recommendation API — Dockerfile
# Multi-stage build for lean production image
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies for faiss, lxml, BeautifulSoup
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SBERT model during build (avoids runtime download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ─── Application code ─────────────────────────────────────────
COPY . .

# ─── Static frontend served via FastAPI ───────────────────────
# The frontend/index.html will be served at /

# ─── Runtime config ───────────────────────────────────────────
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Single worker (FAISS index not fork-safe)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

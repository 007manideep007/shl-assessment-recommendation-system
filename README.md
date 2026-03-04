# SHL Assessment Recommendation Engine

> RAG-based intelligent assessment recommendation system for SHL's product catalog.
> Built for the SHL Generative AI Intern assessment.

---

## Architecture

```
Input (query / JD text / URL)
        │
        ▼
┌───────────────────┐
│  URL Resolver     │  ← fetches JD page if URL given
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LLM Query        │  ← Gemini: extracts job title, skills,
│  Understanding    │    domains, multi-domain flag
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Test Type Mapper │  ← Gemini: maps query to SHL test types
│                   │    (A/B/C/D/E/K/P/S)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│  Hybrid Retrieval                                  │
│  ┌─────────────────┐   ┌──────────────────────┐   │
│  │ Dense (SBERT)   │ + │ Sparse (TF-IDF)      │   │
│  │ FAISS IndexFlat │   │ cosine similarity    │   │
│  │ weight: 0.70    │   │ weight: 0.30         │   │
│  └─────────────────┘   └──────────────────────┘   │
│             combined score → top-30 candidates     │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  LLM Re-ranker  │  ← Gemini re-orders top-20
         │  (Gemini)       │    based on job relevance
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Balance Filter │  ← Ensures multi-domain queries
         │                 │    get mix of K + P types, etc.
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  API Response   │  ← 1–10 assessments, exact schema
         └─────────────────┘
```

### Why these technology choices?

| Component | Choice | Reason |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, fast, excellent semantic quality for short texts |
| Vector DB | FAISS (IndexFlatIP) | Exact search, no approximation needed for <1000 docs |
| Sparse | TF-IDF (ngrams 1-2) | Catches exact skill terms missed by dense embeddings |
| Hybrid | 0.70 dense + 0.30 sparse | Tuned on train set; dense dominates but sparse adds precision |
| LLM | Gemini 1.5 Flash | Free tier (15 RPM), strong JSON instruction following |
| Backend | FastAPI | Async, Pydantic validation, auto-docs, production-grade |

---

## Setup

### Prerequisites
- Python 3.10+
- `GEMINI_API_KEY` from [Google AI Studio](https://ai.google.dev/) (free)

### Installation

```bash
git clone https://github.com/your-username/shl-recommendation
cd shl-recommendation
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Run Full Pipeline

```bash
# Step 1: Scrape SHL catalog (≥377 assessments)
python run_pipeline.py scrape

# Step 2: Clean and normalize data
python run_pipeline.py clean

# Step 3: Build embedding index
python run_pipeline.py build-index

# Step 4: Evaluate on labeled train set
python run_pipeline.py evaluate --train data/train.csv

# Step 5: Generate test predictions
python run_pipeline.py predict --test data/test.csv --output submissions/predictions.csv

# Or run everything:
python run_pipeline.py all
```

### Start API Server

```bash
python run_pipeline.py serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

---

## API Reference

### `GET /health`

```bash
curl https://your-api-url/health
```

```json
{"status": "healthy"}
```

### `POST /recommend`

```bash
curl -X POST https://your-api-url/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a Java developer who can collaborate with business teams"}'
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-new/",
      "name": "Java (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test that measures Java knowledge...",
      "duration": 15,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    },
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/...",
      "name": "Workplace Personality Inventory",
      "adaptive_support": "Yes",
      "description": "Measures personality dimensions relevant to workplace...",
      "duration": 25,
      "remote_support": "Yes",
      "test_type": ["Personality & Behavior"]
    }
  ]
}
```

**With a URL input:**
```bash
curl -X POST https://your-api-url/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "https://example.com/jobs/senior-java-developer"}'
```

---

## Evaluation Results

### Method
- **Metric:** Mean Recall@10
- **Train set:** 10 labeled queries with human-annotated relevant assessments
- Evaluation run at two stages:
  1. Retrieval only (hybrid search, no LLM)
  2. Full pipeline (retrieval + LLM reranking + balancing)

### Results

| Stage | Mean Recall@10 |
|---|---|
| Hybrid Retrieval Only | _(see evaluation/results.json)_ |
| Full Pipeline (LLM) | _(see evaluation/results.json)_ |

### Iteration Strategy

1. **Baseline:** TF-IDF only → established floor
2. **Dense:** SBERT only → significant lift on semantic queries
3. **Hybrid:** 0.7 dense + 0.3 sparse → best on train set
4. **LLM reranking:** Improved handling of multi-requirement queries
5. **Balancing:** Critical for multi-domain queries (e.g., technical + behavioral)

Run to reproduce:
```bash
python evaluation/evaluator.py --train data/train.csv --k 10
```

---

## Project Structure

```
shl-recommendation/
├── scraper/
│   ├── shl_scraper.py          # SHL catalog web crawler
│   └── data_cleaner.py         # Normalization & deduplication
├── embeddings/
│   └── build_index.py          # SBERT encoding + FAISS + TF-IDF
├── retrieval/
│   ├── llm_layer.py            # Gemini: query understanding, reranking
│   ├── balancer.py             # Multi-domain balance logic
│   └── pipeline.py             # End-to-end orchestrator
├── api/
│   └── main.py                 # FastAPI app (/health, /recommend)
├── evaluation/
│   ├── evaluator.py            # Mean Recall@K evaluation
│   └── generate_predictions.py # Test set CSV generator
├── frontend/
│   └── index.html              # Web UI
├── data/
│   ├── raw/                    # Raw scraped JSON
│   ├── processed/              # Cleaned JSON
│   └── index/                  # FAISS + TF-IDF artifacts
├── submissions/
│   └── predictions.csv         # Final test predictions
├── run_pipeline.py             # Pipeline runner CLI
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Deployment (Free Tier)

### Option A: Render.com (recommended)
1. Push to GitHub
2. Create new Web Service on Render, connect repo
3. Set `GEMINI_API_KEY` environment variable
4. Build command: `pip install -r requirements.txt`
5. Start command: `python run_pipeline.py serve`

### Option B: Railway.app
```bash
railway up
railway variables set GEMINI_API_KEY=your_key
```

### Option C: Google Cloud Run
```bash
gcloud run deploy shl-recommender \
  --source . \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key
```

> **Important:** The FAISS index must be pre-built before deploying.
> Either commit `data/index/` to the repo or build it during the Docker build step.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (free at ai.google.dev) |
| `PORT` | No | Server port (default: 8000) |
| `MAX_RESULTS` | No | Max recommendations (default: 10) |

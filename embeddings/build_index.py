"""
Lightweight Search Engine
=========================
Memory-safe version for Render free tier.

Uses TF-IDF retrieval only.
No SentenceTransformer
No FAISS
"""

import json
import logging
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "assessments.json"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


class SearchEngine:

    def __init__(self, assessments, vectorizer, matrix):

        self.assessments = assessments
        self.vectorizer = vectorizer
        self.matrix = matrix

        logger.info(f"SearchEngine loaded with {len(assessments)} assessments")

    def search(self, query: str, top_k: int = 10):

        query_vec = self.vectorizer.transform([query])

        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked = scores.argsort()[::-1][:top_k]

        results = []

        for idx in ranked:

            result = dict(self.assessments[idx])
            result["_score"] = float(scores[idx])

            results.append(result)

        return results


def build_engine():

    logger.info("Loading assessments dataset")

    with open(DATA_PATH, encoding="utf-8") as f:
        assessments = json.load(f)

    texts = [a["embedding_text"] for a in assessments]

    logger.info("Building TF-IDF index")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(texts)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(INDEX_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    with open(INDEX_DIR / "assessments.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f)

    logger.info("Index built successfully")

    return SearchEngine(assessments, vectorizer, matrix)


def load_engine():

    try:

        logger.info("Loading search index")

        with open(INDEX_DIR / "assessments.json", encoding="utf-8") as f:
            assessments = json.load(f)

        with open(INDEX_DIR / "tfidf.pkl", "rb") as f:
            data = pickle.load(f)

        vectorizer = data["vectorizer"]
        matrix = data["matrix"]

        return SearchEngine(assessments, vectorizer, matrix)

    except Exception:

        logger.warning("Index not found. Building new index...")

        return build_engine()


if __name__ == "__main__":

    engine = build_engine()

    results = engine.search("python developer", top_k=5)

    print(results)
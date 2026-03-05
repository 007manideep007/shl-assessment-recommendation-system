"""
Embedding Engine & FAISS Vector Store
=======================================
Builds a dense vector index over all SHL assessments using
sentence-transformers (all-MiniLM-L6-v2) for fast semantic search.

Also builds a TF-IDF sparse index for hybrid retrieval.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "assessments.json"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

SBERT_MODEL = "all-MiniLM-L6-v2"

DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3


class HybridSearchEngine:

    def __init__(
        self,
        assessments: list[dict],
        sbert_model: SentenceTransformer,
        faiss_index: faiss.Index,
        tfidf_vectorizer: TfidfVectorizer,
        tfidf_matrix,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
    ):
        self.assessments = assessments
        self.sbert_model = sbert_model
        self.faiss_index = faiss_index
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.texts = [a["embedding_text"] for a in assessments]

        logger.info(f"HybridSearchEngine ready with {len(assessments)} assessments")

    def _dense_scores(self, query: str, top_k: int):

        query_emb = self.sbert_model.encode([query], normalize_embeddings=True)
        query_emb = query_emb.astype(np.float32)

        scores, indices = self.faiss_index.search(
            query_emb, min(top_k * 2, len(self.assessments))
        )

        return indices[0], scores[0]

    def _sparse_scores(self, query: str):

        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        return scores

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_test_types: Optional[list[str]] = None,
    ):

        n = len(self.assessments)

        dense_indices, dense_raw_scores = self._dense_scores(query, top_k)

        dense_full = np.zeros(n, dtype=np.float32)

        for idx, score in zip(dense_indices, dense_raw_scores):
            if 0 <= idx < n:
                dense_full[idx] = float(score)

        sparse_scores = self._sparse_scores(query)

        combined = self.dense_weight * dense_full + self.sparse_weight * sparse_scores

        if filter_test_types:

            mask = np.zeros(n, dtype=bool)

            for i, a in enumerate(self.assessments):

                if any(t in a.get("test_type", []) for t in filter_test_types):
                    mask[i] = True

            combined = np.where(mask, combined, -np.inf)

        ranked_indices = np.argsort(combined)[::-1][:top_k]

        results = []

        for idx in ranked_indices:

                result = dict(self.assessments[idx])
                result["_score"] = float(combined[idx])

                results.append(result)

        return results
    

def build_faiss_index(embeddings: np.ndarray):

    d = embeddings.shape[1]

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors")

    return index


def build_engine(data_path: Path = DATA_PATH, index_dir: Path = INDEX_DIR):

    logger.info(f"Loading assessments from {data_path}")

    with open(data_path, encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Loaded {len(assessments)} assessments")

    texts = [a["embedding_text"] for a in assessments]

    logger.info(f"Loading SBERT model: {SBERT_MODEL}")

    model = SentenceTransformer(SBERT_MODEL)

    logger.info("Encoding assessment texts...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss_index = build_faiss_index(embeddings)

    logger.info("Building TF-IDF vectorizer...")

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    tfidf_matrix = tfidf.fit_transform(texts)

    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    with open(index_dir / "tfidf.pkl", "wb") as f:
        pickle.dump({"vectorizer": tfidf, "matrix": tfidf_matrix}, f)

    with open(index_dir / "assessments.json", "w", encoding="utf-8") as f:
        json.dump(assessments, f, ensure_ascii=False)

    logger.info("Search index saved")

    engine = HybridSearchEngine(assessments, model, faiss_index, tfidf, tfidf_matrix)

    return engine


def load_engine(index_dir: Path = INDEX_DIR):

    try:

        logger.info("Loading pre-built search index...")

        with open(index_dir / "assessments.json", encoding="utf-8") as f:
            assessments = json.load(f)

        model = SentenceTransformer(SBERT_MODEL)

        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        with open(index_dir / "tfidf.pkl", "rb") as f:
            tfidf_data = pickle.load(f)

        tfidf = tfidf_data["vectorizer"]
        tfidf_matrix = tfidf_data["matrix"]

        engine = HybridSearchEngine(
            assessments,
            model,
            faiss_index,
            tfidf,
            tfidf_matrix,
        )

        logger.info("Search engine loaded successfully")

        return engine

    except Exception as e:

        logger.warning(f"Index not found or corrupted: {e}")
        logger.info("Building search index automatically...")

        return build_engine()


if __name__ == "__main__":

    engine = build_engine()

    results = engine.search(
        "Java developer who collaborates with business teams",
        top_k=10,
    )

    print("\nTest query results:")

    for r in results:
        print(f"[{r['_score']:.3f}] {r['name']} | {r['test_type']}")
"""
Pipeline Runner
================
One-stop script for running all pipeline stages.

Usage:
    python run_pipeline.py scrape          # Step 1: Crawl SHL catalog
    python run_pipeline.py clean           # Step 2: Clean and process data
    python run_pipeline.py build-index     # Step 3: Build FAISS + TF-IDF index
    python run_pipeline.py evaluate        # Step 4: Evaluate on train set
    python run_pipeline.py predict         # Step 5: Generate test predictions
    python run_pipeline.py all             # Run all steps in order
    python run_pipeline.py serve           # Start API server
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def step_scrape():
    logger.info("=" * 60)
    logger.info("STEP 1: Scraping SHL Assessment Catalog")
    logger.info("=" * 60)
    from scraper.shl_scraper import scrape_all, save
    assessments = scrape_all()
    save(assessments)
    logger.info(f"✅ Scraped {len(assessments)} assessments")
    if len(assessments) < 377:
        logger.warning(f"⚠️  Only {len(assessments)} found — need ≥377")
    return len(assessments)


def step_clean():
    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning and Processing Data")
    logger.info("=" * 60)
    from scraper.data_cleaner import run
    cleaned = run()
    logger.info(f"✅ Cleaned {len(cleaned)} assessments")
    return len(cleaned)


def step_build_index():
    logger.info("=" * 60)
    logger.info("STEP 3: Building FAISS + TF-IDF Index")
    logger.info("=" * 60)
    from embeddings.build_index import build_engine
    engine = build_engine()
    logger.info("✅ Index built and saved")
    return engine


def step_evaluate(train_csv: str = "data/train.csv"):
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating on Labeled Train Set")
    logger.info("=" * 60)
    train_path = Path(train_csv)
    if not train_path.exists():
        logger.error(f"Train CSV not found: {train_path}")
        logger.info("Download the train dataset from the assignment link and place at data/train.csv")
        return None
    from evaluation.evaluator import run_evaluation
    metrics = run_evaluation(train_path=train_path)
    logger.info(f"✅ Evaluation complete: {metrics}")
    return metrics


def step_predict(test_csv: str = "data/test.csv", output_csv: str = "submissions/predictions.csv"):
    logger.info("=" * 60)
    logger.info("STEP 5: Generating Test Predictions")
    logger.info("=" * 60)
    test_path = Path(test_csv)
    if not test_path.exists():
        logger.error(f"Test CSV not found: {test_path}")
        logger.info("Download the test dataset from the assignment link and place at data/test.csv")
        return
    from evaluation.generate_predictions import run
    run(test_path=test_path, output_path=Path(output_csv))
    logger.info(f"✅ Predictions saved to {output_csv}")


def step_serve(port: int = 8000):
    logger.info("=" * 60)
    logger.info(f"Starting API server on port {port}")
    logger.info("=" * 60)
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, workers=1)


def main():
    parser = argparse.ArgumentParser(description="SHL Recommendation Pipeline Runner")
    parser.add_argument(
        "command",
        choices=["scrape", "clean", "build-index", "evaluate", "predict", "all", "serve"],
    )
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--output", default="submissions/predictions.csv")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "scrape":
        step_scrape()
    elif args.command == "clean":
        step_clean()
    elif args.command == "build-index":
        step_build_index()
    elif args.command == "evaluate":
        step_evaluate(args.train)
    elif args.command == "predict":
        step_predict(args.test, args.output)
    elif args.command == "serve":
        step_serve(args.port)
    elif args.command == "all":
        logger.info("Running full pipeline end-to-end...")
        n = step_scrape()
        if n < 377:
            logger.warning(f"Only {n} assessments scraped. Continuing anyway.")
        step_clean()
        step_build_index()
        step_evaluate(args.train)
        step_predict(args.test, args.output)
        logger.info("🎉 Full pipeline complete!")
        logger.info(f"  → Predictions: {args.output}")
        logger.info(f"  → Evaluation:  evaluation/results.json")
        logger.info(f"  → Start API:   python run_pipeline.py serve")


if __name__ == "__main__":
    main()

"""
scripts/embed_and_index.py
Reads whatever files kagglehub downloaded, cleans them, embeds with Vertex AI,
and stores in ChromaDB.

Usage:
  python scripts/embed_and_index.py --limit 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("kudiwise.indexer")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_model_name(value: str | None, default: str) -> str:
    raw = (value or default).strip()
    raw = raw.removeprefix("models/")

    legacy_models = {
        "text-embedding-004",
        "embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
        "gemini-embedding-001",
    }
    if raw in legacy_models:
        return default

    return raw


def safe_float(value: Any, default: float = 4.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return float(value)
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return ", ".join(safe_str(v) for v in value if v is not None)
    return str(value).strip()


def clean_text(value: Any, max_len: int = 300) -> str:
    text = safe_str(value)
    text = " ".join(text.split())
    return text[:max_len]


def find_data_file(folder: str, preferred: list[str]) -> Path | None:
    p = Path(folder)
    if not p.exists():
        return None

    for name in preferred:
        c = p / name
        if c.exists():
            return c

    for ext in ["*.jsonl", "*.json", "*.csv"]:
        files = sorted(p.rglob(ext))
        if files:
            return files[0]

    return None


def iter_json_records(path: Path, limit: int) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip()

    if not raw:
        return

    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for i, item in enumerate(parsed):
                    if i >= limit:
                        break
                    if isinstance(item, dict):
                        yield item
            elif isinstance(parsed, dict):
                yield parsed
            return
        except Exception:
            pass

    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def iter_csv_records(path: Path, limit: int) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            yield row


def iter_file(path: Path, limit: int) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        yield from iter_json_records(path, limit)
    elif suffix == ".csv":
        yield from iter_csv_records(path, limit)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_amazon(folder: str, limit: int) -> list[Document]:
    path = find_data_file(folder, [
        "reviews_Electronics.jsonl",
        "reviews_Books.jsonl",
        "Reviews.csv",
        "amazon_reviews.jsonl",
        "amazon_reviews.csv",
    ])
    if not path:
        logger.warning("Amazon: no file in %s", folder)
        return []

    logger.info("Amazon: reading %s", path.name)
    docs: list[Document] = []

    for r in tqdm(iter_file(path, limit), desc="Amazon", total=limit):
        try:
            title = clean_text(
                r.get("title") or r.get("name") or r.get("Summary") or "Unknown Product",
                120,
            )
            cat = r.get("main_category") or r.get("category") or r.get("product_category") or "General"
            if isinstance(cat, list):
                cat = cat[0] if cat else "General"
            cat = clean_text(cat, 60)

            rating = safe_float(r.get("average_rating") or r.get("overall") or r.get("Score") or 4.0)
            desc = clean_text(r.get("description") or r.get("Text") or r.get("review_body") or "", 220)
            item_id = safe_str(r.get("asin") or r.get("ProductId") or r.get("parent_asin") or len(docs))

            docs.append(
                Document(
                    page_content=f"{title} [{cat}] rated {rating:.1f}★. {desc}".strip(),
                    metadata={
                        "item_name": title,
                        "domain": "amazon",
                        "category": cat,
                        "avg_rating": round(rating, 2),
                        "item_id": item_id,
                    },
                )
            )
        except Exception:
            continue

    logger.info("Amazon: %d docs", len(docs))
    return docs


def load_yelp(folder: str, limit: int) -> list[Document]:
    path = find_data_file(folder, [
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_business.jsonl",
        "yelp_businesses.csv",
        "business.json",
    ])
    if not path:
        logger.warning("Yelp: no file in %s", folder)
        return []

    logger.info("Yelp: reading %s", path.name)
    docs: list[Document] = []

    for r in tqdm(iter_file(path, limit), desc="Yelp", total=limit):
        try:
            name = clean_text(r.get("name") or r.get("business_name") or "Unknown Business", 120)
            cats = r.get("categories") or r.get("category") or "Food"
            if isinstance(cats, list):
                cats = ", ".join(safe_str(x) for x in cats[:3])
            cats = clean_text(cats, 80)

            rating = safe_float(r.get("stars") or r.get("avg_stars") or r.get("rating") or 4.0)
            city = clean_text(r.get("city") or r.get("location") or "", 80)
            bid = safe_str(r.get("business_id") or len(docs))

            text = f"{name} [{cats}]"
            if city:
                text += f" {city}"
            text += f". Rated {rating:.1f}★. Affordable dining option."

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "item_name": name,
                        "domain": "yelp",
                        "category": cats,
                        "avg_rating": round(rating, 2),
                        "item_id": bid,
                    },
                )
            )
        except Exception:
            continue

    logger.info("Yelp: %d docs", len(docs))
    return docs


def load_goodreads(folder: str, limit: int) -> list[Document]:
    path = find_data_file(folder, [
        "books.csv",
        "book_data.csv",
        "goodreads_books.jsonl",
        "goodreads_books.json",
        "books_with_genres.csv",
        "book.csv",
    ])
    if not path:
        logger.warning("Goodreads: no file in %s", folder)
        return []

    logger.info("Goodreads: reading %s", path.name)
    docs: list[Document] = []

    for r in tqdm(iter_file(path, limit), desc="Goodreads", total=limit):
        try:
            title = clean_text(
                r.get("title") or r.get("book_title") or r.get("original_title") or r.get("Title") or "Unknown Book",
                140,
            )

            authors = r.get("authors") or r.get("author_name") or r.get("Authors") or r.get("author") or ""
            if isinstance(authors, list):
                authors = ", ".join(
                    a.get("name", "") if isinstance(a, dict) else safe_str(a)
                    for a in authors[:2]
                )
            authors = clean_text(authors, 120)

            rating = safe_float(r.get("average_rating") or r.get("rating") or r.get("Average Rating") or 4.0)
            desc = clean_text(r.get("description") or r.get("genres") or "", 220)
            book_id = safe_str(r.get("book_id") or r.get("work_id") or r.get("bookID") or len(docs))

            text = f'"{title}"'
            if authors:
                text += f" by {authors}"
            text += f". Rated {rating:.1f}★. {desc}"

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "item_name": title,
                        "domain": "goodreads",
                        "category": "Book",
                        "avg_rating": round(rating, 2),
                        "item_id": book_id,
                    },
                )
            )
        except Exception:
            continue

    logger.info("Goodreads: %d docs", len(docs))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Indexing
# ──────────────────────────────────────────────────────────────────────────────

def index_in_batches(docs: list[Document], store: Chroma, batch_size: int = 50, delay: float = 1.2) -> None:
    logger.info("Starting indexing loop...")

    with tqdm(total=len(docs), desc="Progress") as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]

            if i == 0:
                tqdm.write("--> Sending batch #1 to Vertex AI API & Chroma...")

            try:
                store.add_documents(batch)
            except Exception as exc:
                tqdm.write(f"Batch {i} error: {exc} — retrying in 5s...")
                time.sleep(5)
                try:
                    store.add_documents(batch)
                except Exception as e2:
                    tqdm.write(f"✕ Batch {i} failed completely: {e2}")

            pbar.update(len(batch))
            time.sleep(delay)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon", default=os.getenv("AMAZON_DATA_PATH", "data/raw/amazon"))
    parser.add_argument("--yelp", default=os.getenv("YELP_DATA_PATH", "data/raw/yelp"))
    parser.add_argument("--goodreads", default=os.getenv("GOODREADS_DATA_PATH", "data/raw/goodreads"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("INDEX_LIMIT", "50000")))
    parser.add_argument("--chroma", default=os.getenv("CHROMA_PATH", "./data/chroma_db"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "kudiwise_items"))
    parser.add_argument("--batch", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "50")))
    parser.add_argument("--delay", type=float, default=float(os.getenv("EMBED_DELAY_SEC", "1.2")))
    args = parser.parse_args()

    # Vertex AI relies on GCP Project ID configurations
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT_ID") or "zelta-ai"

    # Fall back to flagship enterprise Vertex embedding model if no env custom overrides
    embed_model = normalize_model_name(
        os.getenv("VERTEX_EMBED_MODEL") or os.getenv("GEMINI_EMBED_MODEL"),
        default="text-embedding-004",
    )

    logger.info("Using Vertex AI project: %s", project_id)
    logger.info("Using embedding model: %s", embed_model)
    logger.info("Chroma path: %s", args.chroma)
    logger.info("Collection: %s", args.collection)

    # Initialize Vertex AI native SDK client wrapper
    embeddings = VertexAIEmbeddings(
        model_name=embed_model,
        project=project_id,
    )

    # 🧪 Pre-flight Connection Sanity Check
    logger.info("Running network check: testing Vertex AI connectivity...")
    try:
        _ = embeddings.embed_query("Connectivity check")
        logger.info("✓ Vertex AI API responded successfully.")
    except Exception as e:
        logger.error("✕ Network error: Vertex AI API connection timed out or was rejected: %s", e)
        return

    store = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.chroma,
    )

    all_docs: list[Document] = []
    all_docs += load_amazon(args.amazon, args.limit)
    all_docs += load_yelp(args.yelp, args.limit)
    all_docs += load_goodreads(args.goodreads, args.limit)

    if not all_docs:
        logger.error("No docs loaded. Run download_datasets.py first.")
        return

    logger.info("Total: %d documents", len(all_docs))
    index_in_batches(all_docs, store, args.batch, args.delay)

    try:
        collection = getattr(store, "_collection")
        count = collection.count()
    except Exception:
        count = "unknown"

    logger.info("✓ ChromaDB has %s vectors at: %s", count, args.chroma)
    logger.info("Next: PYTHONPATH=. uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
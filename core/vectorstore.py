"""
core/vectorstore.py
ChromaDB initialization and retrieval helpers.
Uses GoogleGenerativeAIEmbeddings with Vertex AI for embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.cloud import storage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)

# =========================================================
# Constants
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-001"
LOCAL_DB_PATH = Path("/tmp/chroma_db")
BUCKET_NAME = "zelta-ai-data-europe"
GCS_PREFIX = "chroma_db/"

# =========================================================
# GCS Sync
# =========================================================

def sync_chroma_db_from_gcs() -> str:
    """
    Downloads ChromaDB snapshot from GCS into local ephemeral storage.
    """
    db_file = LOCAL_DB_PATH / "chroma.sqlite3"
    LOCAL_DB_PATH.mkdir(parents=True, exist_ok=True)

    if db_file.exists():
        logger.info("ChromaDB local snapshot confirmed at %s. Skipping sync.", db_file)
        return str(LOCAL_DB_PATH)

    logger.info("Starting GCS sync for ChromaDB to %s...", LOCAL_DB_PATH)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)

        download_count = 0
        for blob in blobs:
            relative_path = blob.name[len(GCS_PREFIX):] if blob.name.startswith(GCS_PREFIX) else blob.name
            if not relative_path or relative_path.endswith("/"):
                continue

            dest_path = LOCAL_DB_PATH / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest_path))
            download_count += 1

        logger.info("Successfully synchronized %d database assets.", download_count)
    except Exception as exc:
        logger.exception("Sync failed: %s", exc)

    if not db_file.exists():
        logger.error("CRITICAL: Database file still missing after sync!")

    return str(LOCAL_DB_PATH)

# =========================================================
# Helper Functions
# =========================================================

def _normalize_domain(value: Optional[str]) -> str:
    return (value or "unknown").strip().lower()

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None

def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None

def _infer_domain(metadata: Dict[str, Any], query: str, content: str) -> str:
    """
    Infer missing domains from metadata + content.
    Helps recover 'unknown' retrievals.
    """
    existing = _normalize_domain(metadata.get("domain"))
    if existing != "unknown":
        return existing

    text = f"{query} {metadata.get('item_name','')} {metadata.get('category','')} {content}".lower()

    amazon_keywords = [
        "laptop", "charger", "usb", "headset", "earbuds", "power bank",
        "phone", "calculator", "backpack", "accessory", "accessories",
        "electronics", "flash drive", "keyboard", "mouse", "cooling pad"
    ]
    goodreads_keywords = ["book", "textbook", "novel", "author", "study", "reading"]
    yelp_keywords = ["food", "restaurant", "meal", "burger", "pizza", "drink", "cafe"]
    service_keywords = ["service", "repair", "installation", "tutoring"]

    if any(k in text for k in amazon_keywords):
        return "amazon"
    if any(k in text for k in goodreads_keywords):
        return "goodreads"
    if any(k in text for k in yelp_keywords):
        return "yelp"
    if any(k in text for k in service_keywords):
        return "service"

    return "unknown"

def _ranking_score(distance: float, rating: Optional[float]) -> float:
    """Lower score = better. Slight rating boost helps ranking."""
    rating_bonus = 0.0
    if rating is not None:
        rating_bonus = min(rating, 5.0) * 0.02
    return distance - rating_bonus

# =========================================================
# Embeddings
# =========================================================

@lru_cache()
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a cached GoogleGenerativeAIEmbeddings instance using Vertex AI.
    Works with service account auth.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        vertexai=True,
        project=getattr(settings, "vertexai_project_id", "zelta-ai"),
        location=getattr(settings, "vertexai_location", "europe-west1"),
    )

# =========================================================
# Vector Store
# =========================================================

@lru_cache()
def get_vectorstore() -> Chroma:
    """Returns a cached ChromaDB instance."""
    path = sync_chroma_db_from_gcs()
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=path,
    )

# =========================================================
# Retrieval
# =========================================================

def retrieve_items(
    query: str,
    budget_ngn: Optional[int] = None,
    domains: Optional[List[str]] = None,
    k: int = 10,
    min_rating: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant items from ChromaDB.
    Uses raw distance ordering from Chroma (lower = more similar).
    """
    store = get_vectorstore()

    try:
        results = store.similarity_search_with_score(query, k=max(k * 5, 20))
    except Exception as exc:
        logger.warning("Similarity search failed: %s", exc)
        return []

    logger.info("DEBUG QUERY: %s", query)
    items = []
    allowed_domains = {_normalize_domain(d) for d in domains} if domains else None

    for idx, (doc, distance) in enumerate(results):
        metadata = doc.metadata or {}
        content = doc.page_content or ""
        logger.info("Result %d | distance=%.4f | metadata=%s", idx+1, distance, metadata)

        item_domain = _infer_domain(metadata, query, content)
        if allowed_domains and item_domain not in allowed_domains:
            continue

        item_price = _safe_int(metadata.get("price_ngn"))
        if budget_ngn is not None and item_price is not None and item_price > budget_ngn * 1.3:
            continue

        item_rating = _safe_float(metadata.get("avg_rating"))
        if min_rating is not None and item_rating is not None and item_rating < min_rating:
            continue

        items.append({
            "item_name": metadata.get("item_name", content[:60]),
            "domain": item_domain,
            "category": metadata.get("category", "general"),
            "avg_rating": item_rating,
            "price_ngn": item_price,
            "page_content": content,
            "distance": float(distance),
            "ranking_score": _ranking_score(float(distance), item_rating),
        })

    # Deduplicate and sort by ranking score
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = item["item_name"].strip().lower()
        existing = deduped.get(key)
        if existing is None or item["ranking_score"] < existing["ranking_score"]:
            deduped[key] = item

    sorted_items = sorted(deduped.values(), key=lambda x: (x["ranking_score"], x["distance"]))
    logger.info("Retrieved %d items for query: %s", len(sorted_items), query[:60])

    return sorted_items[:k]
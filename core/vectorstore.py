"""
core/vectorstore.py
ChromaDB initialization and retrieval helpers.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from google.cloud import storage
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "gemini-embedding-001"
LOCAL_DB_PATH = "/tmp/chroma_db"
BUCKET_NAME = "zelta-ai-data-europe"
GCS_PREFIX = "chroma_db/"

def sync_chroma_db_from_gcs() -> str:
    """Downloads ChromaDB snapshot from GCS to local ephemeral storage."""
    db_file = os.path.join(LOCAL_DB_PATH, "chroma.sqlite3")

    if os.path.exists(db_file):
        logger.info("ChromaDB local snapshot confirmed at %s. Skipping sync.", db_file)
        return LOCAL_DB_PATH

    logger.info("Starting GCS sync for ChromaDB to %s...", LOCAL_DB_PATH)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)

        download_count = 0
        for blob in blobs:
            # Flatten path: e.g., 'services/chroma_db/chroma.sqlite3' -> 'chroma.sqlite3'
            relative_path = blob.name.replace(GCS_PREFIX, "")
            if not relative_path or relative_path.endswith("/"):
                continue

            dest_path = os.path.join(LOCAL_DB_PATH, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            download_count += 1

        logger.info("Successfully synchronized %d database assets.", download_count)
    except Exception as exc:
        logger.error("Sync failed: %s", exc)

    if not os.path.exists(db_file):
        logger.error("CRITICAL: Database file still missing after sync!")

    return LOCAL_DB_PATH

@lru_cache()
def get_embeddings() -> VertexAIEmbeddings:
    """Returns a cached Vertex AI embedding model instance."""
    settings = get_settings()
    return VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=getattr(settings, "vertexai_project_id", "zelta-ai"),
    )

@lru_cache()
def get_vectorstore() -> Chroma:
    """Returns a cached ChromaDB instance."""
    path = sync_chroma_db_from_gcs()
    return Chroma(
        collection_name=get_settings().chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=path,
    )

def retrieve_items(
    query: str,
    budget_ngn: Optional[int] = None,
    domains: Optional[List[str]] = None,
    k: int = 10,
    min_rating: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB broadly, then apply post-retrieval filtering for robustness.
    """
    store = get_vectorstore()

    try:
        # Fetch a larger candidate pool to avoid missing relevant items
        results = store.similarity_search_with_score(query, k=max(k * 5, 20))
    except Exception as exc:
        logger.warning("Similarity search failed: %s", exc)
        return []

    items = []
    for doc, score in results:
        meta = doc.metadata or {}

        # Soft filter: domain
        item_domain = (meta.get("domain") or "unknown").lower()
        if domains and item_domain not in {d.lower() for d in domains}:
            continue

        # Filter: budget
        item_price = meta.get("price_ngn")
        if budget_ngn is not None and item_price is not None and item_price > budget_ngn * 1.3:
            continue

        # Filter: rating
        item_rating = meta.get("avg_rating")
        if min_rating is not None and item_rating is not None and item_rating < min_rating:
            continue

        items.append({
            "item_name": meta.get("item_name", doc.page_content[:60]),
            "domain": item_domain,
            "category": meta.get("category", "general"),
            "avg_rating": item_rating,
            "price_ngn": item_price,
            "page_content": doc.page_content,
            "distance": float(score),  # lower is more similar
        })

    # Deduplicate and sort by Chroma distance ascending (more relevant first)
    unique_items = {item["item_name"]: item for item in items}.values()
    sorted_items = sorted(unique_items, key=lambda x: x["distance"])

    logger.info("Retrieved %d items for query: %s", len(sorted_items), query[:60])
    return list(sorted_items)[:k]
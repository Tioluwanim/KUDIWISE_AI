"""
core/vectorstore.py
ChromaDB initialisation and retrieval helpers.
Singleton pattern — one client shared across the app via Vertex AI.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Optional

import chromadb
from google.cloud import storage
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)


def sync_chroma_db_from_gcs() -> str:
    """
    Downloads ChromaDB snapshot from GCS to local container memory (/tmp).
    Bypasses gcsfuse OutOfOrderError by serving SQLite purely from memory/ephemeral disk.
    """
    local_db_path = "/tmp/chroma_db"

    # Prevent re-downloading if files are already synced in this instance lifecycle
    if os.path.exists(os.path.join(local_db_path, "chroma.sqlite3")):
        logger.info("ChromaDB local snapshot already exists at %s. Skipping sync.", local_db_path)
        return local_db_path

    logger.info("Initializing ChromaDB sync from Google Cloud Storage bucket...")
    try:
        storage_client = storage.Client()
        # Hardcoded to match your healthy regional bucket
        bucket_name = "zelta-ai-data-europe"
        prefix = "services/chroma_db/"

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        download_count = 0
        for blob in blobs:
            # Reconstruct the inner file hierarchy relative to the base directory
            relative_path = blob.name.replace(prefix, "")
            if not relative_path or relative_path.endswith("/"):
                continue

            dest_path = os.path.join(local_db_path, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            logger.info("Downloading cloud asset: %s -> %s", blob.name, dest_path)
            blob.download_to_filename(dest_path)
            download_count += 1

        if download_count == 0:
            logger.warning("Zero objects found with prefix '%s' in bucket '%s'!", prefix, bucket_name)
        else:
            logger.info("Successfully synchronized %d database assets to ephemeral environment.", download_count)

    except Exception as exc:
        logger.exception("Fatal fallback: Unable to download database from GCS. Attempting local startup. Error: %s", exc)

    return local_db_path


@lru_cache()
def get_embeddings() -> VertexAIEmbeddings:
    """Return a cached Vertex AI embedding model instance."""
    settings = get_settings()

    # Extract project ID dynamically from settings, fallback safely if not explicitly declared
    project_id = (
        getattr(settings, "vertexai_project_id", None)
        or getattr(settings, "google_cloud_project", None)
        or "zelta-ai"
    )

    # Fallback to the flagship enterprise embedding model if the config still points to AI Studio names
    raw_model = getattr(settings, "vertex_embed_model", None) or settings.gemini_embed_model
    model_name = "text-embedding-004" if "gemini" in raw_model or "embedding-001" in raw_model else raw_model

    logger.info("Initializing VertexAIEmbeddings with model: %s on project: %s", model_name, project_id)

    return VertexAIEmbeddings(
        model_name=model_name,
        project=project_id,
    )


@lru_cache()
def get_vectorstore() -> Chroma:
    """Return a cached ChromaDB Chroma instance."""
    settings = get_settings()
    embeddings = get_embeddings()

    # Secure and download snapshots to the local ephemeral system before client creation
    local_persist_path = sync_chroma_db_from_gcs()

    store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=local_persist_path,
    )
    logger.info("ChromaDB vectorstore loaded via Local Writable Storage: %s", local_persist_path)
    return store


def retrieve_items(
    query: str,
    budget_ngn: Optional[int] = None,
    domains: Optional[list[str]] = None,
    k: int = 10,
    min_rating: Optional[float] = None,
) -> list[dict]:
    """
    Embed `query` via Vertex AI, search ChromaDB, apply optional metadata filters.
    Returns a list of dicts with keys: item_name, domain, category,
    avg_rating, page_content.
    """
    store = get_vectorstore()

    # Build ChromaDB where filter
    where: dict = {}
    conditions = []

    if domains:
        conditions.append({"domain": {"$in": domains}})
    if min_rating is not None:
        conditions.append({"avg_rating": {"$gte": min_rating}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        results = store.similarity_search_with_score(
            query,
            k=k,
            filter=where if where else None,
        )
    except Exception as exc:
        logger.warning("ChromaDB retrieval error: %s — falling back to unfiltered", exc)
        results = store.similarity_search_with_score(query, k=k)

    items = []
    for doc, score in results:
        meta = doc.metadata or {}
        # Apply budget filter post-retrieval (ChromaDB numeric filters can be tricky)
        item_price = meta.get("price_ngn")
        if budget_ngn and item_price and item_price > budget_ngn * 1.3:
            continue  # skip items over 130% of budget
        items.append({
            "item_name": meta.get("item_name", doc.page_content[:60]),
            "domain": meta.get("domain", "amazon"),
            "category": meta.get("category"),
            "avg_rating": meta.get("avg_rating"),
            "price_ngn": item_price,
            "page_content": doc.page_content,
            "similarity_score": round(1 - score, 3),
        })

    logger.info("Retrieved %d items for query: %s", len(items), query[:60])
    return items[:k]
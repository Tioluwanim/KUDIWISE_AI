"""
core/vectorstore.py
ChromaDB initialisation and retrieval helpers.
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Optional

from google.cloud import storage
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)


def sync_chroma_db_from_gcs() -> str:
    """
    Downloads ChromaDB snapshot from GCS to local container memory (/tmp).
    Ensures chroma.sqlite3 exists before returning the path.
    """
    local_db_path = "/tmp/chroma_db"
    db_file = os.path.join(local_db_path, "chroma.sqlite3")

    # If already synced in this lifecycle, return immediately
    if os.path.exists(db_file):
        logger.info("ChromaDB local snapshot found at %s. Skipping sync.", db_file)
        return local_db_path

    logger.info("Starting GCS sync for ChromaDB...")
    try:
        storage_client = storage.Client()
        bucket_name = "zelta-ai-data-europe"
        prefix = "services/chroma_db/"

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        download_count = 0
        for blob in blobs:
            # Reconstruct the inner file hierarchy relative to the base directory
            # If blob is 'services/chroma_db/chroma.sqlite3', relative is 'chroma.sqlite3'
            relative_path = blob.name.replace(prefix, "")

            # Skip empty keys or folder markers
            if not relative_path or relative_path.endswith("/"):
                continue

            dest_path = os.path.join(local_db_path, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            blob.download_to_filename(dest_path)
            download_count += 1

        if download_count > 0:
            logger.info("Successfully synchronized %d database assets.", download_count)
        else:
            logger.warning("No files found in GCS bucket '%s' with prefix '%s'", bucket_name, prefix)

    except Exception as exc:
        logger.error("Sync failed: %s", exc)

    # Verification
    if os.path.exists(db_file):
        logger.info("Sync verified: Database file exists at %s", db_file)
    else:
        logger.error("CRITICAL: Database file still missing after sync attempt!")

    return local_db_path


@lru_cache()
def get_embeddings() -> VertexAIEmbeddings:
    """Return a cached Vertex AI embedding model instance."""
    settings = get_settings()
    project_id = getattr(settings, "vertexai_project_id", "zelta-ai")
    model_name = "text-embedding-004"

    logger.info("Initializing VertexAIEmbeddings with model: %s", model_name)
    return VertexAIEmbeddings(
        model_name=model_name,
        project=project_id,
    )


@lru_cache()
def get_vectorstore() -> Chroma:
    """Return a cached ChromaDB Chroma instance."""
    settings = get_settings()
    embeddings = get_embeddings()

    # Sync files to local ephemeral storage
    local_persist_path = sync_chroma_db_from_gcs()

    # If the sync failed, warn that an empty collection will be created
    if not os.path.exists(os.path.join(local_persist_path, "chroma.sqlite3")):
        logger.warning("ChromaDB path is empty. This will instantiate an empty collection.")

    store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=local_persist_path,
    )

    logger.info("ChromaDB vectorstore loaded from: %s", local_persist_path)
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
        # Apply budget filter post-retrieval
        item_price = meta.get("price_ngn")
        if budget_ngn and item_price and item_price > budget_ngn * 1.3:
            continue

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
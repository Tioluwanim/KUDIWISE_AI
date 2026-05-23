"""
core/vectorstore.py
ChromaDB initialisation and retrieval helpers.
Singleton pattern — one client shared across the app via Vertex AI.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

from core.config import get_settings

logger = logging.getLogger(__name__)


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
    store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )
    logger.info("ChromaDB vectorstore loaded via Vertex AI: %s", settings.chroma_path)
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
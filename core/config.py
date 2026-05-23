"""
core/config.py
Central settings loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Google AI
    google_api_key: str = ""

    # ChromaDB
    chroma_path: str = "./data/chroma_db"
    chroma_collection: str = "kudiwise_items"

    # Gemini models
    gemini_llm_model: str = "gemini-1.5-flash"
    gemini_embed_model: str = "models/text-embedding-004"

    # Retrieval
    retrieval_top_k: int = 10
    cold_start_min_rating: float = 3.5

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
